#!/usr/bin/env python3
import sys, argparse, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from joblib import load
from torchvision.datasets import STL10
from skimage.feature import local_binary_pattern

from src.preprocess import to_gray_uint8

# ---------- util: regiones SPM (1x1 + 2x2) ----------
def spm_regions(img_pil):
    W, H = img_pil.size  # (96,96)
    # full
    regs = [img_pil]
    # quadrants (2x2)
    regs += [
        img_pil.crop((0,     0,     W//2, H//2)),  # TL
        img_pil.crop((W//2,  0,     W,    H//2)),  # TR
        img_pil.crop((0,     H//2,  W//2, H   )),  # BL
        img_pil.crop((W//2,  H//2,  W,    H   )),  # BR
    ]
    return regs  # len=5

# ---------- SIFT denso + RootSIFT ----------
def dense_keypoints(h, w, step=6, sizes=(12,16)):
    import cv2
    kps = []
    for s in sizes:
        for y in range(s//2, h - s//2, step):
            for x in range(s//2, w - s//2, step):
                kps.append(cv2.KeyPoint(float(x), float(y), float(s)))
    return kps

def compute_dense_rootsift(gray_uint8, sift, step=6, sizes=(12,16)):
    import cv2
    kps = dense_keypoints(gray_uint8.shape[0], gray_uint8.shape[1], step=step, sizes=sizes)
    kps, desc = sift.compute(gray_uint8, kps)
    if desc is None or len(desc) == 0:
        return np.empty((0,128), dtype=np.float32)
    desc = desc.astype(np.float32)
    eps = 1e-12
    desc /= (desc.sum(axis=1, keepdims=True) + eps)  # L1
    desc = np.sqrt(desc)                              # RootSIFT
    return desc

# ---------- preproc gris opcional ----------
def gray_for_sift(img_pil, use_clahe=False, sigma=0.0):
    import cv2
    g = to_gray_uint8(img_pil)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    if sigma and sigma > 0:
        g = cv2.GaussianBlur(g, ksize=(0,0), sigmaX=float(sigma), sigmaY=float(sigma))
    return g

# ---------- Fisher Vector encoding ----------
def fisher_vector_encode(desc_d, gmm):
    """
    Codifica descriptores usando Fisher Vector.
    
    Args:
        desc_d: descriptores (N, D)
        gmm: modelo GMM ya entrenado con atributos:
             - means_: (K, D) - medias de las gaussianas
             - covariances_: (K, D) - varianzas diagonales
             - weights_: (K,) - pesos de cada componente
    
    Returns:
        Fisher Vector normalizado de dimensión 2*K*D
    """
    if desc_d.size == 0:
        K, D = gmm.means_.shape
        return np.zeros(2 * K * D, dtype=np.float32)
    
    K, D = gmm.means_.shape
    N = desc_d.shape[0]
    
    # Predecir probabilidades posteriores: gamma(i,k) = P(k|x_i)
    posteriors = gmm.predict_proba(desc_d)  # (N, K)
    
    # Inicializar gradientes de primer y segundo orden
    grad_means = np.zeros((K, D), dtype=np.float32)
    grad_vars = np.zeros((K, D), dtype=np.float32)
    
    # Extraer parámetros del GMM
    means = gmm.means_.astype(np.float32)  # (K, D)
    variances = gmm.covariances_.astype(np.float32)  # (K, D) asumiendo diagonal
    weights = gmm.weights_.astype(np.float32)  # (K,)
    
    # Calcular gradientes para cada componente
    for k in range(K):
        # Normalización por peso
        sqrt_weight = np.sqrt(weights[k]) + 1e-12
        
        # Gradiente de primer orden (medias)
        diff = desc_d - means[k]  # (N, D)
        weighted_diff = posteriors[:, k:k+1] * diff  # (N, D)
        grad_means[k] = weighted_diff.sum(axis=0) / (sqrt_weight * np.sqrt(variances[k]) + 1e-12)
        
        # Gradiente de segundo orden (varianzas)
        normalized_diff_sq = (diff ** 2) / (variances[k] + 1e-12)  # (N, D)
        grad_vars[k] = (posteriors[:, k:k+1] * (normalized_diff_sq - 1)).sum(axis=0) / (sqrt_weight * np.sqrt(2) + 1e-12)
    
    # Concatenar ambos gradientes
    fv = np.concatenate([grad_means.flatten(), grad_vars.flatten()], dtype=np.float32)
    
    # Power normalization (mejora resultados empíricamente)
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-12)
    
    # L2 normalization
    fv /= (np.linalg.norm(fv) + 1e-12)
    
    return fv

# ---------- HSV+LBP global (315 dims) opcional ----------
def color_lbp_feats(img_pil):
    import cv2
    img = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv],[0],None,[128],[0,180]).flatten()
    s = cv2.calcHist([hsv],[1],None,[64],[0,256]).flatten()
    v = cv2.calcHist([hsv],[2],None,[64],[0,256]).flatten()
    hsv_hist = np.concatenate([h,s,v]).astype("float32"); hsv_hist /= (hsv_hist.sum()+1e-12)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist,_ = np.histogram(lbp, bins=np.arange(0,60), range=(0,59))
    lbp_hist = lbp_hist.astype("float32"); lbp_hist /= (lbp_hist.sum()+1e-12)
    return np.concatenate([hsv_hist, lbp_hist], dtype=np.float32)  # 315

def extract_split_spm(ds, pca, gmm, step=6, sizes=(12,16),
                      limit=None, use_clahe=False, sigma=0.0,
                      add_colorlbp=False):
    import cv2
    sift = cv2.SIFT_create()
    X, Y = [], []
    n = len(ds) if limit is None else min(limit, len(ds))
    t0 = time.time()
    for i in tqdm(range(n), ncols=100):
        img_pil, y = ds[i]
        Vs = []
        for reg in spm_regions(img_pil):
            gray = gray_for_sift(reg, use_clahe, sigma)
            desc = compute_dense_rootsift(gray, sift, step=step, sizes=sizes)
            if desc.shape[0] == 0:
                K, D = gmm.means_.shape
                V = np.zeros(2 * K * D, dtype=np.float32)
            else:
                desc_p = pca.transform(desc)
                V = fisher_vector_encode(desc_p, gmm)
            Vs.append(V)  # cada V ya está normalizado
        V_concat = np.concatenate(Vs, dtype=np.float32)  # (5 * 2*K*d,)
        # L2 final de todo el vector concatenado
        V_concat /= (np.linalg.norm(V_concat) + 1e-12)
        if add_colorlbp:
            V_concat = np.concatenate([V_concat, color_lbp_feats(img_pil)], dtype=np.float32)
        X.append(V_concat); Y.append(y)
    t1 = time.time()
    X = np.stack(X); Y = np.array(Y)
    return X, Y, (t1 - t0), (t1 - t0)/n

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fisher Vector + SPM (1x1 + 2x2) para STL-10")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--pca_path", type=str, default=None)
    ap.add_argument("--gmm_path", type=str, default=None, 
                    help="Ruta al modelo GMM entrenado (.joblib)")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--sizes", type=int, nargs="+", default=[12,16])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--pre_blur_sigma", type=float, default=0.0)
    ap.add_argument("--add_colorlbp", action="store_true",
                    help="agrega HSV+LBP global (315 dims)")
    ap.add_argument("--force_high_dim", action="store_true",
                    help="Permite dimensiones > 4096 (puede afectar rendimiento)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT/"data")
    ART  = ROOT/"artifacts"
    FEAT = ROOT/"features"; FEAT.mkdir(exist_ok=True)

    # Buscar archivos automáticamente si no se especifican
    pca_path = Path(args.pca_path) if args.pca_path else (ART/"pca_sift24.joblib")
    gmm_path = Path(args.gmm_path) if args.gmm_path else (ART/"gmm_fisher32.joblib")
    
    # Verificar que existan los archivos
    if not pca_path.exists():
        print(f"❌ No se encontró PCA: {pca_path}")
        print("\n📋 Archivos .joblib disponibles en artifacts/:")
        for f in sorted(ART.glob("*.joblib")):
            print(f"   - {f.name}")
        sys.exit(1)
    
    if not gmm_path.exists():
        print(f"❌ No se encontró GMM: {gmm_path}")
        print("\n📋 Archivos .joblib disponibles en artifacts/:")
        for f in sorted(ART.glob("*.joblib")):
            print(f"   - {f.name}")
        sys.exit(1)

    print(f"📂 Cargando modelos...")
    print(f"   PCA: {pca_path.name}")
    print(f"   GMM: {gmm_path.name}")
    
    pca = load(pca_path)
    gmm = load(gmm_path)
    
    K, d = gmm.means_.shape
    base_dim = 2 * K * d  # Fisher Vector es 2*K*d
    spm_dim  = base_dim * 5  # 1x1 + 2x2
    total_dim = spm_dim + (315 if args.add_colorlbp else 0)
    
    print(f"\n📊 Dimensiones:")
    print(f"   - Componentes GMM (K): {K}")
    print(f"   - Dim después de PCA (d): {d}")
    print(f"   - Dim Fisher Vector por región (2*K*d): {base_dim}")
    print(f"   - Regiones SPM: 5 (1 global + 4 cuadrantes)")
    print(f"   - Dim SPM total (5 * 2*K*d): {spm_dim}")
    if args.add_colorlbp:
        print(f"   - Color+LBP: 315")
    print(f"   - Dimensión final: {total_dim}")
    
    # Validación de dimensión
    if total_dim > 4096:
        if not args.force_high_dim:
            print(f"\n⚠️  ADVERTENCIA: Dimensión final {total_dim} > 4096")
            print("\n💡 Opciones para reducir la dimensión:")
            print(f"   1. Reducir componentes GMM: K={K} → K={K//2} (dim ≈ {total_dim//2})")
            print(f"   2. Reducir componentes PCA: d={d} → d={d//2} (dim ≈ {total_dim//2})")
            print(f"   3. Usar SPM más simple (solo 1x1, sin cuadrantes): dim = {base_dim}")
            print(f"   4. Forzar dimensión alta: --force_high_dim")
            print("\n📝 Ejemplo con K=16 y d=24:")
            ejemplo_dim = 2 * 16 * 24 * 5
            print(f"   python scripts/train_gmm_fisher.py --n_components 16")
            print(f"   → Dimensión resultante: {ejemplo_dim}")
            
            print(f"\n❌ Para continuar con dim={total_dim}, usa: --force_high_dim")
            sys.exit(1)
        else:
            print(f"\n✓ Usando dimensión alta ({total_dim}) con --force_high_dim")

    ds_tr = STL10(root=str(DATA), split="train", download=False)
    ds_te = STL10(root=str(DATA), split="test",  download=False)

    print(f"\n🔧 Extrayendo Fisher Vectors...")
    print(f"   Dataset: STL-10")
    print(f"   Train: {len(ds_tr)} imágenes")
    print(f"   Test: {len(ds_te)} imágenes")

    Xtr, ytr, t_train, tpi_train = extract_split_spm(
        ds_tr, pca, gmm, step=args.step, sizes=tuple(args.sizes),
        limit=args.limit, use_clahe=args.clahe, sigma=args.pre_blur_sigma,
        add_colorlbp=args.add_colorlbp
    )
    np.save(FEAT/"X_train_fisher.npy", Xtr); np.save(FEAT/"y_train.npy", ytr)
    print(f"\n[TRAIN] {len(ytr)} imgs | total = {t_train:.2f}s | img = {tpi_train:.4f}s")
    print(f"        Shape: {Xtr.shape} | Tamaño: {Xtr.nbytes/(1024*1024):.1f} MB")

    Xte, yte, t_test, tpi_test = extract_split_spm(
        ds_te, pca, gmm, step=args.step, sizes=tuple(args.sizes),
        limit=args.limit, use_clahe=args.clahe, sigma=args.pre_blur_sigma,
        add_colorlbp=args.add_colorlbp
    )
    np.save(FEAT/"X_test_fisher.npy", Xte); np.save(FEAT/"y_test.npy", yte)
    print(f"[TEST ] {len(yte)} imgs | total = {t_test:.2f}s | img = {tpi_test:.4f}s")
    print(f"        Shape: {Xte.shape} | Tamaño: {Xte.nbytes/(1024*1024):.1f} MB")
    
    print("\n✅ Listo. Fisher Vectors guardados en 'features/'.")
    print(f"   - X_train_fisher.npy: {Xtr.shape}")
    print(f"   - X_test_fisher.npy: {Xte.shape}")
