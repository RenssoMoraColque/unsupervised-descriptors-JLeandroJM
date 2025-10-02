#!/usr/bin/env python3
"""
Extrae Fisher Vectors + SPM para STL-10
Similar a extract_vlad_spm.py pero usando GMM en lugar de K-means
"""
import sys, argparse, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from joblib import load
from torchvision.datasets import STL10
import cv2

from src.preprocess import to_gray_uint8

# ---------- SPM regions ----------
def spm_regions(img_pil):
    """1x1 + 2x2 = 5 regiones"""
    W, H = img_pil.size
    regs = [img_pil]
    regs += [
        img_pil.crop((0,     0,     W//2, H//2)),
        img_pil.crop((W//2,  0,     W,    H//2)),
        img_pil.crop((0,     H//2,  W//2, H)),
        img_pil.crop((W//2,  H//2,  W,    H)),
    ]
    return regs

# ---------- SIFT denso ----------
def dense_keypoints(h, w, step=6, sizes=(12,16)):
    kps = []
    for s in sizes:
        for y in range(s//2, h - s//2, step):
            for x in range(s//2, w - s//2, step):
                kps.append(cv2.KeyPoint(float(x), float(y), float(s)))
    return kps

def compute_dense_rootsift(gray_uint8, sift, step=6, sizes=(12,16)):
    kps = dense_keypoints(gray_uint8.shape[0], gray_uint8.shape[1], step=step, sizes=sizes)
    kps, desc = sift.compute(gray_uint8, kps)
    if desc is None or len(desc) == 0:
        return np.empty((0,128), dtype=np.float32)
    desc = desc.astype(np.float32)
    eps = 1e-12
    desc /= (desc.sum(axis=1, keepdims=True) + eps)  # L1
    desc = np.sqrt(desc)  # RootSIFT
    return desc

# ---------- Fisher Vector encoding ----------
def fisher_vector(desc_d, gmm):
    """
    Calcula Fisher Vector mejorado usando GMM
    desc_d: (N, d) descriptores después de PCA
    gmm: modelo GaussianMixture de sklearn
    
    Retorna vector de dimensión 2*K*d
    """
    if desc_d.size == 0 or desc_d.shape[0] == 0:
        K = gmm.n_components
        d = gmm.means_.shape[1]
        return np.zeros(2*K*d, dtype=np.float32)
    
    # Posteriors: P(k|x) para cada descriptor
    posteriors = gmm.predict_proba(desc_d)  # (N, K)
    
    K = gmm.n_components
    d = gmm.means_.shape[1]
    
    # Pesos, medias y precisiones del GMM
    w = gmm.weights_  # (K,)
    mu = gmm.means_   # (K, d)
    
    # Diagonal covariance
    if gmm.covariance_type == 'diag':
        sigma = gmm.covariances_  # (K, d)
    else:
        # Si es 'full', extraer diagonal
        sigma = np.array([np.diag(gmm.covariances_[k]) for k in range(K)])
    
    # Inicializar gradientes
    g_mu = np.zeros((K, d), dtype=np.float32)
    g_sigma = np.zeros((K, d), dtype=np.float32)
    
    # Calcular gradientes
    for k in range(K):
        # Diferencias normalizadas
        diff = desc_d - mu[k]  # (N, d)
        diff_norm = diff / (np.sqrt(sigma[k]) + 1e-12)  # (N, d)
        
        # Ponderadas por posterior
        weighted = posteriors[:, k:k+1] * diff_norm  # (N, d)
        g_mu[k] = weighted.sum(axis=0)
        
        # Para sigma (power normalization ayuda)
        weighted_sigma = posteriors[:, k:k+1] * (diff_norm**2 - 1)
        g_sigma[k] = weighted_sigma.sum(axis=0)
    
    # Normalizar por peso y sqrt(2*w[k])
    for k in range(K):
        g_mu[k] /= (np.sqrt(w[k]) + 1e-12)
        g_sigma[k] /= (np.sqrt(2*w[k]) + 1e-12)
    
    # Concatenar
    fv = np.concatenate([g_mu.flatten(), g_sigma.flatten()])
    
    # Power normalization + L2
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-12)
    fv /= (np.linalg.norm(fv) + 1e-12)
    
    return fv.astype(np.float32)

# ---------- Extracción por split ----------
def extract_split_fisher(ds, pca, gmm, step=6, sizes=(12,16), limit=None):
    """
    Extrae Fisher Vectors con SPM para un split completo
    
    ⚠️  IMPORTANTE: Las etiquetas NO se usan para calcular los features.
    Solo se guardan junto a los features para la fase de evaluación posterior.
    """
    sift = cv2.SIFT_create()
    X, Y = [], []
    n = len(ds) if limit is None else min(limit, len(ds))
    t0 = time.time()
    
    for i in tqdm(range(n), ncols=100, desc="Extrayendo Fisher"):
        img_pil, y = ds[i]  # y = etiqueta
        
        # ✅ UNSUPERVISED: Fisher Vector NO usa 'y'
        # Solo usa la imagen, PCA (unsupervised) y GMM (unsupervised)
        fvs = []
        for reg in spm_regions(img_pil):
            gray = to_gray_uint8(reg)
            desc = compute_dense_rootsift(gray, sift, step=step, sizes=sizes)
            
            if desc.shape[0] == 0:
                K = gmm.n_components
                d = pca.n_components_
                fv = np.zeros(2*K*d, dtype=np.float32)
            else:
                # PCA y GMM ya fueron entrenados SIN etiquetas
                desc_pca = pca.transform(desc)  # ✅ Transformación unsupervised
                fv = fisher_vector(desc_pca, gmm)  # ✅ Codificación unsupervised
            
            fvs.append(fv)
        
        # Concatenar regiones SPM
        fv_concat = np.concatenate(fvs, dtype=np.float32)
        fv_concat /= (np.linalg.norm(fv_concat) + 1e-12)
        
        X.append(fv_concat)
        Y.append(y)  # ⚠️ Solo se GUARDA para fase de evaluación, NO se usa aquí
    
    t1 = time.time()
    X = np.stack(X)
    Y = np.array(Y)
    return X, Y, (t1 - t0), (t1 - t0)/n

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Fisher Vector + SPM para STL-10")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--pca_path", type=str, default=None)
    ap.add_argument("--gmm_path", type=str, required=True,
                    help="Ruta al modelo GMM (.joblib)")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--sizes", type=int, nargs="+", default=[12,16])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT/"data")
    ART = ROOT/"artifacts"
    FEAT = ROOT/"features"
    FEAT.mkdir(exist_ok=True)
    
    # Cargar modelos
    pca_path = Path(args.pca_path) if args.pca_path else (ART/"pca_sift24.joblib")
    gmm_path = Path(args.gmm_path)
    
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA no encontrado: {pca_path}")
    if not gmm_path.exists():
        raise FileNotFoundError(f"GMM no encontrado: {gmm_path}")
    
    print(f"📂 Cargando PCA desde {pca_path}...")
    pca = load(pca_path)
    print(f"📂 Cargando GMM desde {gmm_path}...")
    gmm = load(gmm_path)
    
    K = gmm.n_components
    d = pca.n_components_
    base_dim = 2 * K * d  # Fisher = 2*K*d
    spm_dim = base_dim * 5  # SPM: 5 regiones
    
    print(f"\n🔧 Configuración:")
    print(f"   PCA dims: {d}")
    print(f"   GMM components: {K}")
    print(f"   Fisher base dim: {base_dim}")
    print(f"   SPM total dim: {spm_dim}")
    
    if spm_dim > 4096:
        print(f"⚠️  ADVERTENCIA: Dimensión {spm_dim} > 4096")
        print("   Considera usar whitening/PCA después")
    
    # Datasets
    ds_tr = STL10(root=str(DATA), split="train", download=False)
    ds_te = STL10(root=str(DATA), split="test", download=False)
    
    # Extraer train
    print("\n🚀 Extrayendo TRAIN...")
    Xtr, ytr, t_tr, tpi_tr = extract_split_fisher(
        ds_tr, pca, gmm, 
        step=args.step, 
        sizes=tuple(args.sizes),
        limit=args.limit
    )
    np.save(FEAT/"X_train_fisher.npy", Xtr)
    np.save(FEAT/"y_train.npy", ytr)
    print(f"   {len(ytr)} imgs | {t_tr:.1f}s total | {tpi_tr:.3f}s/img")
    
    # Extraer test
    print("\n🚀 Extrayendo TEST...")
    Xte, yte, t_te, tpi_te = extract_split_fisher(
        ds_te, pca, gmm,
        step=args.step,
        sizes=tuple(args.sizes),
        limit=args.limit
    )
    np.save(FEAT/"X_test_fisher.npy", Xte)
    np.save(FEAT/"y_test.npy", yte)
    print(f"   {len(yte)} imgs | {t_te:.1f}s total | {tpi_te:.3f}s/img")
    
    print(f"\n✅ Features guardadas en {FEAT}/")
    print(f"   Dimensión final: {Xtr.shape[1]}")

if __name__ == "__main__":
    main()