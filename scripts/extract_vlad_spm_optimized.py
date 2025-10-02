#!/usr/bin/env python3
import sys, argparse, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from joblib import load
from torchvision.datasets import STL10
from skimage.feature import local_binary_pattern  # <- import corregido
from src.preprocess import to_gray_uint8


# ---------- SPM (1x1 + 2x2 + 4x4 opcional) ----------
def spm_regions(img_pil):
    W, H = img_pil.size  # (96,96)
    regs = [img_pil]
    regs += [
        img_pil.crop((0,     0,     W//2, H//2)),  # TL
        img_pil.crop((W//2,  0,     W,    H//2)),  # TR
        img_pil.crop((0,     H//2,  W//2, H   )),  # BL
        img_pil.crop((W//2,  H//2,  W,    H   )),  # BR
    ]
    return regs  # 5

def spm_regions_enhanced(img_pil):
    W, H = img_pil.size
    regs = [img_pil]  # 1x1
    # 2x2
    regs += [
        img_pil.crop((0,     0,     W//2, H//2)),
        img_pil.crop((W//2,  0,     W,    H//2)),
        img_pil.crop((0,     H//2,  W//2, H   )),
        img_pil.crop((W//2,  H//2,  W,    H   )),
    ]
    # 4x4 (16 celdas)
    for i in range(4):
        for j in range(4):
            x1 = j * (W//4); x2 = (j+1) * (W//4)
            y1 = i * (H//4); y2 = (i+1) * (H//4)
            regs.append(img_pil.crop((x1, y1, x2, y2)))
    return regs  # 21 = 1 + 4 + 16


# ---------- SIFT denso + RootSIFT (versión “enhanced”) ----------
def dense_keypoints_enhanced(h, w, step=4, sizes=(8,12,16,24)):
    import cv2
    kps = []
    for s in sizes:
        for y in range(s//2, h - s//2, step):
            for x in range(s//2, w - s//2, step):
                kps.append(cv2.KeyPoint(float(x), float(y), float(s)))
    return kps

def compute_dense_rootsift_enhanced(gray_uint8, sift, step=4, sizes=(8,12,16,24)):
    import cv2
    kps = dense_keypoints_enhanced(gray_uint8.shape[0], gray_uint8.shape[1], step=step, sizes=sizes)
    kps, desc = sift.compute(gray_uint8, kps)
    if desc is None or len(desc) == 0:
        return np.empty((0,128), dtype=np.float32)
    desc = desc.astype(np.float32)
    eps = 1e-12
    # L2 -> L1 -> sqrt -> L2  (tu variante)
    desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + eps)
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc + eps)
    desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + eps)
    return desc


# ---------- Preproc gris ----------
def gray_for_sift_enhanced(img_pil, use_clahe=True, sigma=0.5, enhance_contrast=True):
    import cv2
    g = to_gray_uint8(img_pil)
    if sigma > 0:
        g = cv2.GaussianBlur(g, ksize=(0,0), sigmaX=float(sigma), sigmaY=float(sigma))
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        g = clahe.apply(g)
    if enhance_contrast:
        g = cv2.convertScaleAbs(g, alpha=1.2, beta=10)
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


# ---------- VLAD (hard y soft) ----------
def vlad_encode_enhanced(desc_d, centers, soft_assignment=True, alpha=5.0, beta=2.0):
    if desc_d.size == 0:
        return np.zeros(centers.shape[0]*centers.shape[1], dtype=np.float32)
    # distancias
    d2 = ((desc_d[:, None, :] - centers[None, :, :])**2).sum(axis=2)  # (N,K)
    K, D = centers.shape
    V = np.zeros((K, D), dtype=np.float32)

    if soft_assignment:
        # softmax( -alpha * d2 )
        w = np.exp(-alpha * d2)
        w /= (w.sum(axis=1, keepdims=True) + 1e-12)
        # umbral para eficiencia
        for i in range(desc_d.shape[0]):
            wi = w[i]
            nz = np.where(wi > 0.01)[0]
            for k in nz:
                V[k] += wi[k] * (desc_d[i] - centers[k])
    else:
        assign = np.argmin(d2, axis=1)
        for i, a in enumerate(assign):
            V[a] += (desc_d[i] - centers[a])

    # intra-norm -> power(norm beta) -> L2
    for k in range(K):
        n = np.linalg.norm(V[k]) + 1e-12
        if n > 0:
            V[k] /= n
    V = np.sign(V) * np.power(np.abs(V) + 1e-12, 1.0 / beta)
    V = V.reshape(-1)
    V /= (np.linalg.norm(V) + 1e-12)
    return V.astype(np.float32)


# ---------- HSV + LBP multiescala (dim = 426) ----------
def color_lbp_feats_enhanced(img_pil):
    import cv2
    img = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv],[0],None,[180],[0,180]).flatten()  # 180
    s = cv2.calcHist([hsv],[1],None,[64],[0,256]).flatten()   # 64
    v = cv2.calcHist([hsv],[2],None,[64],[0,256]).flatten()   # 64
    hsv_hist = np.concatenate([h,s,v]).astype("float32")
    hsv_hist /= (hsv_hist.sum()+1e-12)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp1 = local_binary_pattern(gray, P=8,  R=1, method="uniform")
    lbp2 = local_binary_pattern(gray, P=16, R=2, method="uniform")
    # usamos 59 bins (estilo u2 clásico) para ambos, aunque skimage 'uniform' no lo exige
    lbp1_hist,_ = np.histogram(lbp1, bins=np.arange(0,60), range=(0,59))  # 59
    lbp2_hist,_ = np.histogram(lbp2, bins=np.arange(0,60), range=(0,59))  # 59
    lbp1_hist = lbp1_hist.astype("float32"); lbp1_hist /= (lbp1_hist.sum()+1e-12)
    lbp2_hist = lbp2_hist.astype("float32"); lbp2_hist /= (lbp2_hist.sum()+1e-12)

    return np.concatenate([hsv_hist, lbp1_hist, lbp2_hist], dtype=np.float32)  # 180+64+64+59+59=426


# ---------- Extracción (SPM simple o “enhanced”) ----------
def extract_split_spm_optimized(
    ds, pca, centers, step=4, sizes=(8,12,16,24),
    limit=None, use_clahe=True, sigma=0.5,
    soft_assignment=True, alpha=5.0, beta=2.0,
    add_colorlbp=False, enhanced_spm=False
):
    import cv2
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=8)
    X, Y = [], []
    n = len(ds) if limit is None else min(limit, len(ds))
    t0 = time.time()

    for i in tqdm(range(n), ncols=100, desc="Extrayendo features"):
        img_pil, y = ds[i]
        regions = spm_regions_enhanced(img_pil) if enhanced_spm else spm_regions(img_pil)

        Vs = []
        for reg in regions:
            gray = gray_for_sift_enhanced(reg, use_clahe, sigma, enhance_contrast=True)
            desc = compute_dense_rootsift_enhanced(gray, sift, step=step, sizes=sizes)
            if desc.shape[0] == 0:
                V = np.zeros(centers.shape[0]*centers.shape[1], dtype=np.float32)
            else:
                desc_p = pca.transform(desc)
                V = vlad_encode_enhanced(desc_p, centers, soft_assignment, alpha, beta)
            Vs.append(V)

        V_concat = np.concatenate(Vs, dtype=np.float32)
        V_concat /= (np.linalg.norm(V_concat) + 1e-12)

        if add_colorlbp:
            color_feats = color_lbp_feats_enhanced(img_pil)  # 426
            V_concat = np.concatenate([V_concat, color_feats], dtype=np.float32)
            V_concat /= (np.linalg.norm(V_concat) + 1e-12)

        X.append(V_concat); Y.append(y)

    t1 = time.time()
    X = np.stack(X); Y = np.array(Y)
    return X, Y, (t1 - t0), (t1 - t0)/n


def main():
    ap = argparse.ArgumentParser(description="VLAD + SPM optimizado para STL-10")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--pca_path", type=str, default=None)
    ap.add_argument("--kmeans_path", type=str, default=None)
    ap.add_argument("--step", type=int, default=4, help="paso de rejilla SIFT (default: 4)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8,12,16,24], help="escalas SIFT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=0.5, help="pre-blur sigma")
    ap.add_argument("--clahe", action="store_true", help="aplicar CLAHE")
    ap.add_argument("--soft_assignment", action="store_true", help="usar VLAD con asignación suave")
    ap.add_argument("--alpha", type=float, default=5.0, help="temperatura (soft VLAD)")
    ap.add_argument("--beta", type=float, default=2.0, help="exponente de power-normalization")
    ap.add_argument("--add_colorlbp", action="store_true", help="agregar HSV+LBP (426 dims)")
    ap.add_argument("--enhanced_spm", action="store_true", help="SPM 1x1+2x2+4x4 (21 regiones)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT/"data")
    ART  = ROOT/"artifacts"
    FEAT = ROOT/"features"; FEAT.mkdir(exist_ok=True)

    # Por defecto: artefactos pensados para SPM (K=32, d=24) => celda 768 dims
    pca_path = Path(args.pca_path) if args.pca_path else (ART/"pca_sift24.joblib")
    km_path  = Path(args.kmeans_path) if args.kmeans_path else (ART/"kmeans_vlad32.joblib")

    if not pca_path.exists():
        print(f"❌ No existe {pca_path}")
        return
    if not km_path.exists():
        print(f"❌ No existe {km_path}")
        return

    pca = load(pca_path)
    centers = load(km_path).cluster_centers_
    base_dim = centers.shape[0] * centers.shape[1]

    spm_regions_count = 21 if args.enhanced_spm else 5
    spm_dim = base_dim * spm_regions_count
    color_dim = 426 if args.add_colorlbp else 0
    total_dim = spm_dim + color_dim

    # Fallback si excede 4096
    if total_dim > 4096:
        print(f"⚠️  Dim final {total_dim} > 4096. Ajustando...")
        if args.enhanced_spm:
            args.enhanced_spm = False
            spm_regions_count = 5
            spm_dim = base_dim * spm_regions_count
            total_dim = spm_dim + color_dim
            print(f"   -> Usando SPM simple (5 regiones), dim={total_dim}")
        if total_dim > 4096 and args.add_colorlbp:
            args.add_colorlbp = False
            color_dim = 0
            total_dim = spm_dim
            print(f"   -> Quitando HSV+LBP, dim={total_dim}")
        if total_dim > 4096:
            raise RuntimeError(f"Dim final aún {total_dim} > 4096. Reduce K/d en tus artefactos.")

    # Datasets
    ds_tr = STL10(root=str(DATA), split="train", download=False)
    ds_te = STL10(root=str(DATA), split="test",  download=False)

    print(f"\n🚀 Configuración optimizada")
    print(f"   PCA: {pca.n_components_} dims")
    print(f"   KMeans: {centers.shape[0]} clusters | d={centers.shape[1]}")
    print(f"   VLAD por celda: {base_dim} dims")
    print(f"   SPM regiones: {spm_regions_count} -> SPM dim: {spm_dim}")
    print(f"   Color feats: {color_dim} -> Dim total: {total_dim}")
    print(f"   CLAHE={args.clahe}  softVLAD={args.soft_assignment}  alpha={args.alpha}  beta={args.beta}")

    # Extraer
    Xtr, ytr, t_train, tpi_train = extract_split_spm_optimized(
        ds_tr, pca, centers, step=args.step, sizes=tuple(args.sizes),
        limit=args.limit, use_clahe=args.clahe, sigma=args.sigma,
        soft_assignment=args.soft_assignment, alpha=args.alpha, beta=args.beta,
        add_colorlbp=args.add_colorlbp, enhanced_spm=args.enhanced_spm
    )
    np.save(FEAT/"X_train_vlad.npy", Xtr)
    np.save(FEAT/"y_train.npy", ytr)
    print(f"\n[TRAIN] {len(ytr)} imgs | {t_train:.1f}s total | {tpi_train:.3f}s/img")

    Xte, yte, t_test, tpi_test = extract_split_spm_optimized(
        ds_te, pca, centers, step=args.step, sizes=tuple(args.sizes),
        limit=args.limit, use_clahe=args.clahe, sigma=args.sigma,
        soft_assignment=args.soft_assignment, alpha=args.alpha, beta=args.beta,
        add_colorlbp=args.add_colorlbp, enhanced_spm=args.enhanced_spm
    )
    np.save(FEAT/"X_test_vlad.npy", Xte)
    np.save(FEAT/"y_test.npy", yte)
    print(f"[TEST ] {len(yte)} imgs | {t_test:.1f}s total | {tpi_test:.3f}s/img")
    print("\n✅ Features optimizadas guardadas en features/")

if __name__ == "__main__":
    main()
