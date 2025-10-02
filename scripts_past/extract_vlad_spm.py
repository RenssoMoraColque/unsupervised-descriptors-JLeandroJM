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

# ---------- VLAD (hard) ----------
def vlad_encode(desc_d, centers):
    if desc_d.size == 0:
        return np.zeros(centers.shape[0]*centers.shape[1], dtype=np.float32)
    d2 = ((desc_d[:, None, :] - centers[None, :, :])**2).sum(axis=2)  # (N,K)
    assign = np.argmin(d2, axis=1)
    K, D = centers.shape
    V = np.zeros((K, D), dtype=np.float32)
    for i, a in enumerate(assign):
        V[a] += (desc_d[i] - centers[a])
    # intra -> power -> L2
    for k in range(K):
        n = np.linalg.norm(V[k]) + 1e-12
        if n > 0: V[k] /= n
    V = np.sign(V)*np.sqrt(np.abs(V)+1e-12)
    V = V.reshape(-1)
    V /= (np.linalg.norm(V)+1e-12)
    return V.astype(np.float32)

# ---------- VLAD-k (soft) opcional ----------
def vlad_encode_soft(desc_d, centers, m=5, temp=1.0):
    if desc_d.size == 0:
        return np.zeros(centers.shape[0]*centers.shape[1], dtype=np.float32)
    d2 = ((desc_d[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    idx_sorted = np.argsort(d2, axis=1)[:, :m]
    d2_top = np.take_along_axis(d2, idx_sorted, axis=1)
    w = np.exp(-d2_top / (2.0*(temp**2) + 1e-12))
    w /= (w.sum(axis=1, keepdims=True) + 1e-12)
    K, D = centers.shape
    V = np.zeros((K, D), dtype=np.float32)
    for i in range(desc_d.shape[0]):
        for j in range(m):
            c = idx_sorted[i, j]
            V[c] += w[i, j] * (desc_d[i] - centers[c])
    for k in range(K):
        n = np.linalg.norm(V[k]) + 1e-12
        if n > 0: V[k] /= n
    V = np.sign(V)*np.sqrt(np.abs(V)+1e-12)
    V = V.reshape(-1)
    V /= (np.linalg.norm(V)+1e-12)
    return V.astype(np.float32)

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

def extract_split_spm(ds, pca, centers, step=6, sizes=(12,16),
                      limit=None, use_clahe=False, sigma=0.0,
                      vlad_k=1, temp=1.0, add_colorlbp=False):
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
                V = np.zeros(centers.shape[0]*centers.shape[1], dtype=np.float32)
            else:
                desc_p = pca.transform(desc)
                V = vlad_encode(desc_p, centers) if vlad_k<=1 else vlad_encode_soft(desc_p, centers, m=int(vlad_k), temp=float(temp))
            Vs.append(V)  # cada V ya está normalizado
        V_concat = np.concatenate(Vs, dtype=np.float32)  # (5 * K*d,)
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
    ap = argparse.ArgumentParser(description="VLAD + SPM (1x1 + 2x2) para STL-10")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--pca_path", type=str, default=None)
    ap.add_argument("--kmeans_path", type=str, default=None)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--sizes", type=int, nargs="+", default=[12,16])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--pre_blur_sigma", type=float, default=0.0)
    ap.add_argument("--vlad_k", type=int, default=1)
    ap.add_argument("--soft_temp", type=float, default=1.0)
    ap.add_argument("--add_colorlbp", action="store_true",
                    help="agrega HSV+LBP global (315 dims) — OJO con 4096")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT/"data")
    ART  = ROOT/"artifacts"
    FEAT = ROOT/"features"; FEAT.mkdir(exist_ok=True)

    pca_path = Path(args.pca_path) if args.pca_path else (ART/"pca_sift24.joblib")
    km_path  = Path(args.kmeans_path) if args.kmeans_path else (ART/"kmeans_vlad32.joblib")

    pca = load(pca_path)
    centers = load(km_path).cluster_centers_
    base_dim = centers.shape[0]*centers.shape[1]  # K*d
    spm_dim  = base_dim * 5                       # 1x1 + 2x2
    total_dim = spm_dim + (315 if args.add_colorlbp else 0)
    if total_dim > 4096:
        raise RuntimeError(f"Dim final {total_dim} > 4096. Ajusta K/d o desactiva --add_colorlbp.")

    ds_tr = STL10(root=str(DATA), split="train", download=False)
    ds_te = STL10(root=str(DATA), split="test",  download=False)

    print(f"Dim celda (K*d) = {base_dim} ; Dim SPM = {spm_dim} ; Dim total = {total_dim}")

    Xtr, ytr, t_train, tpi_train = extract_split_spm(
        ds_tr, pca, centers, step=args.step, sizes=tuple(args.sizes),
        limit=args.limit, use_clahe=args.clahe, sigma=args.pre_blur_sigma,
        vlad_k=args.vlad_k, temp=args.soft_temp, add_colorlbp=args.add_colorlbp
    )
    np.save(FEAT/"X_train_vlad.npy", Xtr); np.save(FEAT/"y_train.npy", ytr)
    print(f"[TRAIN] {len(ytr)} imgs | total = {t_train:.2f}s | img = {tpi_train:.4f}s")

    Xte, yte, t_test, tpi_test = extract_split_spm(
        ds_te, pca, centers, step=args.step, sizes=tuple(args.sizes),
        limit=args.limit, use_clahe=args.clahe, sigma=args.pre_blur_sigma,
        vlad_k=args.vlad_k, temp=args.soft_temp, add_colorlbp=args.add_colorlbp
    )
    np.save(FEAT/"X_test_vlad.npy", Xte); np.save(FEAT/"y_test.npy", yte)
    print(f"[TEST ] {len(yte)} imgs | total = {t_test:.2f}s | img = {tpi_test:.4f}s")
    print("\n✅ Listo. Guardado en 'features/'.")

if __name__ == "__main__":
    main()
