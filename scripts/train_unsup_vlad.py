#!/usr/bin/env python3
import sys, argparse, random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # añade raíz del repo al path

import numpy as np
from tqdm import tqdm
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from torchvision.datasets import STL10

from src.preprocess import augment_soft, to_gray_uint8, make_rng

# --- Helpers SIFT densos ---
def dense_keypoints(h, w, step=6, sizes=(12,16)):
    import cv2
    kps = []
    for s in sizes:
        r = range(s//2, h - s//2, step)
        c = range(s//2, w - s//2, step)
        for y in r:
            for x in c:
                kps.append(cv2.KeyPoint(float(x), float(y), float(s)))
    return kps

def compute_dense_rootsift(gray_uint8, sift, step=6, sizes=(12,16)):
    # gray_uint8: (H,W) uint8
    import cv2
    kps = dense_keypoints(gray_uint8.shape[0], gray_uint8.shape[1], step=step, sizes=sizes)
    kps, desc = sift.compute(gray_uint8, kps)
    if desc is None or len(desc) == 0:
        return np.empty((0, 128), dtype=np.float32)
    desc = desc.astype(np.float32)
    eps = 1e-12
    desc /= (desc.sum(axis=1, keepdims=True) + eps)  # L1
    desc = np.sqrt(desc)                             # RootSIFT
    return desc

def main():
    parser = argparse.ArgumentParser(description="Entrena PCA y KMeans para VLAD (no supervisado) con STL-10 unlabeled")
    parser.add_argument("--data_root", type=str, default=None, help="Carpeta del dataset (default: <repo>/data)")
    parser.add_argument("--n_images", type=int, default=8000, help="# imágenes de 'unlabeled' a muestrear")
    parser.add_argument("--per_image", type=int, default=300, help="# descriptores por imagen a tomar (máx)")
    parser.add_argument("--step", type=int, default=6, help="paso de rejilla para SIFT denso")
    parser.add_argument("--sizes", type=int, nargs="+", default=[12,16], help="tamaños de patch SIFT")
    parser.add_argument("--pca_dim", type=int, default=64, help="dimensión PCA para SIFT")
    parser.add_argument("--k", type=int, default=64, help="# centroides para VLAD")
    parser.add_argument("--batch", type=int, default=10000, help="batch_size MiniBatchKMeans")
    parser.add_argument("--seed", type=int, default=0, help="semilla")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT / "data")
    ART  = ROOT / "artifacts"
    ART.mkdir(exist_ok=True)

    # Dataset (sin transform; STL10 ya devuelve PIL.Image)
    ds_u = STL10(root=str(DATA), split="unlabeled", download=False)

    # SIFT
    try:
        import cv2
        sift = cv2.SIFT_create()
    except Exception as e:
        raise RuntimeError(
            "No se pudo crear SIFT. Instala opencv-contrib-python en tu venv:\n"
            "    pip install opencv-contrib-python"
        ) from e

    # Muestreo de imágenes
    idxs = list(range(len(ds_u)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[:args.n_images]

    rng = make_rng(args.seed)
    all_desc = []

    print(f"Usando data en: {DATA}")
    print(f"Recolectando descriptores de {len(idxs)} imágenes unlabeled...")
    for i in tqdm(idxs, ncols=100):
        img_pil, _ = ds_u[i]
        img_aug = augment_soft(img_pil, rng)           # augment suave SOLO aquí
        gray    = to_gray_uint8(img_aug)               # gris uint8 96x96
        desc    = compute_dense_rootsift(gray, sift, step=args.step, sizes=tuple(args.sizes))
        if desc.shape[0] == 0:
            continue
        m = min(args.per_image, desc.shape[0])
        # muestreo por imagen para limitar memoria
        sel = np.random.RandomState(args.seed).choice(desc.shape[0], m, replace=False)
        all_desc.append(desc[sel])

    if not all_desc:
        raise RuntimeError("No se recolectaron descriptores. Revisa que SIFT funcione y que la data esté accesible.")

    all_desc = np.vstack(all_desc)  # [N,128]
    print(f"Total de descriptores recolectados: {all_desc.shape[0]:,}  (dim={all_desc.shape[1]})")

    # PCA
    print(f"Entrenando PCA a d={args.pca_dim}...")
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca.fit(all_desc)
    desc_pca = pca.transform(all_desc)
    print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.sum():.4f}")

    # KMeans (codebook VLAD)
    print(f"Entrenando MiniBatchKMeans K={args.k}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k, batch_size=args.batch, random_state=args.seed, n_init="auto"
    )
    kmeans.fit(desc_pca)

    # Guardar artefactos
    pca_path = ART / f"pca_sift{args.pca_dim}.joblib"
    kmeans_path = ART / f"kmeans_vlad{args.k}.joblib"
    from joblib import dump
    dump(pca, pca_path)
    dump(kmeans, kmeans_path)

    print("\n✅ Listo.")
    print(f"Guardado PCA:     {pca_path}")
    print(f"Guardado KMeans:  {kmeans_path}")
    print(f"Descriptor final VLAD será de dim: {args.k * args.pca_dim} (debe ser ≤ 4096)")

if __name__ == "__main__":
    main()
