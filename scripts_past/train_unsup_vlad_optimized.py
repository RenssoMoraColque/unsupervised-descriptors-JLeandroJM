#!/usr/bin/env python3
"""
Script optimizado para entrenar VLAD con mejores parámetros para mayor accuracy
Basado en train_unsup_vlad.py pero con configuraciones mejoradas
"""
import sys, argparse, random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from torchvision.datasets import STL10

from src.preprocess import augment_soft, to_gray_uint8, make_rng

def dense_keypoints_optimized(h, w, step=4, sizes=(8, 12, 16, 20)):
    """Keypoints más densos con múltiples escalas para mejor cobertura"""
    import cv2
    kps = []
    for s in sizes:
        r = range(s//2, h - s//2, step)
        c = range(s//2, w - s//2, step)
        for y in r:
            for x in c:
                kps.append(cv2.KeyPoint(float(x), float(y), float(s)))
    return kps

def compute_dense_rootsift_optimized(gray_uint8, sift, step=4, sizes=(8, 12, 16, 20)):
    """RootSIFT optimizado con más escalas y mejor preprocessing"""
    import cv2
    
    # CLAHE para mejor contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_uint8)
    
    kps = dense_keypoints_optimized(gray_enhanced.shape[0], gray_enhanced.shape[1], 
                                   step=step, sizes=sizes)
    kps, desc = sift.compute(gray_enhanced, kps)
    
    if desc is None or len(desc) == 0:
        return np.empty((0, 128), dtype=np.float32)
    
    desc = desc.astype(np.float32)
    eps = 1e-12
    desc /= (desc.sum(axis=1, keepdims=True) + eps)  # L1
    desc = np.sqrt(desc)                             # RootSIFT
    
    # Filtrar descriptores muy similares para diversidad
    if desc.shape[0] > 100:
        # Clustering simple para mantener diversidad
        from sklearn.cluster import KMeans
        n_clusters = min(desc.shape[0] // 3, 300)
        if n_clusters > 10:
            kmeans_local = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans_local.fit_predict(desc)
            # Tomar centroide más cercano de cada cluster
            unique_labels = np.unique(labels)
            diverse_desc = []
            for label in unique_labels:
                cluster_desc = desc[labels == label]
                center = cluster_desc.mean(axis=0)
                distances = ((cluster_desc - center)**2).sum(axis=1)
                best_idx = np.argmin(distances)
                diverse_desc.append(cluster_desc[best_idx])
            desc = np.array(diverse_desc, dtype=np.float32)
    
    return desc

def main():
    parser = argparse.ArgumentParser(description="VLAD optimizado para mejor accuracy")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=20000, help="Más imágenes para mejor modelo")
    parser.add_argument("--per_image", type=int, default=500, help="Más descriptores por imagen")
    parser.add_argument("--step", type=int, default=4, help="Step más pequeño = más denso")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8,12,16,20], help="Múltiples escalas")
    parser.add_argument("--pca_dim", type=int, default=80, help="PCA dimension más alta")
    parser.add_argument("--k", type=int, default=96, help="Más centroides para mayor capacidad")
    parser.add_argument("--batch", type=int, default=15000, help="Batch size mayor")
    parser.add_argument("--seed", type=int, default=42, help="semilla")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT / "data")
    ART  = ROOT / "artifacts"
    ART.mkdir(exist_ok=True)

    # Dataset
    ds_u = STL10(root=str(DATA), split="unlabeled", download=False)

    # SIFT optimizado
    try:
        import cv2
        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=15)
    except Exception as e:
        raise RuntimeError("Instala opencv-contrib-python") from e

    # Muestreo de más imágenes
    idxs = list(range(len(ds_u)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[:args.n_images]

    rng = make_rng(args.seed)
    all_desc = []

    print(f"Recolectando descriptores de {len(idxs)} imágenes unlabeled...")
    print(f"Configuración: step={args.step}, sizes={args.sizes}, per_image={args.per_image}")
    
    for i in tqdm(idxs, ncols=100):
        img_pil, _ = ds_u[i]
        img_aug = augment_soft(img_pil, rng)
        gray = to_gray_uint8(img_aug)
        desc = compute_dense_rootsift_optimized(gray, sift, step=args.step, sizes=tuple(args.sizes))
        
        if desc.shape[0] == 0:
            continue
            
        m = min(args.per_image, desc.shape[0])
        sel = np.random.RandomState(args.seed + i).choice(desc.shape[0], m, replace=False)
        all_desc.append(desc[sel])

    if not all_desc:
        raise RuntimeError("No se recolectaron descriptores")

    all_desc = np.vstack(all_desc)
    print(f"Total descriptores: {all_desc.shape[0]:,} (dim={all_desc.shape[1]})")

    # PCA con mayor retención de varianza
    print(f"PCA a d={args.pca_dim}...")
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca.fit(all_desc)
    desc_pca = pca.transform(all_desc)
    print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.4f}")

    # K-means más robusto
    print(f"MiniBatchKMeans K={args.k}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k, 
        batch_size=args.batch, 
        random_state=args.seed, 
        n_init=10,  # Más inicializaciones
        max_iter=500,  # Más iteraciones
        tol=1e-6,  # Convergencia más estricta
        reassignment_ratio=0.01
    )
    kmeans.fit(desc_pca)

    # Guardar con nombre específico
    pca_path = ART / f"pca_sift{args.pca_dim}_opt.joblib"
    kmeans_path = ART / f"kmeans_vlad{args.k}_opt.joblib"
    dump(pca, pca_path)
    dump(kmeans, kmeans_path)

    final_dim = args.k * args.pca_dim * 5  # SPM
    print(f"\n✅ Modelo optimizado guardado!")
    print(f"PCA: {pca_path}")
    print(f"KMeans: {kmeans_path}")
    print(f"Dimensión final VLAD+SPM: {final_dim} (límite: 4096)")
    
    if final_dim > 4096:
        print("⚠️ ADVERTENCIA: Dimensión excede 4096, usa --keep_dim en whitening")

if __name__ == "__main__":
    main()