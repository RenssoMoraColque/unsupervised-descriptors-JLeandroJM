#!/usr/bin/env python3
"""
Extrae descriptores SIFT densos desde el split UNLABELED (100k imágenes)
para entrenamiento no supervisado de PCA y GMM.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from tqdm import tqdm
from torchvision.datasets import STL10
import argparse

from src.preprocess import to_gray_uint8

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

def main():
    ap = argparse.ArgumentParser(description="Extraer SIFT de unlabeled para entrenar PCA/GMM")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--output", type=str, default="artifacts/desc_sift_unlabeled.npy",
                    help="Archivo de salida para descriptores")
    ap.add_argument("--max_images", type=int, default=20000,
                    help="Máximo de imágenes unlabeled a procesar (default: 20k)")
    ap.add_argument("--max_desc_per_image", type=int, default=500,
                    help="Máximo de descriptores por imagen")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--sizes", type=int, nargs="+", default=[12,16])
    args = ap.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT / "data")
    
    print(f"📂 Cargando STL-10 UNLABELED desde {DATA}...")
    # ✅ Usando split UNLABELED (100k imágenes sin etiquetas)
    ds = STL10(root=str(DATA), split="unlabeled", download=False)
    print(f"   Total disponible: {len(ds)} imágenes")
    print(f"   Usando: {min(args.max_images, len(ds))} imágenes")
    
    # Extraer descriptores
    sift = cv2.SIFT_create()
    all_desc = []
    n_images = min(args.max_images, len(ds))
    
    print(f"\n🔧 Extrayendo descriptores RootSIFT...")
    print(f"   ⚠️  NO se usan etiquetas (split unlabeled)\n")
    
    for i in tqdm(range(n_images), ncols=100):
        img_pil, _ = ds[i]  # _ = -1 (no hay etiqueta en unlabeled)
        gray = to_gray_uint8(img_pil)
        
        desc = compute_dense_rootsift(gray, sift, step=args.step, sizes=tuple(args.sizes))
        
        if desc.shape[0] > 0:
            # Submuestrear si hay demasiados
            if desc.shape[0] > args.max_desc_per_image:
                idx = np.random.choice(desc.shape[0], args.max_desc_per_image, replace=False)
                desc = desc[idx]
            all_desc.append(desc)
    
    # Concatenar
    all_desc = np.vstack(all_desc).astype(np.float32)
    print(f"\n✓ Total descriptores: {all_desc.shape}")
    print(f"   Memoria: {all_desc.nbytes / (1024**2):.1f} MB")
    
    # Guardar
    output_path = ROOT / args.output
    output_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(output_path, all_desc)
    
    print(f"\n✅ Descriptores guardados en: {output_path}")
    print(f"\n💡 Siguiente paso - entrenar PCA:")
    print(f"   python scripts/train_pca.py --desc_path {output_path}")

if __name__ == "__main__":
    main()
