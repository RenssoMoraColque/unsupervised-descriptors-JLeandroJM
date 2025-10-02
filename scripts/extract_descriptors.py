#!/usr/bin/env python3
"""
Extrae descriptores SIFT densos, aplica PCA y guarda para entrenar GMM
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from tqdm import tqdm
from torchvision.datasets import STL10
from joblib import load
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
    ap = argparse.ArgumentParser(description="Extraer descriptores SIFT+PCA para GMM")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--pca_path", type=str, default=None,
                    help="Ruta al modelo PCA (.joblib)")
    ap.add_argument("--output", type=str, required=True,
                    help="Ruta de salida para descriptores (.npy)")
    ap.add_argument("--step", type=int, default=6,
                    help="Paso para keypoints densos")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12,16],
                    help="Tamaños de los keypoints")
    ap.add_argument("--max_images", type=int, default=1000,
                    help="Número máximo de imágenes a procesar")
    ap.add_argument("--max_desc_per_image", type=int, default=500,
                    help="Máximo de descriptores por imagen")
    args = ap.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    DATA = Path(args.data_root) if args.data_root else (ROOT / "data")
    ART = ROOT / "artifacts"
    ART.mkdir(exist_ok=True)
    
    # Cargar PCA si existe
    pca_path = Path(args.pca_path) if args.pca_path else (ART / "pca_sift24.joblib")
    if pca_path.exists():
        print(f"📂 Cargando PCA desde {pca_path}...")
        pca = load(pca_path)
        print(f"   Componentes: {pca.n_components_}")
    else:
        print(f"⚠️  No se encontró PCA en {pca_path}")
        print("   Se extraerán descriptores SIFT sin reducción de dimensionalidad")
        pca = None
    
    # Cargar dataset
    print(f"\n📂 Cargando STL-10 desde {DATA}...")
    ds = STL10(root=str(DATA), split="train", download=False)
    
    # Extraer descriptores
    sift = cv2.SIFT_create()
    all_desc = []
    n_images = min(args.max_images, len(ds))
    
    print(f"\n🔧 Extrayendo descriptores de {n_images} imágenes...")
    for i in tqdm(range(n_images), ncols=100):
        img_pil, _ = ds[i]
        gray = to_gray_uint8(img_pil)
        
        desc = compute_dense_rootsift(gray, sift, step=args.step, sizes=tuple(args.sizes))
        
        if desc.shape[0] > 0:
            # Submuestrear si hay demasiados descriptores
            if desc.shape[0] > args.max_desc_per_image:
                idx = np.random.choice(desc.shape[0], args.max_desc_per_image, replace=False)
                desc = desc[idx]
            all_desc.append(desc)
    
    # Concatenar todos los descriptores
    all_desc = np.vstack(all_desc).astype(np.float32)
    print(f"\n✓ Total descriptores extraídos: {all_desc.shape}")
    
    # Aplicar PCA si está disponible
    if pca is not None:
        print(f"🔧 Aplicando PCA ({pca.n_components_} componentes)...")
        all_desc = pca.transform(all_desc).astype(np.float32)
        print(f"✓ Shape después de PCA: {all_desc.shape}")
    
    # Guardar
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(output_path, all_desc)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Descriptores guardados en: {output_path}")
    print(f"   - Shape: {all_desc.shape}")
    print(f"   - Tamaño: {size_mb:.1f} MB")
    print(f"   - Dtype: {all_desc.dtype}")
    
    print(f"\n💡 Siguiente paso - entrenar GMM:")
    print(f"   python scripts/train_gmm_fisher.py --desc_path {output_path} --n_components 32")

if __name__ == "__main__":
    main()
