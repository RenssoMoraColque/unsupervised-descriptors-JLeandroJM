#!/usr/bin/env python3
"""Aplica PCA a descriptores SIFT"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import load
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc_path", type=str, required=True)
    ap.add_argument("--pca_path", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    
    print(f"📂 Cargando descriptores...")
    desc = np.load(args.desc_path)
    print(f"📂 Cargando PCA...")
    pca = load(args.pca_path)
    
    print(f"\n🔧 Aplicando PCA: {desc.shape[1]} → {pca.n_components_} dims...")
    desc_pca = pca.transform(desc).astype(np.float32)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(output_path, desc_pca)
    
    print(f"\n✅ Guardado en: {output_path}")
    print(f"   Shape: {desc_pca.shape}")

if __name__ == "__main__":
    main()
