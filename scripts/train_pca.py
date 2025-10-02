#!/usr/bin/env python3
"""Entrena PCA sobre descriptores SIFT"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import PCA
from joblib import dump
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc_path", type=str, required=True)
    ap.add_argument("--n_components", type=int, default=24)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    ART = ROOT / "artifacts"
    ART.mkdir(exist_ok=True)
    
    print(f"📂 Cargando descriptores desde {args.desc_path}...")
    desc = np.load(args.desc_path)
    print(f"   Shape: {desc.shape}")
    
    print(f"\n🔧 Entrenando PCA ({args.n_components} componentes)...")
    pca = PCA(n_components=args.n_components, random_state=42)
    pca.fit(desc)
    
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"   Varianza explicada: {var_exp:.4f}")
    
    output_path = Path(args.output) if args.output else (ART / f"pca_sift{args.n_components}.joblib")
    dump(pca, output_path)
    print(f"\n✅ PCA guardado en: {output_path}")

if __name__ == "__main__":
    main()
