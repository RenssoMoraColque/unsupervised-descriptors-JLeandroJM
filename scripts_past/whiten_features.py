#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser(description="PCA-whitening de features (VLAD o híbrido)")
    ap.add_argument("--features_root", type=str, default=None, help="carpeta con X_*.npy (default: <repo>/features)")
    ap.add_argument("--keep_dim", type=int, default=None, help="dim final (default: misma dim; <=4096)")
    ap.add_argument("--save_suffix", type=str, default="whiten", help="sufijo de salida")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    FEAT = Path(args.features_root) if args.features_root else (ROOT/"features")

    Xtr = np.load(FEAT/"X_train_vlad.npy")
    Xte = np.load(FEAT/"X_test_vlad.npy")
    keep = min(args.keep_dim or Xtr.shape[1], 4096)
    print(f"PCA-whiten: {Xtr.shape[1]} -> {keep}")

    pca = PCA(n_components=keep, svd_solver="randomized", whiten=True, random_state=0)
    pca.fit(np.vstack([Xtr, Xte]))  # permitido: sin usar etiquetas
    Xtr_w = pca.transform(Xtr); Xte_w = pca.transform(Xte)
    Xtr_w /= (np.linalg.norm(Xtr_w, axis=1, keepdims=True) + 1e-12)
    Xte_w /= (np.linalg.norm(Xte_w, axis=1, keepdims=True) + 1e-12)
    np.save(FEAT/f"X_train_vlad_{args.save_suffix}.npy", Xtr_w)
    np.save(FEAT/f"X_test_vlad_{args.save_suffix}.npy",  Xte_w)
    dump(pca, FEAT/f"pca_whiten_{keep}.joblib")
    print("✅ Features whitened guardados.")

if __name__ == "__main__":
    main()
