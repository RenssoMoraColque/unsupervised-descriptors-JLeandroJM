#!/usr/bin/env python3
import sys, argparse, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    ap = argparse.ArgumentParser(description="Entrena y evalúa SVM lineal sobre VLAD")
    ap.add_argument("--features_root", type=str, default=None, help="Carpeta con X_train_vlad.npy etc. (default: <repo>/features)")
    ap.add_argument("--C", type=float, default=1.0, help="C del LinearSVC")
    ap.add_argument("--seed", type=int, default=0, help="semilla")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    FEAT = Path(args.features_root) if args.features_root else (ROOT / "features")

    Xtr = np.load(FEAT / "X_train_vlad.npy")
    ytr = np.load(FEAT / "y_train.npy")
    Xte = np.load(FEAT / "X_test_vlad.npy")
    yte = np.load(FEAT / "y_test.npy")

    # escalado (sin centrar para no romper L2 previo)
    sc = StandardScaler(with_mean=False)
    t0 = time.time()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    t1 = time.time()

    clf = LinearSVC(C=args.C, random_state=args.seed, dual=False, max_iter=10000, tol=1e-3)

    clf.fit(Xtr_s, ytr)
    t2 = time.time()

    ypr = clf.predict(Xte_s)
    t3 = time.time()

    acc = accuracy_score(yte, ypr)
    f1m = f1_score(yte, ypr, average="macro")
    print(f"\nAccuracy: {acc:.4f}  Macro-F1: {f1m:.4f}\n")
    print(classification_report(yte, ypr, digits=4))

    print("Tiempos:")
    print(f" - Escalado: {t1 - t0:.2f}s")
    print(f" - Entrenamiento SVM: {t2 - t1:.2f}s")
    print(f" - Predicción: {t3 - t2:.2f}s")

if __name__ == "__main__":
    main()
