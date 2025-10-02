#!/usr/bin/env python3
import sys, argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def main():
    ap = argparse.ArgumentParser(description="SVM RBF sobre features (VLAD/SPM/híbrido)")
    ap.add_argument("--features_root", type=str, default=None)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=None,
                    help="Si None, usa 1/D")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    FEAT = Path(args.features_root) if args.features_root else (ROOT/"features")

    Xtr = np.load(FEAT/"X_train_vlad_whiten.npy", allow_pickle=False) if (FEAT/"X_train_vlad_whiten.npy").exists() else np.load(FEAT/"X_train_vlad.npy")
    Xte = np.load(FEAT/"X_test_vlad_whiten.npy",  allow_pickle=False) if (FEAT/"X_test_vlad_whiten.npy").exists()  else np.load(FEAT/"X_test_vlad.npy")
    ytr = np.load(FEAT/"y_train.npy"); yte = np.load(FEAT/"y_test.npy")

    sc = StandardScaler(with_mean=False)
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

    D = Xtr.shape[1]
    gamma = args.gamma if args.gamma is not None else 1.0 / D
    clf = SVC(kernel="rbf", C=args.C, gamma=gamma, random_state=args.seed)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)

    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    print(f"RBF  C={args.C}  gamma={gamma:.6g}  ->  Acc={acc:.4f}  MacroF1={f1m:.4f}")

if __name__ == "__main__":
    main()
