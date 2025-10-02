#!/usr/bin/env python3
"""
Grid search para encontrar mejores parámetros SVM usando SOLO validación cruzada en train.
NO usa etiquetas de test para selección de hiperparámetros (cumple restricciones del concurso).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from itertools import product
import argparse
import time

def load_features(feat_root):
    Xtr = np.load(feat_root / "X_train_fisher.npy")
    ytr = np.load(feat_root / "y_train.npy")
    Xte = np.load(feat_root / "X_test_fisher.npy")
    yte = np.load(feat_root / "y_test.npy")
    return Xtr, ytr, Xte, yte

def main():
    ap = argparse.ArgumentParser(description="Grid search SVM con validación cruzada (sin usar test)")
    ap.add_argument("--features_root", type=str, default="features")
    ap.add_argument("--cv_folds", type=int, default=5,
                    help="Número de folds para validación cruzada")
    ap.add_argument("--final_eval", action="store_true",
                    help="Evaluar mejor modelo en test (solo al final)")
    args = ap.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    FEAT = ROOT / args.features_root
    
    print("📂 Cargando features...")
    Xtr, ytr, Xte, yte = load_features(FEAT)
    print(f"   Train: {Xtr.shape}, Test: {Xte.shape}")
    
    # Escalar SOLO con train
    print("\n🔧 Escalando features...")
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)  # No se usará hasta el final
    
    # Parámetros a probar
    C_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    gamma_values = ['scale', 0.0001, 0.0005, 0.001, 0.005]
    
    print(f"\n🔍 Grid Search con {args.cv_folds}-fold CV en TRAIN")
    print(f"   Combinaciones: {len(C_values)} × {len(gamma_values)} = {len(C_values)*len(gamma_values)}")
    print(f"   ⚠️  NO se usan etiquetas de test para selección\n")
    
    # Configurar CV estratificado
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    results = []
    best_cv_acc = 0
    best_params = None
    
    t0 = time.time()
    
    for C, gamma in product(C_values, gamma_values):
        # Validación cruzada SOLO en train
        clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(clf, Xtr_sc, ytr, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_acc_mean = cv_scores.mean()
        cv_acc_std = cv_scores.std()
        
        # Cross-validation F1 macro
        cv_f1 = cross_val_score(clf, Xtr_sc, ytr, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_f1_mean = cv_f1.mean()
        cv_f1_std = cv_f1.std()
        
        results.append({
            'C': C,
            'gamma': gamma,
            'cv_acc_mean': cv_acc_mean,
            'cv_acc_std': cv_acc_std,
            'cv_f1_mean': cv_f1_mean,
            'cv_f1_std': cv_f1_std
        })
        
        gamma_str = f"{gamma:.6f}" if isinstance(gamma, float) else gamma
        print(f"C={C:5.1f}  gamma={gamma_str:>10}  →  "
              f"CV_Acc={cv_acc_mean:.4f}±{cv_acc_std:.4f}  "
              f"CV_F1={cv_f1_mean:.4f}±{cv_f1_std:.4f}")
        
        if cv_acc_mean > best_cv_acc:
            best_cv_acc = cv_acc_mean
            best_params = {'C': C, 'gamma': gamma}
    
    t1 = time.time()
    
    # Mejores resultados por CV
    results_sorted = sorted(results, key=lambda x: x['cv_acc_mean'], reverse=True)
    
    print(f"\n⏱️  Tiempo total: {t1-t0:.1f}s")
    print(f"\n🏆 TOP 5 configuraciones (por CV accuracy en TRAIN):\n")
    for i, r in enumerate(results_sorted[:5], 1):
        g_str = f"{r['gamma']:.6f}" if isinstance(r['gamma'], float) else r['gamma']
        print(f"{i}. C={r['C']:5.1f}  gamma={g_str:>10}  →  "
              f"CV_Acc={r['cv_acc_mean']:.4f}±{r['cv_acc_std']:.4f}  "
              f"CV_F1={r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f}")
    
    print(f"\n✅ Mejor configuración (por CV):")
    g_str = f"{best_params['gamma']:.6f}" if isinstance(best_params['gamma'], float) else best_params['gamma']
    print(f"   C={best_params['C']}  gamma={g_str}")
    print(f"   CV Accuracy: {best_cv_acc:.4f}")
    
    # Evaluación final en test (SOLO si se solicita explícitamente)
    if args.final_eval:
        print(f"\n" + "="*60)
        print("🎯 EVALUACIÓN FINAL EN TEST (solo para reporte final)")
        print("="*60)
        
        # Entrenar con TODOS los datos de train
        print(f"\nEntrenando modelo final con toda la data de train...")
        final_clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=42)
        final_clf.fit(Xtr_sc, ytr)
        
        # Predecir en test
        pred_test = final_clf.predict(Xte_sc)
        test_acc = accuracy_score(yte, pred_test)
        test_f1 = f1_score(yte, pred_test, average='macro')
        
        print(f"\n📊 Resultados en TEST:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Macro F1: {test_f1:.4f}")
        print(f"\n⚠️  Estos resultados son SOLO para evaluación final.")
        print(f"   NO debieron usarse para seleccionar hiperparámetros.")
    else:
        print(f"\n💡 Para evaluar en test (solo al final):")
        print(f"   python scripts/grid_search_svm.py --final_eval")

if __name__ == "__main__":
    main()
