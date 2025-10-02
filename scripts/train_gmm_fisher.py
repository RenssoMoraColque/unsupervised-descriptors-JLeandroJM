#!/usr/bin/env python3
"""
Entrena un GMM para Fisher Vectors sobre descriptores SIFT+PCA
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.mixture import GaussianMixture
from joblib import dump, load
import argparse

def main():
    ap = argparse.ArgumentParser(description="Entrenar GMM para Fisher Vectors")
    ap.add_argument("--desc_path", type=str, default=None,
                    help="Ruta al .npy con descriptores PCA (default: busca en artifacts/)")
    ap.add_argument("--n_components", type=int, default=32,
                    help="Número de componentes gaussianas (K)")
    ap.add_argument("--output", type=str, default=None,
                    help="Ruta de salida para el GMM (.joblib)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limitar número de descriptores para entrenamiento")
    args = ap.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    ART = ROOT / "artifacts"
    ART.mkdir(exist_ok=True)
    
    # Buscar descriptores PCA en artifacts si no se especifica ruta
    if args.desc_path is None:
        print("🔍 Buscando descriptores PCA en artifacts/...")
        desc_files = list(ART.glob("desc_pca*.npy"))
        
        if not desc_files:
            print("❌ No se encontraron archivos desc_pca*.npy en artifacts/")
            print("\n📋 PREREQUISITO: Necesitas generar descriptores SIFT+PCA primero.")
            print("\n🔧 Pasos para generar los descriptores:")
            print("   1. Ejecuta el script de extracción de descriptores:")
            print("      python scripts/extract_descriptors.py --output artifacts/desc_pca_sample.npy")
            print("\n   O si ya tienes descriptores SIFT sin PCA:")
            print("   2. Entrena PCA sobre descriptores SIFT:")
            print("      python scripts/train_pca.py --n_components 24")
            print("   3. Aplica PCA a los descriptores:")
            print("      python scripts/apply_pca.py --output artifacts/desc_pca_sample.npy")
            
            print("\n📂 Archivos actualmente en artifacts/:")
            art_files = sorted(ART.glob("*"))
            if art_files:
                for f in art_files:
                    size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
                    tipo = "📄" if f.is_file() else "📁"
                    if f.is_file():
                        print(f"  {tipo} {f.name} ({size_mb:.1f} MB)")
                    else:
                        print(f"  {tipo} {f.name}/")
            else:
                print("  (vacío)")
            sys.exit(1)
        
        # Usar el primero encontrado o mostrar opciones
        if len(desc_files) == 1:
            desc_path = desc_files[0]
            print(f"✓ Usando: {desc_path.name}")
        else:
            print(f"✓ Encontrados {len(desc_files)} archivos:")
            for i, f in enumerate(desc_files, 1):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {i}. {f.name} ({size_mb:.1f} MB)")
            desc_path = desc_files[0]
            print(f"→ Usando por defecto: {desc_path.name}")
    else:
        desc_path = Path(args.desc_path)
        if not desc_path.exists():
            print(f"❌ No existe: {desc_path}")
            sys.exit(1)
    
    # Cargar descriptores
    print(f"\n📂 Cargando descriptores desde {desc_path}...")
    desc = np.load(desc_path)
    print(f"   Shape original: {desc.shape}")
    print(f"   Tamaño: {desc.nbytes / (1024*1024):.1f} MB")
    
    if args.max_samples and len(desc) > args.max_samples:
        print(f"   Submuestreando a {args.max_samples} descriptores...")
        idx = np.random.choice(len(desc), args.max_samples, replace=False)
        desc = desc[idx]
        print(f"   Shape final: {desc.shape}")
    
    # Entrenar GMM con covarianzas diagonales
    print(f"\n🔧 Entrenando GMM con {args.n_components} componentes...")
    print(f"   (esto puede tomar varios minutos...)")
    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type='diag',  # diagonal es estándar para FV
        max_iter=100,
        n_init=1,
        verbose=2,
        random_state=42
    )
    gmm.fit(desc)
    
    # Guardar modelo
    output_path = Path(args.output) if args.output else (ART / f"gmm_fisher{args.n_components}.joblib")
    dump(gmm, output_path)
    
    print(f"\n✅ GMM guardado en: {output_path}")
    print(f"   - Componentes (K): {gmm.n_components}")
    print(f"   - Dim descriptores (d): {gmm.means_.shape[1]}")
    print(f"   - Dim Fisher Vector por región: {2 * gmm.n_components * gmm.means_.shape[1]}")
    print(f"   - Dim total con SPM (5 regiones): {5 * 2 * gmm.n_components * gmm.means_.shape[1]}")
    print(f"   - Log-likelihood: {gmm.score(desc[:1000]):.2f}")
    print(f"\n💡 Siguiente paso:")
    print(f"   python scripts/extract_vlad_spm.py --gmm_path {output_path}")

if __name__ == "__main__":
    main()
