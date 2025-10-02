#!/usr/bin/env python3
import argparse, os, sys, math
from pathlib import Path
from torchvision.datasets import STL10
from torchvision import transforms

def human_gib(nbytes):
    return f"{nbytes / (1024**3):.2f} GiB"

def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try: total += p.stat().st_size
            except: pass
    return total

def main():
    ap = argparse.ArgumentParser(description="Descarga y verifica STL-10")
    ap.add_argument("--root", type=Path, default=Path("data"),
                    help="Carpeta raíz para el dataset (default: ./data)")
    ap.add_argument("--splits", nargs="+",
                    default=["unlabeled","train","test"],
                    choices=["unlabeled","train","test","train+unlabeled"],
                    help="Splits a descargar")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToPILImage()])

    # Descarga por split (torchvision no re-descarga si ya existe)
    for sp in args.splits:
        print(f"-> Descargando/verificando split: {sp}")
        ds = STL10(root=str(args.root), split=sp, download=True, transform=tfm)
        print(f"   OK: {sp} contiene {len(ds)} imágenes.")

    # Reporte de tamaños
    bin_dir = args.root / "stl10_binary"
    if bin_dir.exists():
        size_bytes = dir_size(bin_dir)
        print(f"\nCarpeta de datos: {bin_dir}")
        print(f"Tamaño en disco (binarios + metadatos): {human_gib(size_bytes)}")
    else:
        print("\nAdvertencia: no se encuentra stl10_binary todavía (¿falló la descarga?).")
        sys.exit(1)

    # Recordatorio de tamaños teóricos descomprimidos
    px_per_img = 96*96*3
    def imgs(n): return n*px_per_img
    total_theoretical = imgs(100_000) + imgs(5_000) + imgs(8_000)
    print("\nTamaños teóricos (descomprimidos):")
    print(f" - unlabeled (100k): {human_gib(imgs(100_000))}")
    print(f" - train (5k)     : {human_gib(imgs(5_000))}")
    print(f" - test  (8k)     : {human_gib(imgs(8_000))}")
    print(f" - TOTAL          : {human_gib(total_theoretical)}")

    print("\nListo. Si vuelves a correr este script, no re-descargará nada ya existente.")

if __name__ == "__main__":
    main()
