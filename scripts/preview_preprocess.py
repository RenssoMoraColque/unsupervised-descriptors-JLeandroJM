# scripts/preview_preprocess.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # añade raíz del repo al path

from PIL import Image
from torchvision.datasets import STL10
from src.preprocess import augment_soft, apply_attack, make_rng

ROOT = Path(__file__).resolve().parents[1] / "scripts/data"
print("Usando dataset en:", ROOT)

# ¡OJO!: sin transform (STL10 ya da PIL.Image)
ds = STL10(root=str(ROOT), split="train", download=False)
img, _ = ds[0]   # img es PIL.Image (96x96 RGB)

# Guarda original
Image.Image.save(img, "preview_original.png")

# Augment suave (solo demo)
rng = make_rng(0)
Image.fromarray(augment_soft(img, rng)).save("preview_aug_soft.png")

# Ataques de robustez (demo)
for k in ["blur","rot+15","rot-15","scale0.8","scale1.2","bc","jpeg40"]:
    Image.fromarray(apply_attack(img, k)).save(f"preview_attack_{k}.png")

print("OK. Archivos: preview_original.png, preview_aug_soft.png, preview_attack_*.png")
