# src/preprocess.py
from __future__ import annotations
import io, random
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# ---------- Helpers básicos ----------
def _to_pil_rgb(img) -> Image.Image:
    """Acepta PIL o np.ndarray (H,W,3) y devuelve PIL RGB 96x96 uint8."""
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
    # Asumimos STL-10 = 96x96; si no, ajusta:
    if pil.size != (96, 96):
        pil = pil.resize((96, 96), Image.BILINEAR)
    return pil

def to_numpy_uint8(img) -> np.ndarray:
    """Devuelve np.uint8 RGB (H,W,3)."""
    return np.array(_to_pil_rgb(img), dtype=np.uint8)

def to_gray_uint8(img) -> np.ndarray:
    """Devuelve np.uint8 GRAY (H,W) para SIFT/RootSIFT."""
    return np.array(_to_pil_rgb(img).convert("L"), dtype=np.uint8)

# ---------- Augmentación suave (solo para aprendizaje no supervisado) ----------
def augment_soft(img,
                 rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Aplica UNA transformación suave aleatoria y devuelve RGB uint8.
    Usar SOLO al entrenar PCA/k-means con 'unlabeled'.
    """
    rng = rng or random
    pil = _to_pil_rgb(img)
    choice = rng.choice(["none", "blur", "rot", "jpeg", "bc", "scale"])
    if choice == "blur":
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif choice == "rot":
        angle = rng.choice([-15, -10, -5, 5, 10, 15])
        pil = pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    elif choice == "jpeg":
        q = rng.choice([70, 80, 90])
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    elif choice == "bc":
        # brillo/contraste leves (1.0 = igual)
        pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.9, 1.1))
        pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.9, 1.1))
    elif choice == "scale":
        s = rng.uniform(0.9, 1.1)
        sz = (max(1, int(96*s)), max(1, int(96*s)))
        pil = pil.resize(sz, Image.BILINEAR).resize((96, 96), Image.BILINEAR)
    # "none" = sin cambios
    return np.array(pil, dtype=np.uint8)

# ---------- Ataques de robustez del README (para evaluación) ----------
def apply_attack(img, kind: str) -> np.ndarray:
    """
    kind ∈ {'blur','rot+15','rot-15','scale0.8','scale1.2','bc','jpeg40'}
    Devuelve RGB uint8 atacado (96x96).
    """
    pil = _to_pil_rgb(img)
    if kind == "blur":
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif kind == "rot+15":
        pil = pil.rotate(15, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    elif kind == "rot-15":
        pil = pil.rotate(-15, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    elif kind == "scale1.2":
        pil = pil.resize((int(96*1.2), int(96*1.2)), Image.BILINEAR).resize((96, 96), Image.BILINEAR)
    elif kind == "scale0.8":
        pil = pil.resize((int(96*0.8), int(96*0.8)), Image.BILINEAR).resize((96, 96), Image.BILINEAR)
    elif kind == "bc":
        pil = ImageOps.autocontrast(pil)
    elif kind == "jpeg40":
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=40)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    else:
        raise ValueError(f"attack desconocido: {kind}")
    return np.array(pil, dtype=np.uint8)

# ---------- Semillas reproducibles ----------
def make_rng(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng
