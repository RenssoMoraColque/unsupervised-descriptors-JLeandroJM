# scripts/peek_stl10.py
from torchvision.datasets import STL10

# Ver una imagen del split de entrenamiento
ds = STL10(root="data", split="train", download=False)  # ya existentes
img, label = ds[0]  # PIL.Image, etiqueta int [0..9]
print(img.size, img.mode)   # (96, 96) RGB
img.show()                  # abre Preview en macOS

# También puedes probar 'unlabeled'
ds_u = STL10(root="data", split="unlabeled", download=False)
img_u, _ = ds_u[0]
img_u.show()
