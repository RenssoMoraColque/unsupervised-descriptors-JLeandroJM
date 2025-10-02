[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/_q5OXIZr)
# 🏆 Hackathon de Descriptores de Imagen (sin Deep Learning) con STL-10

## 🎯 Objetivo
Desarrollar **descriptores de imagen** y pipelines basados **únicamente en técnicas no-deep-learning** (clásicas) usando **STL-10 (split no etiquetado)** para aprender representaciones, y evaluar su utilidad para tareas supervisadas y de clustering usando las etiquetas **solo en la fase de evaluación**.

---

## 📁 Dataset y restricciones de uso
- **Dataset principal:** [STL-10](https://cs.stanford.edu/~acoates/stl10/).
  - Usar el **split `unlabeled` (100.000 imágenes)** para cualquier aprendizaje/clusterización no supervisada.  
  - Las **etiquetas solo pueden usarse en la fase de evaluación**: entrenar clasificadores simples (p. ej. SVM, k-NN) usando la porción etiquetada estándar (5k train / 8k test).  
- **Prohibido:**
  - Entrenar modelos con supervisión usando etiquetas en la fase de construcción del descriptor.  
  - Usar redes neuronales profundas (pretrained o entrenamiento desde cero).  
- **Permitido:**
  - Usar técnicas clásicas no supervisadas (PCA, k-means, GMM, ICA, etc.).  
  - Usar transformaciones/augmentations para robustez (rotaciones leves, flip, blur, etc.).  

---

## 🧰 Métodos permitidos
- **Detectores / descriptores locales clásicos:** SIFT, SURF, ORB, BRIEF, BRISK.  
- **Descriptores globales:** HOG, LBP, GIST, color histograms.  
- **Bag-of-visual-words y derivados:** SIFT + k-means (BoVW), VLAD, Fisher Vectors.  
- **Reducción de dimensión:** PCA, ICA, NMF, Random Projection.  
- **Clustering / modelado no supervisado:** k-means, GMM, spectral clustering, agglomerative.  
- **Hashing / índices:** LSH, Product Quantization (para retrieval).  
- **Clasificadores simples (solo en evaluación):** SVM lineal/RBF, k-NN, logistic regression.  

> ⚠️ Nota: si el algoritmo produce vectores de longitud variable (ej. sets de keypoints), deben aplicar un **pooling/encoding** (BoVW, VLAD, Fisher) para producir vectores de dimensión fija.

---

## 🧾 Protocolo de evaluación

### 1. Fase de aprendizaje (no supervisada)  
- Usar **solo las 100k imágenes `unlabeled`** para entrenar/ajustar cualquier modelo no supervisado (ej. construir codebook k-means, aprender GMM, PCA, etc.).  
- También se permite usar las 5k/8k etiquetadas **sin sus etiquetas** para aumentar el set no supervisado.  

### 2. Extracción de descriptores  
- Para cada imagen (train/test), extraer el descriptor final.  
- Resultado esperado: **vectores de dimensión fija** por imagen.  

### 3. Entrenamiento del evaluador (supervisado)  
- Con representaciones ya obtenidas, usar las etiquetas del split de entrenamiento (5k) para entrenar un clasificador simple (ej. SVM lineal o k-NN).  

### 4. Evaluación  
- Usar el split de test (8k). Reportar:
  - **Accuracy (Top-1)**  
  - **Macro F1**  
  - **mAP** (si hacen retrieval, opcional)  
  - **NMI / ARI / Purity** (si entregan clustering, opcional)  

- **Evaluación de robustez:** aplicar transformaciones y reportar caída en accuracy:
  - Blur gaussiano σ=1.5  
  - Rotación ±15°  
  - Escala 0.8–1.2  
  - Cambios de brillo/contraste  
  - JPEG compression (calidad 40%)  

### 5. Repetibilidad  
- Ejecutar cada experimento **al menos 3 veces** con distintas semillas.  
- Reportar **media ± desviación estándar**.  

### 6. Restricciones prácticas  
- Dimensión máxima del descriptor: **4096**.  
- Reportar tiempo promedio de extracción por imagen.  

---

## 🏆 Criterios de juzgamiento
- **Performance (accuracy):** 40%  
- **Robustez (caída ante transformaciones):** 20%  
- **Eficiencia (tiempo/memoria):** 15%  
- **Creatividad / justificación del método:** 15%  
- **Reproducibilidad y claridad (repositorio/documentación):** 10%  

---

## 📦 Entregables
1. Código completo (idealmente en GitHub).  
2. Script reproducible que:  
   - Descargue STL-10.  
   - Entrene/ajuste descriptor con `unlabeled`.  
   - Extraiga descriptores de train/test.  
   - Entrene clasificadores y genere métricas.  
3. Informe técnico (máx. 4 páginas):  
   - Descripción del método.  
   - Hiperparámetros.  
   - Resultados (tablas + gráficos).  
   - Análisis.  
4. Archivo `requirements.txt` con dependencias.  
5. (Opcional) Notebook demo en Google Colab.  

---

## 🧪 Baselines sugeridas
- **Baseline A (rápida):**  
  HOG (3780 dim) → PCA(512) → SVM lineal.  
- **Baseline B (BoVW):**  
  SIFT (dense) → k-means K=512 → BoVW histograma L2 → SVM.  
- **Baseline C (global simple):**  
  Color histogram (HSV 64 bins) + LBP → concatenado → k-NN.  

---

## 🕒 Cronograma
- **Día 0:** Lanzamiento e inscripciones.  
- **Día 1–14:** Desarrollo (2 semanas).  
- **Día 15–16:** Entrega final.  
- **Día 17:** Presentaciones (5–10 min por equipo).  
- **Día 17:** Premiación.  

---

## 💻 Stack tecnológico recomendado
- Python 3.10+  
- [OpenCV](https://opencv.org/)  
- [scikit-image](https://scikit-image.org/)  
- [scikit-learn](https://scikit-learn.org/stable/)  
- [numpy](https://numpy.org/), [scipy](https://scipy.org/)  
- Opcionales: [faiss](https://github.com/facebookresearch/faiss), [pyflann](https://www.cs.ubc.ca/research/flann/), [gensim](https://radimrehurek.com/gensim/)  

---

## ✅ Snippet baseline en Python (HOG + PCA + SVM)
```python
from torchvision.datasets import STL10
from torchvision import transforms
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1) Cargar imágenes etiquetadas (solo para evaluación)
transform = transforms.Compose([transforms.ToPILImage()])
train_ds = STL10(root='./data', split='train', download=True, transform=transform)
test_ds  = STL10(root='./data', split='test', download=True, transform=transform)

def img_to_hog(img_pil):
    img = np.array(img_pil.convert('L'))  # gris
    f = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    return f

# Extraer HOG de train/test
X_train = np.array([img_to_hog(x[0]) for x in train_ds])
y_train = np.array([x[1] for x in train_ds])
X_test  = np.array([img_to_hog(x[0]) for x in test_ds])
y_test  = np.array([x[1] for x in test_ds])

# 2) Escalado + PCA
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

pca = PCA(n_components=512, random_state=0).fit(X_train_s)
X_train_p = pca.transform(X_train_s)
X_test_p  = pca.transform(X_test_s)

# 3) SVM
clf = SVC(kernel='linear', C=1.0, random_state=0)
clf.fit(X_train_p, y_train)
acc = clf.score(X_test_p, y_test)
print(f'Accuracy HOG+PCA512+SVM: {acc:.4f}')
