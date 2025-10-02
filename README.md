# Clasificación STL-10 con Fisher Vectors Optimizados
## Pipeline No Supervisado + SVM para Hackathon de Descriptores de Imagen

[![Accuracy](https://img.shields.io/badge/Accuracy-~67%25-green)]() [![Dimensión](https://img.shields.io/badge/Dimensión-3840%2F4096-blue)]() [![Restricciones](https://img.shields.io/badge/Restricciones-✓%20Cumplidas-success)]()

## 1. Descripción del Método

### 🧠 **Arquitectura del Pipeline**
Pipeline avanzado basado en descriptores locales clásicos y codificación estadística de segunda orden:

#### **🔍 Extracción de Características Locales:**
- **SIFT Denso + RootSIFT**: Grid regular multi-escala para cobertura completa
  - *Justificación*: SIFT captura gradientes locales robustos a rotación/escala
  - *RootSIFT*: Mejora discriminación mediante normalización L1→√→L2 (+5-10% accuracy típico)
  - *Grid denso*: Asegura features en todas las regiones vs. keypoints automáticos irregulares

#### **📉 Reducción Dimensional Inteligente:**
- **PCA (128 → 24 dims)** entrenado sobre 20k imágenes unlabeled
  - *Justificación*: Elimina redundancia SIFT (~24 dims capturan 85-90% varianza)
  - *Beneficio*: GMM más estable + 5x menos memoria + mismo poder discriminativo

#### **🎯 Modelado Estadístico:**
- **GMM (K=16, covarianza diagonal)** sobre descriptores PCA
  - *Justificación*: Captura distribución multimodal de patches visuales
  - *K=16*: Balance optimal para STL-10 (10 clases → ~1.6 gaussianas/clase)
  - *Diagonal*: Estándar en Fisher Vectors + computacionalmente eficiente

#### **🚀 Codificación Avanzada:**
- **Fisher Vectors** (gradientes de medias y varianzas) + **SPM** (1×1 + 2×2)
  - *Superioridad vs VLAD*: Incluye información segundo orden (varianza) → +10-15% accuracy
  - *SPM*: Preserva información espacial sin explotar dimensionalidad
  - *5 regiones*: 1 global + 4 cuadrantes = contexto local+global

#### **⚖️ Normalización Multi-Nivel:**
- **RootSIFT**: L1 → √ → L2 (nivel descriptor)
- **Fisher**: Power normalization + L2 (nivel región)  
- **L2 final**: Post-concatenación SPM (nivel imagen)
- *Beneficio*: Cada nivel reduce dominancia de outliers y mejora separabilidad

#### **🎪 Clasificación Final:**
- **SVM RBF** con hiperparámetros optimizados por CV
- *Justificación*: Kernel RBF captura relaciones no-lineales residuales post-Fisher

### ✅ **Cumplimiento de Restricciones del Hackathon:**
- **Dimensión final**: 2×16×24×5 = **3,840 ≤ 4,096** ✓
- **No deep learning**: Solo técnicas clásicas de CV ✓  
- **No supervisión en descriptores**: Etiquetas SOLO en SVM final ✓
- **Dataset correcto**: Unlabeled para training, train/test para evaluación ✓

## 2. Pipeline de Ejecución (Comandos Optimizados)

### 🔄 **FASE 1: Entrenamiento No Supervisado** 
*Usando 20k imágenes UNLABELED (sin etiquetas)*

```bash
# 1. 🔍 Extracción masiva de descriptores SIFT
python scripts/extract_descriptors_unlabeled.py \
  --max_images 20000 \          # Máximo volumen de datos unlabeled
  --max_desc_per_image 500 \    # Balance calidad/memoria
  --output artifacts/desc_sift_unlabeled_20k.npy
# → Genera ~10M descriptores SIFT (128D) para training robusto
```

```bash
# 2. 📉 Entrenamiento PCA para reducción dimensional
python scripts/train_pca.py \
  --desc_path artifacts/desc_sift_unlabeled_20k.npy \
  --n_components 24 \           # Retiene ~85-90% varianza SIFT
  --output artifacts/pca_sift24.joblib
# → PCA sin supervisión: solo análisis de componentes principales
```

```bash
# 3. 🔄 Aplicación PCA a descriptores
python scripts/apply_pca.py \
  --desc_path artifacts/desc_sift_unlabeled_20k.npy \
  --pca_path artifacts/pca_sift24.joblib \
  --output artifacts/desc_pca24_unlabeled_20k.npy  
# → Transforma 128D→24D manteniendo poder discriminativo
```

```bash
# 4. 🎯 Entrenamiento GMM para Fisher Vectors
python scripts/train_gmm_fisher.py \
  --desc_path artifacts/desc_pca24_unlabeled_20k.npy \
  --n_components 16 \           # K=16: balance capacidad/dimensión
  --output artifacts/gmm_fisher16.joblib
# → GMM captura distribución multimodal de patches visuales
```

### 🎨 **FASE 2: Extracción de Features**
*Usando splits etiquetados train/test (5k/8k) pero SIN usar etiquetas*

```bash
# 5. 🚀 Generación Fisher Vectors + SPM
python scripts/extract_fisher_spm.py \
  --pca_path artifacts/pca_sift24.joblib \
  --gmm_path artifacts/gmm_fisher16.joblib \
  --step 6 \                    # Densidad grid SIFT
  --sizes 12 16                 # Multi-escala para robustez
# → Aplica pipeline completo a train/test: SIFT→PCA→Fisher→SPM
# → Genera X_train_fisher.npy, X_test_fisher.npy (3840D cada imagen)
```

### ⚖️ **FASE 3: Evaluación Supervisada**
*Usando etiquetas SOLO aquí*

```bash
# 6a. 🔍 Grid Search automático (recomendado)
python scripts/grid_search_svm.py \
  --cv_folds 5                  # CV estratificado en train únicamente
# → Encuentra automáticamente mejores C, gamma sin tocar test
```

```bash
# 6b. ✅ Evaluación final con mejores hiperparámetros  
python scripts/grid_search_svm.py \
  --cv_folds 5 \
  --final_eval                  # Evalúa en test SOLO una vez
# → Reporte final de accuracy/F1 para submission
```

### 🎚️ **Pipeline Manual (alternativo):**
```bash
# Evaluación directa con parámetros específicos
python scripts/train_eval_svm_rbf.py \
  --C 5.0 \
  --gamma 0.000100              # Valor encontrado por grid search
```

### 💡 **Comandos de Diagnóstico:**
```bash
# Verificar dimensiones generadas
python -c "import numpy as np; X=np.load('features/X_train_fisher.npy'); print(f'Shape: {X.shape}, Max dim: {X.shape[1]}')"

# Verificar archivos generados
ls -la artifacts/ features/
```


### 🎯 **Decisiones de Diseño Clave:**

#### **¿Por qué Fisher Vectors vs VLAD/BoVW?**
- **VLAD**: Solo primer momento (diferencias medias)
- **Fisher**: Primer + segundo momento (medias + varianzas)
- **Ganancia empírica**: 15-20% accuracy en datasets similares
- **Costo**: 2x dimensión pero mejor discriminación

#### **¿Por qué K=16 gaussianas?**
- **K=8**: Subreprenta diversidad visual (underfitting)  
- **K=32**: Dimensión final 6,720 > 4,096 (violación restricción)
- **K=16**: Sweet spot: expresividad + cumplimiento límite

#### **¿Por qué PCA 128→24?**
- **Sin PCA**: GMM inestable en alta dimensión + memoria prohibitiva
- **PCA mayor**: Retiene ruido → GMM overfit
- **24 dims**: Varianza explicada 85-90% + GMM convergente

#### **¿Por qué SPM 1×1+2×2 vs pirámides más profundas?**
- **SPM 1×1**: Info global, pierde localización
- **SPM 1×1+2×2+4×4**: 21 regiones → dimensión explota  


### ⚡ **Eficiencia Temporal**
- **Extracción SIFT (20k unlabeled)**: ~15 min
- **Entrenamiento PCA**: ~1 min  
- **Entrenamiento GMM**: ~2 min
- **Extracción Fisher (13k train+test)**: ~6 min
- **Grid Search SVM (25 configs)**: ~8 min
- **TOTAL pipeline**: ~32 min


### 🔧 **Setup del Entorno**
```bash
# 1. Clonar repositorio
git clone https://github.com/RenssoMoraColque/unsupervised-descriptors-JLeandroJM.git
cd unsupervised-descriptors-JLeandroJM

# 2. Instalar dependencias
pip install -r scripts/requirements.txt
# Incluye: numpy, scipy, scikit-learn, scikit-image, opencv-contrib-python, torchvision, tqdm, joblib

# 3. Descargar STL-10 (automático en primer uso)
python scripts/download_stl10.py --root data

# 4. Verificar estructura
tree data/ -L 2
```

### 📁 **Estructura de Archivos Generados**
```
proyecto/
├── data/
│   └── stl10_binary/          # Dataset STL-10 (descarga automática)
├── artifacts/                 # Modelos entrenados (no supervisado)
│   ├── desc_sift_unlabeled_20k.npy      # ~400MB
│   ├── pca_sift24.joblib                # ~1MB  
│   ├── desc_pca24_unlabeled_20k.npy     # ~80MB
│   └── gmm_fisher16.joblib              # ~2MB
├── features/                  # Descriptores finales
│   ├── X_train_fisher.npy               # 5k × 3840 (~75MB)
│   ├── X_test_fisher.npy                # 8k × 3840 (~120MB)  
│   ├── y_train.npy                      # Etiquetas train
│   └── y_test.npy                       # Etiquetas test
└── scripts/                   # Pipeline ejecutable
```

## REPOSITORIO

https://github.com/RenssoMoraColque/unsupervised-descriptors-JLeandroJM



