import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Simulación de datos de probabilidad y etiquetas verdaderas
np.random.seed(42)
y_true = np.random.randint(0, 2, size=1000)  # 0 o 1 (clases reales)
y_scores = np.random.rand(1000)  # Probabilidades del modelo

# Cálculo de Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Título de la aplicación
st.title("Curva Precision-Recall Interactiva")

# **Barra deslizante para cambiar el umbral**
odd = st.slider("Selecciona el valor de odd", min_value=0.5, max_value=0.9, step=0.1, value=0.7)

# **Cálculo de nuevo umbral basado en odd**
threshold_opt = np.percentile(y_scores, (1 - odd) * 100)
y_pred = (y_scores >= threshold_opt).astype(int)

# Nuevo cálculo de precisión y recall con umbral ajustado
precision_adjusted = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
recall_adjusted = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)

# **Gráfico dinámico**
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall, precision, color='blue', label="Curva PR")
ax.scatter(recall_adjusted, precision_adjusted, color='red', s=100, label=f'Punto Actual (odd={odd:.1f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Curva Precision-Recall con Odd = {odd:.1f}')
ax.legend()
ax.grid()

# Mostrar gráfico en Streamlit
st.pyplot(fig)
