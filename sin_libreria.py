from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Label'] = data.target
print(df.head())

# ------------------ Funciones base ------------------

def sigmoid(z):
    """Función sigmoide para convertir valores en probabilidades"""
    return 1 / (1 + np.exp(-z))

def initialize_params(n_features):
    """Inicializa los pesos y el sesgo en ceros"""
    W = np.zeros((n_features, 1))
    b = 0.0
    return W, b

def forward_propagation(X, W, b):
    """Calcula y_hat = sigmoid(XW + b)"""
    z = np.dot(X, W) + b
    return sigmoid(z)

def compute_loss(y, y_hat):
    """Calcula el log-loss"""
    m = y.shape[0]
    # Evita log(0) con clipping
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    loss = - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss

def backward_propagation(X, y, y_hat):
    """Calcula gradientes de W y b"""
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db

def update_params(W, b, dw, db, lr):
    """Actualiza pesos y bias"""
    W -= lr * dw
    b -= lr * db
    return W, b

def predict(X, W, b, threshold=0.5):
    """Predicciones binarias basado en la probabilidad"""
    y_prob = forward_propagation(X, W, b)
    return (y_prob >= threshold).astype(int)

# ------------------ Entrenamiento ------------------
def split_train_test(X, y, test_size=0.2, random_state=1):
    """
    Divide X e y en train y test de forma manual.
    """
    np.random.seed(random_state)
    m = X.shape[0]                 # número total de muestras
    indices = np.arange(m)
    np.random.shuffle(indices)      # mezcla aleatoria de índices
    
    test_count = int(m * test_size) # cantidad para test
    
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def logistic_regression(X, y, lr=0.01, epochs=1000, verbose=True):
    """
    X: numpy array (m, n)
    y: numpy array (m, 1)
    """
    m, n = X.shape
    W, b = initialize_params(n)
    losses = []

    for epoch in range(epochs):
        # Forward
        y_hat = forward_propagation(X, W, b)
        
        # Loss
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # Backward
        dw, db = backward_propagation(X, y, y_hat)

        # Update
        W, b = update_params(W, b, dw, db, lr)

        # Opcional: mostrar progreso
        if verbose and (epoch+1) % (epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    return W, b, losses
# ------------------ Matriz de Confusion ------------------
def confusion_matrix(y_true, y_pred):
    """
    Calcula matriz de confusión 2x2:
    [[TN, FP],
     [FN, TP]]
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP],
                     [FN, TP]])

def classification_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return accuracy, precision, recall, f1

# ------------------ Ejemplo con datos ------------------

# Separar X y y
# (m, 30) con todas las columnas numéricas
X_raw = df[data.feature_names].values.astype(float)
y = df[['Label']].values.astype(float)  # (m,1)

#Separar en train y test
X_train_raw, X_test_raw, y_train, y_test = split_train_test(X_raw, y, test_size=0.2, random_state=42)

print(f"Tamaño Train: {X_train_raw.shape[0]} muestras")
print(f"Tamaño Test : {X_test_raw.shape[0]} muestras")

# Estandariza
mu = X_train_raw.mean(axis=0, keepdims=True)
sd = X_train_raw.std(axis=0, keepdims=True) + 1e-12

X_train = (X_train_raw - mu) / sd
X_test  = (X_test_raw - mu) / sd 

# Entrenar (usa y_train)
W, b, losses = logistic_regression(X_train, y_train, lr=0.1, epochs=1000, verbose=True)

# Predicciones SOLO en test
y_pred = predict(X_test, W, b)

# Precisión en test
accuracy = np.mean(y_pred == y_test)
print("\nPesos finales:\n", W)
print("Bias final:", b)
print("Precisión del modelo (TEST):", accuracy)

# --------- Guardar predicciones en df SOLO en filas de test (sin romper tamaños) ---------
# Reproducir los índices de test con el mismo random_state y tamaño:
m = X_raw.shape[0]
np.random.seed(1)
indices = np.arange(m); np.random.shuffle(indices)
test_count = int(m * 0.2)
test_idx = indices[:test_count]

df['Predicciones'] = np.nan
df.loc[test_idx, 'Predicciones'] = y_pred.ravel().astype(int)

print("\nPredicciones (muestra):\n", df[['Label','Predicciones']].dropna().head())

# --------------- Matriz de Confusion (TEST) ------------------
cm = confusion_matrix(y_test, y_pred)
accuracy, precision, recall, f1 = classification_metrics(cm)

fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap='Blues')

# Títulos
ax.set_title("Matriz de Confusión - Breast Cancer (TEST)")
ax.set_xlabel("Predicciones")
ax.set_ylabel("Valores Reales")

# Etiquetas
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Benigno (0)', 'Maligno (1)'])
ax.set_yticklabels(['Benigno (0)', 'Maligno (1)'])

# Números en celdas
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# -------------- Metricas -----------------

print("\n=== MÉTRICAS DEL MODELO (TEST) ===")
print(f"Accuracy (Exactitud): {accuracy:.3f}")
print(f"Precision (Precisión): {precision:.3f}")
print(f"Recall (Sensibilidad): {recall:.3f}")
print(f"F1-score: {f1:.3f}")

print("\n=== Matriz de Confusión (TEST) ===")
print(pd.DataFrame(cm,
                   index=['Real: Benigno (0)','Real: Maligno (1)'],
                   columns=['Pred: Benigno (0)','Pred: Maligno (1)']))
