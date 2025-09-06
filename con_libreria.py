from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # =======================
    # 1) Cargar dataset
    # =======================
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Label'] = data.target
    print(df.head())
    X = df[data.feature_names].values
    y = df['Label'].values


    # =======================
    # 2) Train/Test split
    # =======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # =======================
    # 3) Pipeline (Scaler + LogReg)
    # =======================
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])

    pipe.fit(X_train, y_train)

    # =======================
    # 4) Predicciones y métricas
    # =======================
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Benigno (0)', 'Maligno (1)'])
    auc = roc_auc_score(y_test, y_prob)

    acc = (y_pred == y_test).mean()
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    print("\n=== MÉTRICAS (TEST) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC AUC:   {auc:.3f}")
    print("\n=== Classification Report ===\n", report)

    # =======================
    # 5) Matriz de Confusión (heatmap)
    # =======================
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Matriz de Confusión - Breast Cancer (TEST)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benigno (0)', 'Maligno (1)'])
    ax.set_yticklabels(['Benigno (0)', 'Maligno (1)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()