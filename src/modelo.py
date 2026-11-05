"""Configuración del clasificador, compartida por el entrenamiento y la app (que reentrena en el navegador)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def particionar(X, y):
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def crear_modelo():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=0)
