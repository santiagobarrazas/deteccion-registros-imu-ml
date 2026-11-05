"""Entrena el clasificador de calidad, lo evalúa y guarda modelo, métricas y ejemplos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import cross_val_score

from features import NOMBRES, extraer
from modelo import crear_modelo, particionar
from simulador import CLASES, simular_dataset, simular_registro

RAIZ = Path(__file__).resolve().parents[1]
N_POR_CLASE = 250


def construir_tabla(dataset):
    X = pd.DataFrame([extraer(r) for r, _ in dataset], columns=NOMBRES)
    y = np.array([c for _, c in dataset])
    return X, y


def main():
    X, y = construir_tabla(simular_dataset(N_POR_CLASE))
    # La app reentrena con esta tabla en el navegador: así no depende de un modelo serializado.
    X.round(6).assign(clase=y).to_csv(RAIZ / "data" / "indicadores_simulados.csv", index=False)
    X_tr, X_te, y_tr, y_te = particionar(X, y)

    modelo = crear_modelo()
    cv = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring="f1_macro")
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)

    clases = list(CLASES)
    defecto_real, defecto_pred = y_te != "OK", pred != "OK"
    metricas = {
        "n_entrenamiento": len(X_tr),
        "n_prueba": len(X_te),
        "f1_macro_cv": [round(float(cv.mean()), 4), round(float(cv.std()), 4)],
        "reporte": classification_report(y_te, pred, labels=clases, output_dict=True, zero_division=0),
        "matriz_confusion": confusion_matrix(y_te, pred, labels=clases).tolist(),
        "clases": clases,
        "recall_defectos": round(float(recall_score(defecto_real, defecto_pred)), 4),
        "falsas_alarmas": round(float((defecto_pred & ~defecto_real).sum() / (~defecto_real).sum()), 4),
        "rango_ok": {c: [round(float(v), 4) for v in X_tr[y_tr == "OK"][c].quantile([0.02, 0.98])] for c in NOMBRES},
        "importancia": dict(sorted(zip(NOMBRES, modelo.feature_importances_.round(4).tolist()), key=lambda kv: -kv[1])),
    }

    (RAIZ / "models").mkdir(exist_ok=True)
    (RAIZ / "models" / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))

    ejemplos = RAIZ / "data" / "ejemplos"
    ejemplos.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    for clase in clases:
        registro = simular_registro(rng, clase, duracion_s=25)
        (ejemplos / f"ejemplo_{clase.lower()}.json").write_text(json.dumps(registro))

    print(classification_report(y_te, pred, labels=clases, zero_division=0))
    print(f"F1 macro CV: {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"Recall de defectos: {metricas['recall_defectos']:.3f}  Falsas alarmas: {metricas['falsas_alarmas']:.3f}")


if __name__ == "__main__":
    main()
