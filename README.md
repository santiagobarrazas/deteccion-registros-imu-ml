# Detección automática de registros IMU defectuosos con ML

Prototipo académico. Un clasificador Random Forest revisa cada prueba de sensores IMU en el momento de la captura y avisa
si los datos quedaron dañados (pérdida de paquetes, sensor desconectado o sin movimiento, saturación, duplicados), para
repetir la prueba mientras el paciente sigue presente.

- **Explicación completa** (problema, contexto, solución y métricas): la página principal del sitio de GitHub Pages
  (`index.html`).
- **Prototipo en vivo:** `app/`, una app de Streamlit que corre en el navegador con [stlite](https://github.com/whitphx/stlite).

**Todos los datos son simulados.** El repositorio no contiene datos personales ni registros de personas reales.

## Resultados

| Métrica | Valor |
|---|---|
| Recall de defectos | 97,4 % |
| Falsas alarmas | 4,8 % |
| Exactitud (6 clases) | 96,3 % |
| F1 macro, validación cruzada de 5 particiones | 0,960 ± 0,008 |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para regenerar los datos simulados, las métricas y los ejemplos:

```bash
python src/train.py
```

Para ver el sitio tal como queda en GitHub Pages, sírvalo como archivos estáticos y abra http://localhost:8000:

```bash
python -m http.server 8000
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `index.html` | Página que explica el proyecto |
| `app/index.html` | Carga la app de Streamlit en el navegador (stlite) |
| `app.py` | Interfaz en Streamlit |
| `src/simulador.py` | Genera registros IMU sintéticos e inyecta las 5 fallas |
| `src/features.py` | Calcula los 13 indicadores de calidad |
| `src/modelo.py` | Configuración del Random Forest y partición de datos |
| `src/train.py` | Simula el dataset, entrena, evalúa y guarda métricas y ejemplos |
| `data/indicadores_simulados.csv` | Indicadores de los 1.500 registros simulados (la app entrena con ellos) |
| `data/ejemplos/` | Un registro simulado por clase |
| `models/metricas.json` | Métricas de evaluación |
