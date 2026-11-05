"""Extracción de características de calidad de un registro IMU (JSON de la app)."""
import numpy as np

from simulador import ACC_RANGE_G, DT_MS, GYRO_RANGE_DPS

DESCRIPCION = {
    "n_sensores": "Sensores presentes en el registro",
    "duracion_s": "Duración de la prueba (s)",
    "ratio_muestras_lados": "Muestras del sensor con menos datos / el de más datos",
    "cobertura_min": "Fracción del tiempo cubierta por el sensor que menos cubre",
    "hueco_max_s": "Hueco más largo sin datos (s)",
    "frac_huecos": "Fracción de intervalos > 2 periodos de muestreo",
    "frac_duplicados": "Fracción de muestras repetidas (mismo sensor y secuencia)",
    "frac_ts_desordenado": "Fracción de marcas de tiempo que retroceden",
    "gyro_std_min": "Variabilidad del giroscopio del sensor más quieto (°/s)",
    "gyro_std_ratio": "Variabilidad del giro: sensor más quieto / más activo",
    "frac_saturacion": "Fracción de muestras en el límite del rango del sensor",
    "acc_media_norma": "Norma media de la aceleración (g)",
    "corr_lados": "Correlación entre la actividad de ambos tobillos",
}
NOMBRES = list(DESCRIPCION)


def _por_sensor(imu):
    sensores = {}
    for m in imu:
        sensores.setdefault(m["deviceId"], []).append(m)
    return sensores


def _arrays(muestras):
    acc = np.array([[m["accelerometer"][k] for k in "xyz"] for m in muestras], dtype=float)
    gyro = np.array([[m["gyroscope"][k] for k in "xyz"] for m in muestras], dtype=float)
    ts = np.array([m["timestamp"] for m in muestras], dtype=float)
    seq = np.array([m["sequenceNumber"] for m in muestras])
    return acc, gyro, ts, seq


def extraer(registro):
    imu = registro.get("imuData") or []
    sensores = _por_sensor(imu)
    if not sensores:
        return dict.fromkeys(NOMBRES, 0.0)

    datos = {d: _arrays(m) for d, m in sensores.items()}
    t0 = min(ts.min() for _, _, ts, _ in datos.values())
    t1 = max(ts.max() for _, _, ts, _ in datos.values())
    duracion = max((t1 - t0) / 1000, 1e-3)

    conteos, coberturas, huecos_max, frac_huecos, dups, desorden, gyro_std, sat, acc_norma = ([] for _ in range(9))
    actividad = {}
    for dev, (acc, gyro, ts, seq) in datos.items():
        conteos.append(len(ts))
        dups.append(1 - len(set(seq.tolist())) / len(seq))
        desorden.append(np.mean(np.diff(ts) < 0) if len(ts) > 1 else 0)
        orden = np.sort(np.unique(ts))
        dt = np.diff(np.concatenate([[t0], orden, [t1]]))
        huecos_max.append(dt.max() / 1000)
        frac_huecos.append(np.mean(dt > 2.5 * DT_MS))
        coberturas.append(min(1.0, len(orden) * DT_MS / 1000 / duracion))
        gyro_std.append(np.linalg.norm(gyro, axis=1).std())
        sat.append(np.mean((np.abs(acc) >= ACC_RANGE_G * 0.98).any(1) | (np.abs(gyro) >= GYRO_RANGE_DPS * 0.98).any(1)))
        acc_norma.append(np.linalg.norm(acc, axis=1).mean())
        rejilla = np.arange(t0, t1, 200)
        actividad[dev] = np.interp(rejilla, ts[np.argsort(ts)], np.linalg.norm(gyro, axis=1)[np.argsort(ts)])

    corr = 0.0
    if len(actividad) >= 2:
        a, b = list(actividad.values())[:2]
        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])

    return {
        "n_sensores": float(len(sensores)),
        "duracion_s": duracion,
        "ratio_muestras_lados": min(conteos) / max(conteos) if len(conteos) > 1 else 0.0,
        "cobertura_min": min(coberturas) if len(coberturas) > 1 else 0.0,
        "hueco_max_s": max(huecos_max),
        "frac_huecos": max(frac_huecos),
        "frac_duplicados": max(dups),
        "frac_ts_desordenado": max(desorden),
        "gyro_std_min": min(gyro_std),
        "gyro_std_ratio": min(gyro_std) / max(max(gyro_std), 1e-6),
        "frac_saturacion": max(sat),
        "acc_media_norma": float(np.mean(acc_norma)),
        "corr_lados": corr,
    }


def series(registro):
    """Series por sensor para graficar: tiempo (s), norma del giroscopio y de la aceleración."""
    salida = {}
    for dev, muestras in _por_sensor(registro.get("imuData") or []).items():
        acc, gyro, ts, _ = _arrays(muestras)
        orden = np.argsort(ts, kind="stable")
        salida[dev] = {
            "t": ts[orden] / 1000,
            "gyro": np.linalg.norm(gyro, axis=1)[orden],
            "acc": np.linalg.norm(acc, axis=1)[orden],
        }
    return salida
