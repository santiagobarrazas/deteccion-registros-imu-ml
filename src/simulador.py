"""Simulador de registros IMU de la prueba TAPPING_TOE con fallas inyectadas.

Genera registros con el esquema JSON de una app de captura en tablet (imuData con acelerómetro
en g, giroscopio en °/s, ~50 Hz, un sensor por tobillo). Todo es sintético: no hay datos de personas.
"""
import numpy as np

DT_MS = 19
DEVICES = ("LEFT-ANKLE", "RIGHT-ANKLE")
ACC_RANGE_G = 4.0
GYRO_RANGE_DPS = 500.0

CLASES = {
    "OK": "Registro válido",
    "PERDIDA_PAQUETES": "Pérdida de paquetes / huecos en la señal",
    "SENSOR_DESCONECTADO": "Un sensor se desconectó durante la prueba",
    "SENSOR_SIN_MOVIMIENTO": "Sensor sin movimiento (mal puesto o no colocado)",
    "SATURACION": "Saturación del sensor (rango excedido)",
    "DUPLICADO": "Muestras duplicadas (subida concurrente)",
}


def _senal_tobillo(rng, n, intensidad, frecuencia, fase):
    """Golpeteo del pie: ráfagas periódicas en el giroscopio y la aceleración, más gravedad."""
    t = np.arange(n) * DT_MS / 1000
    golpe = np.maximum(0, np.sin(2 * np.pi * frecuencia * t + fase)) ** 3
    fatiga = 1 - 0.3 * t / t[-1] * rng.uniform(0, 1)
    pulso = golpe * fatiga * intensidad
    gravedad = np.array([0.97, rng.uniform(-0.3, 0.3), rng.uniform(-0.25, 0)])
    acc = gravedad + np.outer(pulso, rng.uniform([0.2, 0.15, 0.05], [0.5, 0.4, 0.2])) * rng.choice([-1, 1], 3)
    gyro = np.outer(pulso, rng.uniform([40, 15, 20], [90, 40, 45])) * rng.choice([-1, 1], 3)
    acc += rng.normal(0, 0.02, acc.shape)
    gyro += rng.normal(0, 1.5, gyro.shape)
    return np.clip(acc, -ACC_RANGE_G, ACC_RANGE_G), np.clip(gyro, -GYRO_RANGE_DPS, GYRO_RANGE_DPS)


def _marcas_tiempo(rng, n, inicio):
    """Marcas de tiempo con el jitter típico de una tablet (19-20 ms, a veces 38-39) y pausas BLE ocasionales."""
    dt = rng.choice([19, 20, 38, 39], n, p=[0.73, 0.23, 0.02, 0.02])
    for pos in rng.integers(1, n, rng.integers(0, 3)):
        dt[pos] += rng.integers(100, 500)
    return inicio + np.concatenate([[0], np.cumsum(dt[1:])])


def _muestras(device, acc, gyro, timestamps, secuencias):
    return [
        {
            "accelerometer": {k: round(float(v), 4) for k, v in zip("xyz", a)},
            "deviceId": device,
            "gyroscope": {k: round(float(v), 3) for k, v in zip("xyz", g)},
            "sequenceNumber": int(s),
            "timestamp": int(ts),
        }
        for a, g, ts, s in zip(acc, gyro, timestamps, secuencias)
    ]


def _aplicar_falla(rng, clase, lado, acc, gyro, idx, es_lado_falla):
    """Devuelve (acc, gyro, idx) del sensor tras inyectar la falla correspondiente."""
    n = len(idx)
    if clase == "PERDIDA_PAQUETES":
        for _ in range(rng.integers(2, 9)):
            inicio = rng.integers(0, n - 60)
            largo = rng.integers(15, 90)
            idx = idx[(idx < inicio) | (idx >= inicio + largo)]
    elif clase == "SENSOR_DESCONECTADO" and es_lado_falla:
        corte = int(n * rng.uniform(0.1, 0.9))
        idx = idx[idx < corte]
    elif clase == "SENSOR_SIN_MOVIMIENTO" and es_lado_falla:
        quieto = np.array([0.97, rng.uniform(-0.3, 0.3), rng.uniform(-0.25, 0)])
        acc = quieto + rng.normal(0, 0.01, acc.shape)
        gyro = rng.normal(0, rng.uniform(0.8, 4.0), gyro.shape)
    elif clase == "SATURACION":
        escala = rng.uniform(2.2, 5)
        acc = np.clip((acc - acc.mean(0)) * escala + acc.mean(0), -ACC_RANGE_G, ACC_RANGE_G)
        gyro = np.clip(gyro * escala * 2, -GYRO_RANGE_DPS, GYRO_RANGE_DPS)
    return acc, gyro, idx


def simular_registro(rng, clase="OK", duracion_s=None):
    duracion_s = duracion_s or rng.uniform(20, 40)
    n = int(duracion_s * 1000 / DT_MS)
    frecuencia = rng.uniform(1.2, 3.0)
    intensidad = rng.uniform(0.15, 2.2)
    fase = rng.uniform(0, 2 * np.pi)
    lado_falla = rng.choice(DEVICES)
    inicio_ts = int(rng.integers(100, 400))
    imu = []
    for i, lado in enumerate(DEVICES):
        # Los pies golpean alternados: el segundo va en contrafase.
        fase_lado = fase + i * (np.pi + rng.normal(0, 0.5))
        acc, gyro = _senal_tobillo(rng, n, intensidad * rng.uniform(0.7, 1.3), frecuencia, fase_lado)
        tiempos = _marcas_tiempo(rng, n, inicio_ts)
        # Pérdida natural de algunos paquetes BLE, también presente en registros válidos.
        idx = np.sort(rng.choice(n, n - rng.integers(0, max(2, n // 80)), replace=False))
        acc, gyro, idx = _aplicar_falla(rng, clase, lado, acc, gyro, idx, lado == lado_falla)
        ts = tiempos[idx]
        imu += _muestras(lado, acc[idx], gyro[idx], ts, idx)
    if clase == "DUPLICADO":
        copia = int(len(imu) * rng.uniform(0.05, 0.9))
        imu = imu + imu[:copia]
    else:
        imu.sort(key=lambda m: (m["timestamp"], m["deviceId"]))
    return {"testType": "TAPPING_TOE", "timestamp": 0, "endTimestamp": n * DT_MS, "imuData": imu}


def simular_dataset(n_por_clase, semilla=7):
    rng = np.random.default_rng(semilla)
    return [(simular_registro(rng, clase), clase) for clase in CLASES for _ in range(n_por_clase)]
