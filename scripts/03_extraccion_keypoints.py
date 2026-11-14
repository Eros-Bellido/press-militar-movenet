"""
03_extraccion_keypoints.py
Extrae los 17 keypoints de MoveNet Thunder (TFLite FP16) en cada fotograma de los
2 260 videos etiquetados.

Entradas : outputs/videos_etiquetados.csv, data/videos/<key>.mp4,
           models/movenet_thunder_fp16.tflite
Salidas  : data/keypoints/<key>.npy   arreglo float32 (T, 17, 3) = (x, y, confianza);
                                      x e y normalizados al ancho y alto del video original
           outputs/keypoints_por_video.csv
           outputs/keypoints_resumen.txt    (cifras para el artículo)
Uso      : python scripts/03_extraccion_keypoints.py
           Si se interrumpe, al volver a correrlo salta los videos ya procesados.
"""
import os, time, hashlib, platform
import numpy as np, pandas as pd, cv2
import tensorflow as tf

try:                                    # LiteRT (reemplazo oficial de tf.lite)
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "ai_edge_litert"
except ImportError:
    Interpreter = tf.lite.Interpreter
    RUNTIME = "tf.lite"

MODELO  = os.path.join("models", "movenet_thunder_fp16.tflite")
VIDEOS  = os.path.join("data", "videos")
SALIDA  = os.path.join("data", "keypoints")
OUT     = "outputs"
TAM     = 256          # entrada de MoveNet Thunder
HILOS   = 4            # fijo, para que el tiempo sea comparable entre corridas
UMBRAL  = 0.2          # solo para reportar; el filtrado se hace en el script 04
SUP     = {5: "hombro_izq", 6: "hombro_der", 7: "codo_izq", 8: "codo_der",
           9: "muneca_izq", 10: "muneca_der", 11: "cadera_izq", 12: "cadera_der"}

os.makedirs(SALIDA, exist_ok=True)
sha = hashlib.sha256(open(MODELO, "rb").read()).hexdigest()
interp = Interpreter(model_path=MODELO, num_threads=HILOS)
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
assert tuple(inp["shape"]) == (1, TAM, TAM, 3), inp["shape"]

def keypoints(frame_bgr):
    """Escala con relación de aspecto conservada + relleno (letterbox), infiere y
    devuelve (17, 3) con x, y normalizados al fotograma original."""
    h, w = frame_bgr.shape[:2]
    s = TAM / max(h, w)
    nh, nw = round(h * s), round(w * s)
    top, left = (TAM - nh) // 2, (TAM - nw) // 2
    lienzo = np.zeros((TAM, TAM, 3), np.uint8)
    lienzo[top:top + nh, left:left + nw] = cv2.resize(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_LINEAR)
    interp.set_tensor(inp["index"], lienzo[None].astype(inp["dtype"]))
    interp.invoke()
    k = interp.get_tensor(out["index"])[0, 0]          # (17, 3) = (y, x, conf) en el lienzo
    x = (k[:, 1] * TAM - left) / nw
    y = (k[:, 0] * TAM - top) / nh
    return np.stack([x, y, k[:, 2]], axis=1).astype(np.float32)

df = pd.read_csv(os.path.join(OUT, "videos_etiquetados.csv"))
filas, t_total, n_inf = [], 0.0, 0
for i, r in enumerate(df.itertuples(), 1):
    destino = os.path.join(SALIDA, os.path.basename(r.key) + ".npy")
    if os.path.exists(destino):
        kp, dt = np.load(destino), np.nan
    else:
        cap, lista = cv2.VideoCapture(os.path.join(VIDEOS, r.key + ".mp4")), []
        t0 = time.perf_counter()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            lista.append(keypoints(fr))
        dt = time.perf_counter() - t0
        cap.release()
        kp = np.stack(lista) if lista else np.zeros((0, 17, 3), np.float32)
        np.save(destino, kp)
        t_total += dt; n_inf += len(kp)
    fila = {"key": r.key, "split": r.split, "y": r.y,
            "fotogramas_csv": r.fotogramas, "fotogramas_leidos": len(kp), "seg": dt}
    for j, nom in SUP.items():
        fila[f"conf_{nom}"] = float(kp[:, j, 2].mean()) if len(kp) else np.nan
        fila[f"bajo_umbral_{nom}"] = float((kp[:, j, 2] < UMBRAL).mean()) if len(kp) else np.nan
    filas.append(fila)
    if i % 100 == 0:
        print(f"{i}/{len(df)} videos")

res = pd.DataFrame(filas)
res.to_csv(os.path.join(OUT, "keypoints_por_video.csv"), index=False)

dif = res["fotogramas_leidos"] - res["fotogramas_csv"]
L = [f"Modelo: {MODELO} | SHA256 {sha}",
     f"Entrada: {tuple(inp['shape'])} {np.dtype(inp['dtype']).name} | runtime {RUNTIME} | "
     f"TF {tf.__version__} | OpenCV {cv2.__version__} | hilos {HILOS} | {platform.processor()}",
     f"Videos procesados: {len(res)} | sin fotogramas: {(res.fotogramas_leidos == 0).sum()}",
     f"Fotogramas procesados: {int(res.fotogramas_leidos.sum())}",
     f"Diferencia leídos - frames.csv: igual en {(dif == 0).sum()} videos | "
     f"min {dif.min()} | max {dif.max()}"]
for nom in SUP.values():
    L.append(f"{nom}: confianza media {res[f'conf_{nom}'].mean():.3f} | "
             f"fotogramas < {UMBRAL}: {res[f'bajo_umbral_{nom}'].mean():.1%}")
if n_inf:
    L.append(f"Tiempo de esta corrida: {t_total/60:.1f} min | {1000*t_total/n_inf:.1f} ms/fotograma "
             f"(incluye lectura y preprocesamiento; NO comparable con la medición en Android)")
open(os.path.join(OUT, "keypoints_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
