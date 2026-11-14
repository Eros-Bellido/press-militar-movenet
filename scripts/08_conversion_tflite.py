"""
08_conversion_tflite.py
Prepara los modelos para el smartphone y verifica, en la laptop, que la conversión no altere las predicciones.

1. Elige la semilla de la LSTM con mayor F1 medio en VALIDACIÓN.
2. Reescribe esa LSTM como modelo "paso a paso": recibe un fotograma y el estado (h, c) de cada capa,
   y devuelve la probabilidad y el nuevo estado. Es causal y es como funcionaría en una app.
3. Comprueba que el modelo paso a paso da las mismas probabilidades que el modelo Keras original.
4. Lo convierte a TensorFlow Lite (FP32, solo operaciones nativas) y recalcula el F1 de PRUEBA con el
   intérprete de TFLite, para compararlo con el F1 del modelo original.
5. Copia MoveNet Thunder FP16 junto al modelo convertido.

Entradas : outputs/modelos/lstm_s*.keras, outputs/predicciones_lstm.csv, outputs/lstm_mejores_parametros.json,
           data/features/secuencias.npz, models/movenet_thunder_fp16.tflite
Salidas  : outputs/movil/lstm_paso.tflite, outputs/movil/movenet_thunder_fp16.tflite
           outputs/conversion_resumen.txt
Uso      : python scripts/08_conversion_tflite.py
"""
import os, json, math, shutil, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, tensorflow as tf, keras
from sklearn.metrics import f1_score

OUT, MOVIL = "outputs", os.path.join("outputs", "movil")
os.makedirs(MOVIL, exist_ok=True)
FRACCIONES = [25, 50, 75, 100]

# ---------- 1. semilla elegida por validación ----------
P = pd.read_csv(os.path.join(OUT, "predicciones_lstm.csv"))
P["key"] = P["key"].astype(str)
def f1_medio(d):
    return np.mean([f1_score(g.y, g.prob >= g.umbral.iloc[0], zero_division=0) for _, g in d.groupby("fraccion")])
f1_val = P[P.split == "val"].groupby("semilla").apply(f1_medio, include_groups=False)
SEM = int(f1_val.idxmax())
UMBRAL = float(P[P.semilla == SEM].umbral.iloc[0])
m = keras.models.load_model(os.path.join(OUT, "modelos", f"lstm_s{SEM}.keras"))
lstms = [l for l in m.layers if isinstance(l, keras.layers.LSTM)]
densa = [l for l in m.layers if isinstance(l, keras.layers.Dense)][-1]
U = lstms[0].units
NF = m.input_shape[-1]

# ---------- 2. modelo paso a paso (pesos copiados del modelo entrenado) ----------
W = [[tf.constant(w) for w in l.get_weights()] for l in lstms]      # kernel, recurrent_kernel, bias
Wd = [tf.constant(w) for w in densa.get_weights()]

NE = 2 * len(W)                                                      # h y c por capa

class LSTMPaso(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, NF], tf.float32, name="x"),
                                  tf.TensorSpec([NE, U], tf.float32, name="estado")])
    def paso(self, x, estado):
        z, nuevo = x, []
        for k, (ker, rec, b) in enumerate(W):
            h, c = estado[2 * k:2 * k + 1], estado[2 * k + 1:2 * k + 2]
            g = tf.matmul(z, ker) + tf.matmul(h, rec) + b
            i, f, cc, o = tf.split(g, 4, axis=1)                         # orden de compuertas de Keras
            c = tf.sigmoid(f) * c + tf.sigmoid(i) * tf.tanh(cc)
            h = tf.sigmoid(o) * tf.tanh(c)
            nuevo += [h, c]; z = h
        p = tf.sigmoid(tf.matmul(z, Wd[0]) + Wd[1])
        return {"prob": p, "estado_nuevo": tf.concat(nuevo, axis=0)}

mod = LSTMPaso()
conv = tf.lite.TFLiteConverter.from_concrete_functions([mod.paso.get_concrete_function()], mod)
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tfl_bytes = conv.convert()
ruta_lstm = os.path.join(MOVIL, "lstm_paso.tflite")
open(ruta_lstm, "wb").write(tfl_bytes)

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter
it = Interpreter(model_path=ruta_lstm); it.allocate_tensors()
correr = it.get_signature_runner()
ent = it.get_input_details()

# ---------- 3 y 4. verificación sobre el conjunto de prueba ----------
d = np.load(os.path.join("data", "features", "secuencias.npz"), allow_pickle=True)
X, off, lon, KEYS, SPLIT, Y = d["X"], d["offsets"], d["longitudes"], d["keys"].astype(str), d["split"].astype(str), d["y"]
ids = np.where(SPLIT == "test")[0]

def correr_tflite(seq):
    est = np.zeros((NE, U), np.float32)
    probs = np.empty(len(seq), np.float32)
    for t in range(len(seq)):
        r = correr(x=seq[t:t + 1].astype(np.float32), estado=est)
        probs[t], est = r["prob"][0, 0], r["estado_nuevo"]
    return probs

dif_max, filas = 0.0, []
for k in ids:
    seq = X[off[k]: off[k] + lon[k]].astype(np.float32)
    ref = m.predict(seq[None], verbose=0)[0, :, 0]
    tfl = correr_tflite(seq)
    dif_max = max(dif_max, float(np.abs(ref - tfl).max()))
    for f in FRACCIONES:
        t = math.ceil(f / 100 * lon[k]) - 1
        filas.append(dict(key=KEYS[k], fraccion=f, y=int(Y[k]), prob_keras=float(ref[t]), prob_tflite=float(tfl[t])))
V = pd.DataFrame(filas)
f1k = {f: f1_score(g.y, g.prob_keras >= UMBRAL, zero_division=0) for f, g in V.groupby("fraccion")}
f1t = {f: f1_score(g.y, g.prob_tflite >= UMBRAL, zero_division=0) for f, g in V.groupby("fraccion")}
pred_cambian = int(((V.prob_keras >= UMBRAL) != (V.prob_tflite >= UMBRAL)).sum())

# ---------- 5. MoveNet ----------
ruta_mn = os.path.join(MOVIL, "movenet_thunder_fp16.tflite")
shutil.copy(os.path.join("models", "movenet_thunder_fp16.tflite"), ruta_mn)

kb = lambda p: os.path.getsize(p) / 1024
L = [f"TensorFlow {tf.__version__}",
     "F1 medio de validación por semilla (LSTM): " + ", ".join(f"s{s} {v:.4f}" for s, v in f1_val.items()),
     f"Semilla elegida: {SEM} | umbral {UMBRAL} | {len(lstms)} capas LSTM de {U} unidades | {NF} entradas",
     f"Modelo paso a paso -> {ruta_lstm} | {kb(ruta_lstm):.1f} KB | solo operaciones nativas de TFLite (FP32)",
     f"Entradas TFLite: " + "; ".join(f"{d['name']} {list(d['shape'])}" for d in ent),
     f"Diferencia máxima de probabilidad Keras vs TFLite (prueba, todos los fotogramas): {dif_max:.2e}",
     f"Predicciones que cambian de clase con el umbral: {pred_cambian} de {len(V)}",
     "F1 de PRUEBA, semilla elegida (Keras -> TFLite):"]
L += [f"  {f} %: {f1k[f]:.4f} -> {f1t[f]:.4f}" for f in FRACCIONES]
L.append(f"MoveNet Thunder FP16 -> {ruta_mn} | {kb(ruta_mn)/1024:.2f} MB")
open(os.path.join(OUT, "conversion_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
