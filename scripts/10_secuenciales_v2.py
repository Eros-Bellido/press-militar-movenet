"""
10_secuenciales_v2.py  — Segunda ronda de experimentos (modelos secuenciales mejorados)
Cambios respecto al script 06, con los MISMOS hiperparámetros ya seleccionados (no se repite la búsqueda):
  1. Supervisión por fotograma: además de la etiqueta del video, el modelo aprende en qué fotogramas
     está presente el error, según los intervalos de data/Labels/error_elbows.json (segunda salida).
  2. Aumento de datos por espejo: en entrenamiento, cada video se voltea horizontalmente con
     probabilidad 0.5 (se intercambian izquierda y derecha y se invierte la coordenada x).
  3. Ensamble: promedio de las probabilidades de las 5 semillas.
Todo se decide con VALIDACIÓN. Este script NO muestra métricas de prueba (eso lo hace el script 11).

Uso:  python scripts/10_secuenciales_v2.py lstm      (luego tcn y transformer)
Entradas : data/features/secuencias.npz, data/Labels/error_elbows.json, outputs/estandarizacion.csv,
           outputs/<arq>_mejores_parametros.json, outputs/predicciones_<arq>.csv (ronda 1, para comparar)
Salidas  : outputs/modelos_v2/<arq>_s<semilla>.keras, outputs/predicciones_<arq>_v2.csv,
           outputs/<arq>_v2_resumen.txt
"""
import os, sys, json, math, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, tensorflow as tf, keras
from keras import layers
from sklearn.metrics import f1_score

ARQ = sys.argv[1].lower() if len(sys.argv) > 1 else ""
assert ARQ in ("lstm", "tcn", "transformer"), "Uso: python scripts/10_secuenciales_v2.py lstm|tcn|transformer"
SEMILLAS, FRACCIONES, FPS = [0, 1, 2, 3, 4], [25, 50, 75, 100], 30
LOTE, EPOCAS, PACIENCIA, MAX_POS = 32, 60, 8, 1024
PESO_FOTOGRAMA, P_ESPEJO = 1.0, 0.5
UMBRALES = np.round(np.arange(0.05, 0.951, 0.01), 2)
OUT = "outputs"; os.makedirs(os.path.join(OUT, "modelos_v2"), exist_ok=True)
NOMBRE = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer"}[ARQ]

# ---------- datos ----------
d = np.load(os.path.join("data", "features", "secuencias.npz"), allow_pickle=True)
X, off, lon = d["X"].astype(np.float32), d["offsets"], d["longitudes"]
KEYS, SPLIT, Y = d["keys"].astype(str), d["split"].astype(str), d["y"].astype(np.float32)
COLS = list(d["columnas"].astype(str)); NF = X.shape[1]
SEQ = [X[o:o + n] for o, n in zip(off, lon)]
idx = {s: np.where(SPLIT == s)[0] for s in ("train", "val", "test")}

# Etiqueta por fotograma a partir de los intervalos (en segundos)
intervalos = json.load(open(os.path.join("data", "Labels", "error_elbows.json")))
def etiqueta_fotogramas(key, n):
    iv = intervalos.get(key, intervalos.get(key + ".mp4", intervalos.get(os.path.basename(key), [])))
    z = np.zeros(n, np.float32)
    t = np.arange(n) / FPS
    for a, b in iv:
        z[(t >= a) & (t <= b)] = 1.0
    return z
ZF = [etiqueta_fotogramas(k, n) for k, n in zip(KEYS, lon)]
pos = [i for i in range(len(Y)) if Y[i] == 1]
sin_marca = int(sum(ZF[i].sum() == 0 for i in pos))

# Espejo: pares izquierda/derecha y columnas x, en el espacio sin estandarizar
est = pd.read_csv(os.path.join(OUT, "estandarizacion.csv")).set_index("caracteristica")
NC = len(est)                                     # características continuas (20)
mu = est.loc[COLS[:NC], "media_train"].values.astype(np.float32)
de = est.loc[COLS[:NC], "de_train"].values.astype(np.float32)
perm = np.arange(NF)
for i, c in enumerate(COLS):
    otro = c.replace("_izq", "_TMP").replace("_der", "_izq").replace("_TMP", "_der")
    perm[i] = COLS.index(otro)
SIGNO = np.array([-1.0 if c.startswith("x_") else 1.0 for c in COLS[:NC]], np.float32)

def espejo(s):
    """Voltea horizontalmente una secuencia estandarizada (T, NF)."""
    falt = (s[:, :NC] == 0.0)                     # valores vacíos (se fijaron en 0)
    crudo = s[:, :NC] * de + mu
    cont = (crudo[:, perm[:NC]] * SIGNO - mu) / de
    cont[falt[:, perm[:NC]]] = 0.0
    return np.concatenate([cont, s[:, NC:][:, perm[NC:] - NC]], axis=1).astype(np.float32)

def puntos(n):
    return [math.ceil(p / 100 * n) - 1 for p in FRACCIONES]

class Lotes(keras.utils.PyDataset):
    def __init__(self, ids, peso_pos, entrenamiento, semilla=0):
        super().__init__()
        self.ids, self.pp, self.ent = np.array(ids), peso_pos, entrenamiento
        self.rng = np.random.default_rng(semilla)
        orden = self.ids[np.argsort(lon[self.ids], kind="stable")]
        self.lotes = [orden[i:i + LOTE] for i in range(0, len(orden), LOTE)]
        self.on_epoch_end()
    def __len__(self): return len(self.lotes)
    def on_epoch_end(self):
        self.orden = self.rng.permutation(len(self.lotes)) if self.ent else np.arange(len(self.lotes))
    def __getitem__(self, i):
        b = self.lotes[self.orden[i]]; T = int(lon[b].max())
        x = np.zeros((len(b), T, NF), np.float32)
        yv = np.zeros((len(b), T, 1), np.float32); wv = np.zeros((len(b), T), np.float32)
        yf = np.zeros((len(b), T, 1), np.float32); wf = np.zeros((len(b), T), np.float32)
        for j, k in enumerate(b):
            n = lon[k]; s = SEQ[k]
            if self.ent and self.rng.random() < P_ESPEJO:
                s = espejo(s)
            x[j, :n] = s
            pts = puntos(n)
            yv[j, pts, 0] = Y[k]; wv[j, pts] = self.pp if Y[k] == 1 else 1.0
            yf[j, :n, 0] = ZF[k]; wf[j, :n] = 4.0 / n      # cada video pesa lo mismo que sus 4 puntos
        return x, {"video": yv, "fotograma": yf}, {"video": wv, "fotograma": wf}

# ---------- arquitecturas (idénticas al script 06, con una segunda salida) ----------
class PosEmb(layers.Layer):
    def __init__(self, d_model, **kw):
        super().__init__(**kw); self.d_model = d_model; self.emb = layers.Embedding(MAX_POS, d_model)
    def call(self, x): return x + self.emb(tf.range(tf.shape(x)[1]))[None]
    def get_config(self): return {**super().get_config(), "d_model": self.d_model}

def construir(h):
    ent = keras.Input((None, NF)); z = ent
    if ARQ == "lstm":
        for _ in range(h["capas"]):
            z = layers.LSTM(h["unidades"], return_sequences=True, dropout=h["dropout"])(z)
    elif ARQ == "tcn":
        z = layers.Conv1D(h["filtros"], 1)(z)
        for i in range(h["niveles"]):
            r = z
            for _ in range(2):
                z = layers.Conv1D(h["filtros"], 3, padding="causal", dilation_rate=2 ** i, activation="relu")(z)
                z = layers.Dropout(h["dropout"])(z)
            z = layers.Add()([r, z])
    else:
        z = PosEmb(h["d_model"])(layers.Dense(h["d_model"])(z))
        for _ in range(h["capas"]):
            a = layers.MultiHeadAttention(h["cabezas"], h["d_model"] // h["cabezas"], dropout=h["dropout"])(
                z, z, use_causal_mask=True)
            z = layers.LayerNormalization()(z + a)
            f = layers.Dropout(h["dropout"])(layers.Dense(h["d_model"])(layers.Dense(2 * h["d_model"], activation="relu")(z)))
            z = layers.LayerNormalization()(z + f)
    m = keras.Model(ent, {"video": layers.Dense(1, activation="sigmoid", name="video")(z),
                          "fotograma": layers.Dense(1, activation="sigmoid", name="fotograma")(z)})
    m.compile(keras.optimizers.Adam(h["lr"]),
              loss={"video": "binary_crossentropy", "fotograma": "binary_crossentropy"},
              loss_weights={"video": 1.0, "fotograma": PESO_FOTOGRAMA})
    return m

def predecir(m, ids):
    ds = Lotes(ids, 1.0, False); prob, fr, ks = [], [], []
    for i in range(len(ds)):
        b = ds.lotes[i]; x = ds[i][0]
        p = m.predict_on_batch(x)["video"][..., 0]
        for j, k in enumerate(b):
            for f, t in zip(FRACCIONES, puntos(lon[k])):
                prob.append(p[j, t]); fr.append(f); ks.append(k)
    ks = np.array(ks)
    return pd.DataFrame(dict(key=KEYS[ks], fraccion=fr, y=Y[ks].astype(int), prob=np.array(prob, np.float64)))

def mejor_umbral(y, prob):
    return float(UMBRALES[int(np.argmax([f1_score(y, prob >= u, zero_division=0) for u in UMBRALES]))])
def f1_fr(d, u):
    return {f: f1_score(g.y, g.prob >= u, zero_division=0) for f, g in d.groupby("fraccion")}

h = json.load(open(os.path.join(OUT, f"{ARQ}_mejores_parametros.json")))
t0 = time.perf_counter(); partes, f1s, epocas, umbrales = [], [], [], []
for s in SEMILLAS:
    ruta_m = os.path.join(OUT, "modelos_v2", f"{ARQ}_s{s}.keras")
    ruta_p = os.path.join(OUT, "modelos_v2", f"{ARQ}_s{s}_pred.csv")
    if os.path.exists(ruta_p):
        df_s = pd.read_csv(ruta_p)
    else:
        print(f"[{ARQ} v2] semilla {s}")
        keras.utils.set_random_seed(s)
        m = construir(h)
        hist = m.fit(Lotes(idx["train"], h["peso_pos"], True, s), validation_data=Lotes(idx["val"], h["peso_pos"], False),
                     epochs=EPOCAS, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping("val_video_loss", mode="min", patience=PACIENCIA,
                                                              restore_best_weights=True)])
        m.save(ruta_m)
        df_s = pd.concat([predecir(m, idx[sp]).assign(split=sp) for sp in ("val", "test")])
        v = df_s[df_s.split == "val"]
        df_s["umbral"] = mejor_umbral(v.y.values, v.prob.values)
        df_s["epocas"] = int(np.argmin(hist.history["val_video_loss"])) + 1
        df_s.insert(0, "semilla", s); df_s.insert(0, "modelo", NOMBRE + " v2")
        df_s.to_csv(ruta_p, index=False); keras.backend.clear_session()
    v = df_s[df_s.split == "val"]; u = float(df_s.umbral.iloc[0])
    f1s.append(f1_fr(v, u)); epocas.append(int(df_s.epocas.iloc[0])); umbrales.append(u)
    partes.append(df_s.drop(columns="epocas"))
P = pd.concat(partes)

# Ensamble: promedio de probabilidades de las 5 semillas; umbral elegido en validación
E = P.groupby(["split", "key", "fraccion", "y"], as_index=False).prob.mean()
ue = mejor_umbral(E[E.split == "val"].y.values, E[E.split == "val"].prob.values)
E["umbral"] = ue; E.insert(0, "semilla", -1); E.insert(0, "modelo", NOMBRE + " v2 ensamble")
f1e = f1_fr(E[E.split == "val"], ue)
pd.concat([P, E]).to_csv(os.path.join(OUT, f"predicciones_{ARQ}_v2.csv"), index=False)

# Comparación con la ronda 1 (validación)
r1 = pd.read_csv(os.path.join(OUT, f"predicciones_{ARQ}.csv")); r1 = r1[r1.split == "val"]
F1r1 = pd.DataFrame([f1_fr(g, float(g.umbral.iloc[0])) for _, g in r1.groupby("semilla")])
F = pd.DataFrame(f1s)
L = [f"{NOMBRE} v2 | TensorFlow {tf.__version__} | Keras {keras.__version__}",
     f"Cambios: supervisión por fotograma (peso {PESO_FOTOGRAMA}), espejo (p={P_ESPEJO}), ensamble de {len(SEMILLAS)} semillas",
     "Hiperparámetros (los mismos de la ronda 1): " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}),
     f"Videos positivos sin ningún fotograma marcado por los intervalos: {sin_marca} de {len(pos)}",
     f"Fotogramas con error (positivos): {np.mean([ZF[i].mean() for i in pos]):.1%} en promedio",
     f"Épocas por semilla: {epocas} | umbral por semilla: {umbrales} | umbral del ensamble: {ue}",
     "F1 clase incorrecta en VALIDACIÓN:",
     "  fracción | ronda 1 (media ± DE) | v2 (media ± DE) | v2 ensamble"]
for p in FRACCIONES:
    L.append(f"  {p:>3} %    | {F1r1[p].mean():.4f} ± {F1r1[p].std(ddof=1):.4f}   | {F[p].mean():.4f} ± {F[p].std(ddof=1):.4f} | {f1e[p]:.4f}")
L.append(f"  PROMEDIO | {F1r1.values.mean():.4f}            | {F.values.mean():.4f}          | {np.mean(list(f1e.values())):.4f}")
L.append(f"Tiempo de esta corrida: {(time.perf_counter() - t0) / 60:.1f} min")
open(os.path.join(OUT, f"{ARQ}_v2_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
