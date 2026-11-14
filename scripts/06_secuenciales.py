"""
06_secuenciales.py
Modelos secuenciales causales (LSTM, TCN, Transformer) sobre las secuencias por fotograma.
Cada modelo emite una probabilidad por fotograma usando solo fotogramas <= t. La predicción para
la fracción p es la salida en el fotograma ceil(p*T). La pérdida se calcula en esos 4 puntos
(25, 50, 75 y 100 %), igual que la supervisión que recibió LightGBM.

Uso (una arquitectura por corrida):
    python scripts/06_secuenciales.py lstm
    python scripts/06_secuenciales.py tcn
    python scripts/06_secuenciales.py transformer
Si se interrumpe, al volver a correr el mismo comando continúa donde quedó
(Optuna se guarda en outputs/optuna_<arq>.db y cada semilla final en su propio archivo).

Entradas : data/features/secuencias.npz
Salidas  : outputs/optuna_<arq>.db, outputs/<arq>_mejores_parametros.json
           outputs/modelos/<arq>_s<semilla>.keras
           outputs/predicciones_<arq>.csv   (validación y prueba)
           outputs/<arq>_resumen.txt        (solo métricas de VALIDACIÓN; la prueba se evalúa en 07)
"""
import os, sys, json, math, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, optuna
import tensorflow as tf, keras
from keras import layers
from sklearn.metrics import f1_score

ARQ = sys.argv[1].lower() if len(sys.argv) > 1 else ""
assert ARQ in ("lstm", "tcn", "transformer"), "Uso: python scripts/06_secuenciales.py lstm|tcn|transformer"
N_INTENTOS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEMILLAS, SEMILLA_OPTUNA = [0, 1, 2, 3, 4], 42
FRACCIONES = [25, 50, 75, 100]
LOTE, EPOCAS, PACIENCIA, MAX_POS = 32, 60, 8, 1024
UMBRALES = np.round(np.arange(0.05, 0.951, 0.01), 2)
OUT = "outputs"; os.makedirs(os.path.join(OUT, "modelos"), exist_ok=True)

# ---------- datos ----------
d = np.load(os.path.join("data", "features", "secuencias.npz"), allow_pickle=True)
X, off, lon = d["X"], d["offsets"], d["longitudes"]
SEQ = [X[o:o + n] for o, n in zip(off, lon)]
KEYS, SPLIT, Y = d["keys"], d["split"], d["y"].astype(np.float32)
NF = X.shape[1]
idx = {s: np.where(SPLIT == s)[0] for s in ("train", "val", "test")}
assert max(lon) <= MAX_POS

def puntos(n):
    return [math.ceil(p / 100 * n) - 1 for p in FRACCIONES]

class Lotes(keras.utils.PyDataset):
    """Lotes de longitud parecida (menos relleno). Relleno al final: no afecta a modelos causales."""
    def __init__(self, ids, peso_pos, mezclar, semilla=0):
        super().__init__()
        self.ids, self.pp, self.mezclar = np.array(ids), peso_pos, mezclar
        self.rng = np.random.default_rng(semilla)
        orden = self.ids[np.argsort(lon[self.ids], kind="stable")]
        self.lotes = [orden[i:i + LOTE] for i in range(0, len(orden), LOTE)]
        self.on_epoch_end()
    def __len__(self):
        return len(self.lotes)
    def on_epoch_end(self):
        self.orden = self.rng.permutation(len(self.lotes)) if self.mezclar else np.arange(len(self.lotes))
    def __getitem__(self, i):
        b = self.lotes[self.orden[i]]
        T = int(lon[b].max())
        x = np.zeros((len(b), T, NF), np.float32)
        y = np.zeros((len(b), T, 1), np.float32)
        w = np.zeros((len(b), T), np.float32)
        for j, k in enumerate(b):
            x[j, :lon[k]] = SEQ[k]
            pts = puntos(lon[k])
            y[j, pts, 0] = Y[k]
            w[j, pts] = self.pp if Y[k] == 1 else 1.0
        return x, y, w

# ---------- arquitecturas ----------
class PosEmb(layers.Layer):
    def __init__(self, d_model, **kw):
        super().__init__(**kw); self.d_model = d_model
        self.emb = layers.Embedding(MAX_POS, d_model)
    def call(self, x):
        return x + self.emb(tf.range(tf.shape(x)[1]))[None]
    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}

def construir(h):
    ent = keras.Input((None, NF))
    z = ent
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
        z = layers.Dense(h["d_model"])(z)
        z = PosEmb(h["d_model"])(z)
        for _ in range(h["capas"]):
            a = layers.MultiHeadAttention(h["cabezas"], h["d_model"] // h["cabezas"], dropout=h["dropout"])(
                z, z, use_causal_mask=True)
            z = layers.LayerNormalization()(z + a)
            f = layers.Dense(2 * h["d_model"], activation="relu")(z)
            f = layers.Dropout(h["dropout"])(layers.Dense(h["d_model"])(f))
            z = layers.LayerNormalization()(z + f)
    sal = layers.Dense(1, activation="sigmoid")(z)
    m = keras.Model(ent, sal)
    m.compile(keras.optimizers.Adam(h["lr"]), loss="binary_crossentropy")
    return m

def espacio(t):
    h = dict(lr=t.suggest_float("lr", 1e-4, 3e-3, log=True),
             dropout=t.suggest_float("dropout", 0.0, 0.4),
             peso_pos=t.suggest_float("peso_pos", 1.0, 4.0))
    if ARQ == "lstm":
        h.update(unidades=t.suggest_categorical("unidades", [32, 64, 128]), capas=t.suggest_int("capas", 1, 2))
    elif ARQ == "tcn":
        h.update(filtros=t.suggest_categorical("filtros", [32, 64]), niveles=t.suggest_int("niveles", 3, 5))
    else:
        h.update(d_model=t.suggest_categorical("d_model", [32, 64]), cabezas=t.suggest_categorical("cabezas", [2, 4]),
                 capas=t.suggest_int("capas", 1, 2))
    return h

# ---------- entrenamiento y evaluación ----------
def predecir(m, ids):
    ds = Lotes(ids, 1.0, False)
    prob, fr, ks, ys = [], [], [], []
    for i in range(len(ds)):
        b = ds.lotes[i]
        x, _, _ = ds[i]
        p = m.predict_on_batch(x)[..., 0]
        for j, k in enumerate(b):
            for f, t in zip(FRACCIONES, puntos(lon[k])):
                prob.append(p[j, t]); fr.append(f); ks.append(k); ys.append(Y[k])
    return np.array(prob), np.array(fr), np.array(ks), np.array(ys)

def mejor_umbral(y, prob):
    return float(UMBRALES[int(np.argmax([f1_score(y, prob >= u, zero_division=0) for u in UMBRALES]))])

def f1_fr(y, prob, fr, u):
    return {f: f1_score(y[fr == f], prob[fr == f] >= u, zero_division=0) for f in FRACCIONES}

class Poda(keras.callbacks.Callback):
    def __init__(self, trial): super().__init__(); self.t = trial
    def on_epoch_end(self, ep, logs=None):
        self.t.report(-logs["val_loss"], ep)
        if self.t.should_prune():
            raise optuna.TrialPruned()

def entrenar(h, semilla, trial=None):
    keras.utils.set_random_seed(semilla)
    m = construir(h)
    cb = [keras.callbacks.EarlyStopping("val_loss", patience=PACIENCIA, restore_best_weights=True)]
    if trial is not None:
        cb.append(Poda(trial))
    hist = m.fit(Lotes(idx["train"], h["peso_pos"], True, semilla),
                 validation_data=Lotes(idx["val"], h["peso_pos"], False),
                 epochs=EPOCAS, callbacks=cb, verbose=0)
    return m, int(np.argmin(hist.history["val_loss"])) + 1

def objetivo(trial):
    h = espacio(trial)
    m, _ = entrenar(h, 0, trial)
    p, fr, _, y = predecir(m, idx["val"])
    keras.backend.clear_session()
    return float(np.mean(list(f1_fr(y, p, fr, mejor_umbral(y, p)).values())))

t0 = time.perf_counter()
optuna.logging.set_verbosity(optuna.logging.WARNING)
estudio = optuna.create_study(study_name=ARQ, storage=f"sqlite:///{OUT}/optuna_{ARQ}.db", load_if_exists=True,
                              direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
hechos = len([t for t in estudio.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                      optuna.trial.TrialState.PRUNED)])
if hechos < N_INTENTOS:
    print(f"[{ARQ}] Optuna: {hechos} intentos previos, faltan {N_INTENTOS - hechos}")
    estudio.optimize(objetivo, n_trials=N_INTENTOS - hechos, show_progress_bar=True)
mejores = estudio.best_params
h = {**mejores}
json.dump(mejores, open(os.path.join(OUT, f"{ARQ}_mejores_parametros.json"), "w"), indent=2)

filas, f1_val, epocas, umbrales, n_par = [], [], [], [], None
for s in SEMILLAS:
    ruta_m = os.path.join(OUT, "modelos", f"{ARQ}_s{s}.keras")
    ruta_p = os.path.join(OUT, "modelos", f"{ARQ}_s{s}_pred.csv")
    if os.path.exists(ruta_m) and os.path.exists(ruta_p):
        df_s = pd.read_csv(ruta_p); m = keras.models.load_model(ruta_m, custom_objects={"PosEmb": PosEmb})
        ep = int(df_s.epocas.iloc[0])
    else:
        print(f"[{ARQ}] semilla final {s}")
        m, ep = entrenar(h, s)
        m.save(ruta_m)
        partes = []
        for sp in ("val", "test"):
            p, fr, ks, y = predecir(m, idx[sp])
            partes.append(pd.DataFrame(dict(modelo=ARQ.upper() if ARQ != "transformer" else "Transformer",
                                            semilla=s, split=sp, key=KEYS[ks], fraccion=fr, y=y.astype(int), prob=p)))
        df_s = pd.concat(partes); df_s["epocas"] = ep
        v = df_s[df_s.split == "val"]
        df_s["umbral"] = mejor_umbral(v.y.values, v.prob.values)
        df_s.to_csv(ruta_p, index=False)
    n_par = m.count_params()
    v = df_s[df_s.split == "val"]
    u = float(df_s.umbral.iloc[0])
    f1_val.append(f1_fr(v.y.values, v.prob.values, v.fraccion.values, u)); epocas.append(ep); umbrales.append(u)
    filas.append(df_s.drop(columns="epocas"))
    keras.backend.clear_session()
pd.concat(filas).to_csv(os.path.join(OUT, f"predicciones_{ARQ}.csv"), index=False)

estados = pd.Series([t.state.name for t in estudio.trials]).value_counts().to_dict()
F = pd.DataFrame(f1_val)
L = [f"{ARQ} | TensorFlow {tf.__version__} | Keras {keras.__version__} | Optuna {optuna.__version__}",
     f"Datos: {len(idx['train'])} train, {len(idx['val'])} val, {len(idx['test'])} test | {NF} entradas por fotograma",
     f"Optuna: {N_INTENTOS} intentos, TPE semilla {SEMILLA_OPTUNA}, MedianPruner | estados {estados} | "
     f"mejor objetivo (F1 medio val) {estudio.best_value:.4f}",
     "Mejores hiperparámetros: " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in mejores.items()}),
     f"Fijos: lote {LOTE}, máx. {EPOCAS} épocas, parada temprana paciencia {PACIENCIA} (val_loss), Adam, "
     f"pérdida BCE ponderada en 4 puntos | parámetros del modelo: {n_par}",
     f"Épocas (mejor) por semilla: {epocas} | umbral por semilla: {umbrales}",
     "F1 clase incorrecta en VALIDACIÓN (media ± DE, 5 semillas):"]
L += [f"  {p} %: {F[p].mean():.4f} ± {F[p].std(ddof=1):.4f}" for p in FRACCIONES]
L.append(f"Tiempo de esta corrida: {(time.perf_counter() - t0) / 60:.1f} min")
open(os.path.join(OUT, f"{ARQ}_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
