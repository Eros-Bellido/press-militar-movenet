"""
04_caracteristicas.py
Construye, a partir de los keypoints de MoveNet, las dos representaciones del estudio:
  (a) secuencias por fotograma (modelos secuenciales)  y
  (b) estadísticos agregados por video para cada fracción observada (modelo de referencia).
Todo el cálculo por fotograma es causal: el valor en t solo usa fotogramas <= t.

Entradas : outputs/videos_etiquetados.csv, data/keypoints/<key>.npy  (T, 17, 3) = (x, y, conf)
Salidas  : data/features/secuencias.npz    X (sum T, 26) estandarizado, offsets, longitudes, keys, split, y
           data/features/agregadas_p25.csv, _p50, _p75, _p100   (146 características + key, split, y)
           outputs/estandarizacion.csv     media y desviación de entrenamiento por característica
           outputs/caracteristicas_resumen.txt
Uso      : python scripts/04_caracteristicas.py
"""
import os, math
import numpy as np, pandas as pd

KP, FEAT, OUT = os.path.join("data", "keypoints"), os.path.join("data", "features"), "outputs"
UMBRAL, FPS = 0.2, 30
FRACCIONES = [0.25, 0.50, 0.75, 1.00]
I = dict(hi=5, hd=6, ci=7, cd=8, mi=9, md=10, ki=11, kd=12)   # hombro, codo, muñeca, cadera (izq/der)
BRAZO = ["hi", "hd", "ci", "cd", "mi", "md"]
CONT = (["ang_codo_izq", "ang_codo_der", "ang_hombro_izq", "ang_hombro_der"]
        + [f"{c}_{p}" for p in ["hombro_izq", "hombro_der", "codo_izq", "codo_der", "muneca_izq", "muneca_der"] for c in ("x", "y")]
        + ["vel_codo_izq", "vel_codo_der", "vel_hombro_izq", "vel_hombro_der"])
MASK = [f"valido_{p}" for p in ["hombro_izq", "hombro_der", "codo_izq", "codo_der", "muneca_izq", "muneca_der"]]
os.makedirs(FEAT, exist_ok=True)

def relleno_causal(p, ok):
    """Repite el último valor válido; antes del primero, NaN. p: (T, 2), ok: (T,)"""
    q = np.full_like(p, np.nan)
    ult = None
    for t in range(len(p)):
        if ok[t]:
            ult = p[t]
        if ult is not None:
            q[t] = ult
    return q

def angulo(a, b, c):
    """Ángulo en b (grados) entre los vectores b->a y b->c."""
    u, v = a - b, c - b
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def por_fotograma(kp, ancho, alto):
    """kp (T,17,3) -> (T, 20) continuas con NaN donde aún no hay dato, (T, 6) máscaras."""
    px = kp[:, :, :2] * np.array([ancho, alto], np.float32)          # a píxeles
    ok = kp[:, :, 2] >= UMBRAL
    P = {k: relleno_causal(px[:, j], ok[:, j]) for k, j in I.items()}
    hombros = (P["hi"] + P["hd"]) / 2
    torso = np.linalg.norm(hombros - (P["ki"] + P["kd"]) / 2, axis=1)
    torso[torso < 1e-6] = np.nan
    ang = [angulo(P["hi"], P["ci"], P["mi"]), angulo(P["hd"], P["cd"], P["md"]),
           angulo(P["ci"], P["hi"], P["ki"]), angulo(P["cd"], P["hd"], P["kd"])]
    coords = []
    for k in BRAZO:
        n = (P[k] - hombros) / torso[:, None]
        coords += [n[:, 0], n[:, 1]]
    vel = []
    for a in ang:
        d = np.diff(a, prepend=np.nan) * FPS                          # grados/s, causal
        d[0] = 0.0
        vel.append(d)
    F = np.stack(ang + coords + vel, axis=1).astype(np.float32)
    M = np.stack([ok[:, I[k]] for k in BRAZO], axis=1).astype(np.float32)
    return F, M

def agregar(F, M):
    """Estadísticos sobre los fotogramas observados (ignora NaN iniciales)."""
    with np.errstate(all="ignore"):
        est = [np.nanmean(F, 0), np.nanstd(F, 0), np.nanmin(F, 0), np.nanmax(F, 0),
               *np.nanpercentile(F, [25, 50, 75], axis=0)]
    return np.concatenate(est + [M.mean(0)])

NOM_AGR = [f"{c}_{s}" for s in ["media", "de", "min", "max", "p25", "p50", "p75"] for c in CONT] \
          + [f"prop_{m}" for m in MASK]

df = pd.read_csv(os.path.join(OUT, "videos_etiquetados.csv"))
secs, mascs, agr = [], [], {p: [] for p in FRACCIONES}
for i, r in enumerate(df.itertuples(), 1):
    kp = np.load(os.path.join(KP, os.path.basename(r.key) + ".npy"))
    F, M = por_fotograma(kp, r.ancho, r.alto)
    secs.append(F); mascs.append(M)
    for p in FRACCIONES:
        n = math.ceil(p * len(F))
        agr[p].append(agregar(F[:n], M[:n]))
    if i % 500 == 0:
        print(f"{i}/{len(df)} videos")

# Estandarización con estadísticos de entrenamiento únicamente
ent = np.concatenate([s for s, sp in zip(secs, df.split) if sp == "train"])
mu, de = np.nanmean(ent, 0), np.nanstd(ent, 0)
de[de < 1e-8] = 1.0
pd.DataFrame({"caracteristica": CONT, "media_train": mu, "de_train": de}).to_csv(
    os.path.join(OUT, "estandarizacion.csv"), index=False)

todo = np.concatenate(secs)
nan_frac = np.isnan(todo).mean(0)
X = np.concatenate([np.nan_to_num((np.concatenate(secs) - mu) / de, nan=0.0),
                    np.concatenate(mascs)], axis=1).astype(np.float32)
long_ = np.array([len(s) for s in secs])
np.savez_compressed(os.path.join(FEAT, "secuencias.npz"), X=X,
                    offsets=np.concatenate([[0], np.cumsum(long_)[:-1]]), longitudes=long_,
                    keys=df.key.values.astype(str), split=df.split.values.astype(str), y=df.y.values,
                    columnas=np.array(CONT + MASK))
for p in FRACCIONES:
    A = pd.DataFrame(np.vstack(agr[p]), columns=NOM_AGR)
    A.insert(0, "y", df.y.values); A.insert(0, "split", df.split.values); A.insert(0, "key", df.key.values)
    A.to_csv(os.path.join(FEAT, f"agregadas_p{int(p*100)}.csv"), index=False)

L = [f"Videos: {len(df)} | fotogramas: {len(X)} | umbral de confianza {UMBRAL}",
     f"Secuencias: {len(CONT)} continuas + {len(MASK)} máscaras = {X.shape[1]} por fotograma | "
     f"longitud min {long_.min()} | mediana {int(np.median(long_))} | max {long_.max()}",
     f"Agregadas: {len(NOM_AGR)} características por video | fracciones {FRACCIONES}",
     f"Rango ángulos (todo el conjunto): {np.nanmin(todo[:, :4]):.1f} a {np.nanmax(todo[:, :4]):.1f} grados",
     "Fotogramas sin dato aún (NaN antes del primer punto válido), por característica:"]
L += [f"  {c}: {v:.2%}" for c, v in zip(CONT, nan_frac)]
L.append(f"Videos con alguna característica sin dato en todo el video: "
         f"{int(sum(np.isnan(s).all(0).any() for s in secs))}")
open(os.path.join(OUT, "caracteristicas_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
