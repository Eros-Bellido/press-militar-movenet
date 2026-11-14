"""
02_auditoria.py — Auditoría del subconjunto OHP etiquetado de Fitness-AQA.
Entradas : data/Labels/error_elbows.json, data/Splits/*_keys.json, outputs/frames.csv
Salidas  : outputs/videos_etiquetados.csv   (una fila por video etiquetado)
           outputs/videos_sin_etiqueta.txt  (videos presentes sin etiqueta ni partición)
           outputs/auditoria_resumen.txt    (cifras para el artículo)
           outputs/fig_duracion.png, outputs/fig_inicio_error.png
Uso      : python scripts/02_auditoria.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
labels = json.load(open(os.path.join("data", "Labels", "error_elbows.json"), encoding="utf-8"))
splits = {s: json.load(open(os.path.join("data", "Splits", f"{s}_keys.json"), encoding="utf-8"))
          for s in ["train", "val", "test"]}
frames = pd.read_csv(os.path.join(OUT, "frames.csv"))
frames["key"] = frames["archivo"].str.replace(r"\.mp4$", "", regex=True)

# --- Cruce videos / etiquetas / particiones ---
split_of = {k: s for s, ks in splits.items() for k in ks}
extras = sorted(set(frames["key"]) - set(labels))
faltan = sorted(set(labels) - set(frames["key"]))
with open(os.path.join(OUT, "videos_sin_etiqueta.txt"), "w") as f:
    f.write("\n".join(extras))

df = frames[frames["key"].isin(labels)].copy()
df["split"] = df["key"].map(split_of)
df["y"] = df["key"].map(lambda k: int(len(labels[k]) > 0))   # regla: >=1 intervalo => incorrecta
df["dur_s"] = df["fotogramas"] / df["fps"]
df["n_intervalos"] = df["key"].map(lambda k: len(labels[k]))
df["inicio_error_s"] = df["key"].map(lambda k: labels[k][0][0] if labels[k] else np.nan)
df["fin_error_s"] = df["key"].map(lambda k: max(i[1] for i in labels[k]) if labels[k] else np.nan)
df["inicio_error_frac"] = df["inicio_error_s"] / df["dur_s"]
df.drop(columns=["archivo"]).to_csv(os.path.join(OUT, "videos_etiquetados.csv"), index=False)

# --- Resumen ---
P = df[df["y"] == 1]
pct = [0, 10, 25, 50, 75, 90, 100]
L = []
L.append(f"Archivos de video: {len(frames)} | etiquetados: {len(df)} | sin etiqueta: {len(extras)} | etiquetas sin video: {len(faltan)}")
L.append(f"Solapes entre particiones: {len(set(splits['train']) & set(splits['val']))}, "
         f"{len(set(splits['train']) & set(splits['test']))}, {len(set(splits['val']) & set(splits['test']))}")
for s in ["train", "val", "test"]:
    d = df[df["split"] == s]
    L.append(f"{s}: n={len(d)} | clase1={d['y'].sum()} ({d['y'].mean():.1%})")
L.append(f"Total: clase0={int((df['y']==0).sum())} | clase1={int(df['y'].sum())}")
L.append(f"FPS distintos: {sorted(df['fps'].unique())}")
L.append("Resoluciones (ancho x alto): " + str((df['ancho'].astype(str) + 'x' + df['alto'].astype(str)).value_counts().to_dict()))
L.append("Fotogramas percentiles " + str(dict(zip(pct, np.percentile(df['fotogramas'], pct).round(1)))))
L.append("Duración (s) percentiles " + str(dict(zip(pct, np.percentile(df['dur_s'], pct).round(2)))) + f" | media {df['dur_s'].mean():.2f}")
L.append(f"Mediana duración clase0={df[df.y==0]['dur_s'].median():.2f}s | clase1={P['dur_s'].median():.2f}s")
L.append(f"p90 fotogramas (train): {np.percentile(df[df.split=='train']['fotogramas'], 90):.1f}")
L.append(f"Positivos con 1 / 2+ intervalos: {(P.n_intervalos==1).sum()} / {(P.n_intervalos>=2).sum()}")
L.append(f"Intervalos que exceden la duración del video (+0.1 s): {(P['fin_error_s'] > P['dur_s'] + 0.1).sum()}")
for f in [0.0, 0.25, 0.50, 0.75, 1.0]:
    L.append(f"Positivos con inicio de error <= {f:.0%} del video: {(P['inicio_error_frac'] <= f).mean():.1%}")
txt = "\n".join(L)
open(os.path.join(OUT, "auditoria_resumen.txt"), "w", encoding="utf-8").write(txt)
print(txt)

# --- Figuras ---
plt.figure(figsize=(6, 3.5))
plt.hist(df["dur_s"], bins=50)
plt.xlabel("Duración del video (s)"); plt.ylabel("Número de videos")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_duracion.png"), dpi=200); plt.close()

plt.figure(figsize=(6, 3.5))
x = np.sort(P["inicio_error_frac"].values)
plt.step(x, np.arange(1, len(x) + 1) / len(x), where="post")
for f in [0.25, 0.5, 0.75]:
    plt.axvline(f, linestyle="--", linewidth=0.8)
plt.xlabel("Fracción del video en que inicia el primer intervalo de error")
plt.ylabel("Proporción acumulada\n(videos incorrectos)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_inicio_error.png"), dpi=200); plt.close()
print("\nlisto: revisa la carpeta outputs/")
