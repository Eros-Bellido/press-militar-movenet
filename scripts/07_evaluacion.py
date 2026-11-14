"""
07_evaluacion.py
Evaluación final en el conjunto de PRUEBA (se ejecuta una sola vez), con los umbrales elegidos en validación.

Entradas : outputs/predicciones_lgbm.csv, predicciones_lstm.csv, predicciones_tcn.csv, predicciones_transformer.csv
Salidas  : outputs/resultados_prueba.csv        métricas por modelo, semilla y fracción
           outputs/resultados_prueba_resumen.csv media y DE por modelo y fracción
           outputs/mcnemar.csv                  mejor secuencial vs LightGBM, por semilla y fracción (Holm)
           outputs/fig_f1_fraccion.png          F1 de la clase incorrecta según fracción observada
           outputs/evaluacion_resumen.txt
Uso      : python scripts/07_evaluacion.py
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, accuracy_score

OUT = "outputs"
ARCH = {"LightGBM": "lgbm", "LSTM": "lstm", "TCN": "tcn", "Transformer": "transformer"}
SECUENCIALES = ["LSTM", "TCN", "Transformer"]
FRACCIONES = [25, 50, 75, 100]
ALFA = 0.05

P = pd.concat([pd.read_csv(os.path.join(OUT, f"predicciones_{a}.csv")).assign(modelo=m)
               for m, a in ARCH.items()], ignore_index=True)
P["key"] = P["key"].astype(str)
P["pred"] = (P.prob >= P.umbral).astype(int)

# Comprobaciones: mismos videos en todos los modelos y un umbral por semilla
claves = P[P.split == "test"].groupby("modelo").key.apply(lambda s: frozenset(s))
assert claves.nunique() == 1, "Los modelos no tienen los mismos videos de prueba"
assert (P.groupby(["modelo", "semilla"]).umbral.nunique() == 1).all()

def metricas(d):
    return pd.Series({"F1": f1_score(d.y, d.pred, zero_division=0),
                      "Precision": precision_score(d.y, d.pred, zero_division=0),
                      "Recall": recall_score(d.y, d.pred, zero_division=0),
                      "PR_AUC": average_precision_score(d.y, d.prob),
                      "Exactitud": accuracy_score(d.y, d.pred),
                      "n": len(d)})

R = {sp: P[P.split == sp].groupby(["modelo", "semilla", "fraccion"]).apply(metricas, include_groups=False).reset_index()
     for sp in ("val", "test")}
R["test"].to_csv(os.path.join(OUT, "resultados_prueba.csv"), index=False)
agg = R["test"].groupby(["modelo", "fraccion"])[["F1", "Precision", "Recall", "PR_AUC", "Exactitud"]].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(os.path.join(OUT, "resultados_prueba_resumen.csv"), index=False)

# Mejor secuencial: mayor F1 medio en VALIDACIÓN (criterio fijado en la metodología)
f1_val = R["val"].groupby("modelo").F1.mean()
mejor = f1_val[SECUENCIALES].idxmax()

# McNemar exacto: mejor secuencial vs LightGBM, misma semilla y fracción; Holm sobre las 4 fracciones
filas = []
for s in sorted(P.semilla.unique()):
    for f in FRACCIONES:
        a = P[(P.split == "test") & (P.modelo == mejor) & (P.semilla == s) & (P.fraccion == f)].set_index("key")
        b = P[(P.split == "test") & (P.modelo == "LightGBM") & (P.semilla == s) & (P.fraccion == f)].set_index("key")
        b = b.loc[a.index]
        ok_a, ok_b = a.pred == a.y, b.pred == b.y
        n01 = int((~ok_a & ok_b).sum())     # acierta LightGBM, falla el secuencial
        n10 = int((ok_a & ~ok_b).sum())     # acierta el secuencial, falla LightGBM
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
        filas.append(dict(semilla=s, fraccion=f, solo_secuencial=n10, solo_lightgbm=n01, p=p))
M = pd.DataFrame(filas)
def holm(g):
    orden = g.p.sort_values().index
    k, ajust, prev = len(g), {}, 0.0
    for i, idx in enumerate(orden):
        prev = max(prev, min(1.0, (k - i) * g.loc[idx, "p"]))
        ajust[idx] = prev
    return pd.Series(ajust)
M["p_holm"] = M.groupby("semilla", group_keys=False).apply(holm, include_groups=False)
M["significativo"] = M.p_holm < ALFA
M["favorece"] = np.where(M.solo_secuencial > M.solo_lightgbm, mejor,
                         np.where(M.solo_secuencial < M.solo_lightgbm, "LightGBM", "empate"))
M.insert(0, "comparacion", f"{mejor} vs LightGBM")
M.to_csv(os.path.join(OUT, "mcnemar.csv"), index=False)

# Figura
plt.figure(figsize=(6, 3.6))
estilos = {"LightGBM": ("o", "-"), "LSTM": ("s", "--"), "TCN": ("^", "-."), "Transformer": ("D", ":")}
for m in ARCH:
    d = agg[agg.modelo == m]
    plt.errorbar(d.fraccion, d.F1_mean, yerr=d.F1_std, marker=estilos[m][0], linestyle=estilos[m][1],
                 capsize=3, color="black", markersize=5, label=m)
plt.xticks(FRACCIONES, [f"{f} %" for f in FRACCIONES])
plt.xlabel("Fracción observada del video"); plt.ylabel("F1 de la clase incorrecta")
plt.legend(frameon=False, fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_f1_fraccion.png"), dpi=300); plt.close()

# Resumen
L = [f"Conjunto de prueba: {R['test'].n.iloc[0]:.0f} videos | semillas {sorted(P.semilla.unique())} | umbrales de validación",
     "F1 medio en VALIDACIÓN (criterio de selección): " + ", ".join(f"{m} {v:.4f}" for m, v in f1_val.items()),
     f"Mejor modelo secuencial (validación): {mejor}", "",
     "PRUEBA – media ± DE (5 semillas):"]
for m in ARCH:
    for f in FRACCIONES:
        r = agg[(agg.modelo == m) & (agg.fraccion == f)].iloc[0]
        L.append(f"  {m:<11} {f:>3} %: F1 {r.F1_mean:.4f} ± {r.F1_std:.4f} | P {r.Precision_mean:.4f} | "
                 f"R {r.Recall_mean:.4f} | PR-AUC {r.PR_AUC_mean:.4f} ± {r.PR_AUC_std:.4f} | Exact. {r.Exactitud_mean:.4f}")
L += ["", f"McNemar exacto {mejor} vs LightGBM (Holm sobre 4 fracciones, alfa {ALFA}):"]
for f in FRACCIONES:
    d = M[M.fraccion == f]
    L.append(f"  {f:>3} %: significativo en {int(d.significativo.sum())}/5 semillas | p_holm mediana {d.p_holm.median():.4f} | "
             f"favorece: " + ", ".join(f"{k} {v}" for k, v in d.favorece.value_counts().items()))
open(os.path.join(OUT, "evaluacion_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
