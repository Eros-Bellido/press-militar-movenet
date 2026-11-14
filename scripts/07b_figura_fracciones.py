"""
07b_figura_fracciones.py
Redibuja la figura de F1 según la fracción observada a partir de outputs/resultados_prueba_resumen.csv,
con un pequeño desplazamiento horizontal para que las barras de error no se superpongan.
Salida: outputs/fig_f1_fraccion_v2.png
"""
import os, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "outputs"
d = pd.read_csv(os.path.join(OUT, "resultados_prueba_resumen.csv"))
FR = [25, 50, 75, 100]
est = {"LightGBM": ("o", "-", -3), "LSTM": ("s", "--", -1), "TCN": ("^", "-.", 1), "Transformer": ("D", ":", 3)}
plt.figure(figsize=(6.2, 3.6))
for m, (mk, ls, dx) in est.items():
    s = d[d.modelo == m].set_index("fraccion").loc[FR]
    plt.errorbar([f + dx for f in FR], s.F1_mean, yerr=s.F1_std, marker=mk, linestyle=ls, color="black",
                 capsize=3, markersize=5, linewidth=1.2, elinewidth=0.8, label=m)
plt.xticks(FR, [f"{f} %" for f in FR]); plt.xlim(17, 108)
plt.xlabel("Fracción observada del video"); plt.ylabel("F1 de la clase incorrecta")
plt.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_f1_fraccion_v2.png"), dpi=300); plt.close()
print("listo")
