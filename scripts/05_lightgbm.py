"""
05_lightgbm.py
Modelo de referencia: LightGBM sobre las 146 características agregadas por video.
Un único modelo entrenado con las cuatro fracciones observadas (25, 50, 75 y 100 %).

Entradas : data/features/agregadas_p25.csv, _p50, _p75, _p100
Salidas  : outputs/lgbm_mejores_parametros.json
           outputs/predicciones_lgbm.csv   (probabilidades de validación y prueba, por semilla y fracción)
           outputs/lgbm_resumen.txt        (solo métricas de VALIDACIÓN; la prueba se evalúa en 07)
Uso      : python scripts/05_lightgbm.py
"""
import warnings
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
import os, json, time, numpy as np, pandas as pd, lightgbm as lgb, optuna, sklearn
from sklearn.metrics import f1_score

FEAT, OUT = os.path.join("data", "features"), "outputs"
FRACCIONES = [25, 50, 75, 100]
SEMILLAS = [0, 1, 2, 3, 4]
N_INTENTOS, SEMILLA_OPTUNA, HILOS = 100, 42, 4
UMBRALES = np.round(np.arange(0.05, 0.951, 0.01), 2)

tablas = {p: pd.read_csv(os.path.join(FEAT, f"agregadas_p{p}.csv")) for p in FRACCIONES}
COLS = [c for c in tablas[100].columns if c not in ("key", "split", "y")]

def particion(sp):
    partes = []
    for p in FRACCIONES:
        t = tablas[p][tablas[p].split == sp].copy()
        t["fraccion"] = p
        partes.append(t)
    d = pd.concat(partes, ignore_index=True)
    return d[COLS].values.astype(np.float32), d.y.values, d.fraccion.values, d.key.values

Xtr, ytr, _, _ = particion("train")
Xva, yva, fva, kva = particion("val")
Xte, yte, fte, kte = particion("test")

def mejor_umbral(y, prob):
    f1s = [f1_score(y, prob >= u, zero_division=0) for u in UMBRALES]
    i = int(np.argmax(f1s))
    return float(UMBRALES[i])

def f1_por_fraccion(y, prob, frac, u):
    return {p: f1_score(y[frac == p], prob[frac == p] >= u, zero_division=0) for p in FRACCIONES}

def entrenar(params, semilla):
    m = lgb.LGBMClassifier(objective="binary", n_estimators=2000, random_state=semilla,
                           n_jobs=HILOS, deterministic=True, force_row_wise=True,
                           bagging_freq=1, verbose=-1, **params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

def objetivo(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 4.0))
    prob = entrenar(params, 0).predict_proba(Xva)[:, 1]
    u = mejor_umbral(yva, prob)
    return float(np.mean(list(f1_por_fraccion(yva, prob, fva, u).values())))

t0 = time.perf_counter()
optuna.logging.set_verbosity(optuna.logging.WARNING)
estudio = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=SEMILLA_OPTUNA))
estudio.optimize(objetivo, n_trials=N_INTENTOS, show_progress_bar=True)
mejores = estudio.best_params
json.dump(mejores, open(os.path.join(OUT, "lgbm_mejores_parametros.json"), "w"), indent=2)

filas, f1_val, iters, umbrales = [], [], [], []
for s in SEMILLAS:
    m = entrenar(mejores, s)
    pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    u = mejor_umbral(yva, pva)
    f1_val.append(f1_por_fraccion(yva, pva, fva, u)); iters.append(m.best_iteration_); umbrales.append(u)
    for sp, k, y, f, p in (("val", kva, yva, fva, pva), ("test", kte, yte, fte, pte)):
        filas.append(pd.DataFrame(dict(modelo="LightGBM", semilla=s, split=sp, key=k,
                                       fraccion=f, y=y, prob=p, umbral=u)))
pd.concat(filas).to_csv(os.path.join(OUT, "predicciones_lgbm.csv"), index=False)

F = pd.DataFrame(f1_val)
L = [f"LightGBM {lgb.__version__} | Optuna {optuna.__version__} | scikit-learn {sklearn.__version__}",
     f"Entrenamiento: {len(Xtr)} filas ({len(Xtr)//4} videos x 4 fracciones) | {len(COLS)} características",
     f"Optuna: {N_INTENTOS} intentos, TPE semilla {SEMILLA_OPTUNA} | mejor objetivo (F1 medio val) {estudio.best_value:.4f}",
     "Mejores hiperparámetros: " + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                              for k, v in mejores.items()}),
     f"Fijos: objective=binary, n_estimators=2000 con parada temprana (100 rondas, logloss de validación), bagging_freq=1",
     f"Iteraciones por semilla: {iters} | umbral por semilla: {umbrales}",
     "F1 clase incorrecta en VALIDACIÓN (media ± DE, 5 semillas):"]
L += [f"  {p} %: {F[p].mean():.4f} ± {F[p].std(ddof=1):.4f}" for p in FRACCIONES]
L.append(f"Tiempo total: {(time.perf_counter()-t0)/60:.1f} min")
open(os.path.join(OUT, "lgbm_resumen.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("listo")
