# Registro de ejecución – Seminario II

2026-10-28 | entorno .venv Python 3.13.5, tensorflow 2.21.0 | requirements.txt
2026-10-28 | 01_duracion_videos.py | outputs/frames.csv | 2367 archivos (2260 etiquetados + 107 sin etiqueta)
2026-10-28 | 02_auditoria.py | auditoria_resumen.txt, videos_etiquetados.csv, 2 figuras | ver resumen

2026-11-04 | 03_extraccion_keypoints.py | data/keypoints/*.npy, keypoints_por_video.csv, keypoints_resumen.txt | 2260 videos, 301103 fotogramas, 0 diferencias con frames.csv; MoveNet Thunder FP16 SHA256 41641538…; XNNPACK, 4 hilos

2026-11-04 | 04_caracteristicas.py | data/features/secuencias.npz, agregadas_p25/50/75/100.csv, estandarizacion.csv, caracteristicas_resumen.txt | 26 por fotograma (20 + 6 máscaras), 146 agregadas; vacíos ≤ 2.31 %; 3 videos con un punto sin detectar

2026-11-04 | 05_lightgbm.py | lgbm_mejores_parametros.json, predicciones_lgbm.csv, lgbm_resumen.txt | Optuna 100 intentos (TPE, semilla 42); 5 semillas; F1 val clase incorrecta 0.50–0.52 según fracción; prueba no evaluada (va en 07)

2026-11-05 | 06_secuenciales.py tcn | optuna_tcn.db, tcn_mejores_parametros.json, modelos/tcn_s0-4.keras, predicciones_tcn.csv, tcn_resumen.txt | Optuna 30 intentos (17 completos, 13 podados); 4 niveles (alcance 61 fotogramas), 64 filtros, 100 609 parámetros; F1 val 0.43–0.48 según fracción; prueba no evaluada

2026-11-05 | 06_secuenciales.py lstm | optuna_lstm.db, lstm_mejores_parametros.json, modelos/lstm_s0-4.keras, predicciones_lstm.csv, lstm_resumen.txt | Optuna 30 intentos (14 completos, 16 podados); 2 capas de 64 unidades, 56 385 parámetros; F1 val 0.45–0.49 según fracción; prueba no evaluada

2026-11-05 | 06_secuenciales.py transformer | optuna_transformer.db, transformer_mejores_parametros.json, modelos/transformer_s0-4.keras, predicciones_transformer.csv, transformer_resumen.txt | Optuna 30 intentos (13 completos, 17 podados); 1 capa, d_model 32, 4 cabezas, 42 209 parámetros; F1 val 0.42–0.47 según fracción; prueba no evaluada

2026-11-05 | 07_evaluacion.py | resultados_prueba.csv, resultados_prueba_resumen.csv, mcnemar.csv, fig_f1_fraccion.png, evaluacion_resumen.txt | evaluación única en prueba (339 videos); mejor secuencial por validación: LSTM; McNemar exacto con Holm

2026-11-06 | 07b_figura_fracciones.py | fig_f1_fraccion_v2.png | Figura 4: F1 según fracción observada, redibujada desde resultados_prueba_resumen.csv con marcadores desplazados

2026-11-06 | 08_conversion_tflite.py | outputs/movil/lstm_paso.tflite, outputs/movil/movenet_thunder_fp16.tflite, conversion_resumen.txt | LSTM semilla 0 (mayor F1 val), paso a paso FP32, 227 KB; dif. máx. Keras-TFLite 4.2e-7; 0/1356 predicciones cambian; F1 de prueba idéntico

2026-11-08 | 09_benchmark_android.ps1 | outputs/benchmark/dispositivo.txt, movenet_rep1-3.txt, lstm_rep1-3.txt | Galaxy A32 SM-A325F (Helio G80, Android 13), 4 hilos, XNNPACK; MoveNet 76.4 ms, LSTM 0.045 ms; FPS estimado 13.1

2026-11-11 | 10_secuenciales_v2.py lstm/tcn/transformer | modelos_v2/, predicciones_*_v2.csv, *_v2_resumen.txt | Segunda ronda: supervisión por fotograma + espejo + ensamble, mismos hiperparámetros. F1 val promedio: LSTM 0.472→0.469, TCN 0.458→0.449, Transformer 0.450→0.449. No mejora; prueba no evaluada

