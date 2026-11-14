# Detección temprana del error de codo en el press militar con MoveNet Thunder

Código y resultados del artículo de investigación (Seminario de Investigación II, Universidad de Lima, 2026-2).

## Datos
Usa el conjunto Fitness-AQA (Parmar et al., 2022), ejercicio OverheadPress. Los videos no se incluyen por su licencia; se solicitan a sus autores. Ubicarlos en `data/`.

## Entorno
Python 3.13.5. Instalar con `pip install -r requirements.txt`.

## Orden de ejecución
1. `01_duracion_videos.py` – duración y fotogramas
2. `02_auditoria.py` – auditoría del dataset
3. `03_extraccion_keypoints.py` – keypoints con MoveNet Thunder
4. `04_caracteristicas.py` – características causales y agregadas
5. `05_lightgbm.py` – modelo de referencia
6. `06_secuenciales.py lstm|tcn|transformer` – modelos secuenciales
7. `07_evaluacion.py` y `07b_figura_fracciones.py` – evaluación en prueba y McNemar
8. `08_conversion_tflite.py` – conversión a TFLite
9. `09_benchmark_android.ps1` – medición en smartphone Android
10. `10_secuenciales_v2.py` – segunda ronda de experimentos

`REGISTRO.md` relaciona cada cifra del artículo con su output.