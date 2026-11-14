# 09_benchmark_android.ps1
# Mide en el smartphone Android la latencia de MoveNet Thunder FP16 y de la LSTM paso a paso
# con la herramienta oficial de benchmark de TensorFlow Lite (entradas sintéticas, sin videos ni datos personales).
# Uso (desde la carpeta Seminario2, con el teléfono conectado y autorizado):
#     & .\scripts\09_benchmark_android.ps1
# Salidas: outputs\benchmark\dispositivo.txt y outputs\benchmark\<modelo>_rep<n>.txt
# Al terminar, borra del teléfono todos los archivos copiados.

$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$tmp = "/data/local/tmp"
$url = "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
$HILOS = 4
$REPETICIONES = 3
$PAUSA_S = 60   # enfriamiento entre repeticiones

New-Item -ItemType Directory -Force "outputs\benchmark" | Out-Null

# 1. Herramienta de benchmark (binario oficial para arm64)
if (-not (Test-Path "models\benchmark_model")) {
    Write-Host "Descargando la herramienta de benchmark..."
    Invoke-WebRequest -Uri $url -OutFile "models\benchmark_model"
}
$mb = (Get-Item "models\benchmark_model").Length / 1MB
if ($mb -lt 1) {
    Write-Host "La descarga no es valida ($([math]::Round($mb,3)) MB). Borra models\benchmark_model y avisa." -ForegroundColor Red
    exit 1
}

# 2. Copiar herramienta y modelos al telefono
& $adb push "models\benchmark_model" "$tmp/"
& $adb push "outputs\movil\movenet_thunder_fp16.tflite" "$tmp/"
& $adb push "outputs\movil\lstm_paso.tflite" "$tmp/"
& $adb shell "chmod +x $tmp/benchmark_model"

# 3. Datos del dispositivo
$info = @(
    "fecha: " + (Get-Date -Format "yyyy-MM-dd HH:mm"),
    "modelo: " + (& $adb shell getprop ro.product.model),
    "plataforma: " + (& $adb shell getprop ro.board.platform),
    "android: " + (& $adb shell getprop ro.build.version.release),
    "abi: " + (& $adb shell getprop ro.product.cpu.abi),
    "nucleos: " + (& $adb shell "cat /sys/devices/system/cpu/possible"),
    "bateria: " + ((& $adb shell dumpsys battery | Select-String "level") -join " "),
    "hilos: $HILOS | repeticiones: $REPETICIONES | pausa: $PAUSA_S s | delegado: XNNPACK (CPU)",
    "benchmark_model: $([math]::Round($mb,2)) MB descargado de $url"
)
$info | Set-Content "outputs\benchmark\dispositivo.txt" -Encoding UTF8
$info | ForEach-Object { Write-Host $_ }

# 4. Mediciones
$modelos = @(
    @{ nombre = "movenet"; archivo = "movenet_thunder_fp16.tflite"; corridas = 100 },
    @{ nombre = "lstm";    archivo = "lstm_paso.tflite";            corridas = 1000 }
)
foreach ($m in $modelos) {
    foreach ($i in 1..$REPETICIONES) {
        Write-Host "== $($m.nombre): repeticion $i de $REPETICIONES" -ForegroundColor Cyan
        $cmd = "$tmp/benchmark_model --graph=$tmp/$($m.archivo) --num_threads=$HILOS --use_xnnpack=true --warmup_runs=10 --num_runs=$($m.corridas) 2>&1"
        & $adb shell $cmd | Tee-Object -FilePath "outputs\benchmark\$($m.nombre)_rep$i.txt"
        Start-Sleep -Seconds $PAUSA_S
    }
}

# 5. Limpiar el telefono
& $adb shell "rm -f $tmp/benchmark_model $tmp/movenet_thunder_fp16.tflite $tmp/lstm_paso.tflite"
Write-Host "listo: resultados en outputs\benchmark\" -ForegroundColor Green
