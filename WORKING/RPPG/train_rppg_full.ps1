# train_rppg_full.ps1 - run existing rPPG extract + train scripts end-to-end
# Usage:
#   .\train_rppg_full.ps1                        full run (all 3293 DFDC videos)
#   .\train_rppg_full.ps1 -MaxPerClass 20        smoke test (20 per class)
#   .\train_rppg_full.ps1 -Workers 8             parallel extraction (default = all CPU cores)
#   .\train_rppg_full.ps1 -Force                 kill any stale extract process and restart
param(
    [int]$MaxPerClass = 0,
    [int]$Workers = 0,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$VENV  = 'C:\Users\JHASHANK\Desktop\Maj_Proj\venv\Scripts\python.exe'
$RPPG  = 'C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\RPPG'
$PIPE  = Join-Path $RPPG 'rppg-pipeline'
$OUT   = 'C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\output\rppg'
$CSV   = Join-Path $OUT 'dataset_features.csv'
$STDERR_LOG = Join-Path $OUT ("extract_stderr_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

# --- guard against duplicate runs -------------------------------------
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*extract_dataset_features.py*' }

if ($existing) {
    if ($Force) {
        foreach ($p in $existing) { Stop-Process -Id $p.ProcessId -Force }
        Start-Sleep 2
        Write-Host "Stopped stale extract process(es). Restarting..."
    } else {
        Write-Host "ERROR: extract_dataset_features.py is already running:"
        $existing | ForEach-Object { Write-Host "  PID $($_.ProcessId): $($_.CommandLine)" }
        Write-Host "Re-run with -Force to stop it and restart, or wait for it to finish."
        exit 1
    }
}

Set-Location -LiteralPath $RPPG

Write-Host '[1/3] Extracting rPPG features from DFDC_Dataset ...'
$extractArgs = @('--method', 'POS')
if ($MaxPerClass -gt 0) { $extractArgs += @('--max-per-class', "$MaxPerClass") }
if ($Workers -gt 0) { $extractArgs += @('--workers', "$Workers") }
$extractArgs += @('--output', $CSV)

# stderr (MediaPipe/TFLite noise) goes to a log so the console stays readable
& $VENV (Join-Path $PIPE 'extract_dataset_features.py') @extractArgs 2> $STDERR_LOG
if ($LASTEXITCODE -ne 0) {
    Write-Host "Feature extraction FAILED (exit $LASTEXITCODE). See $STDERR_LOG"
    exit 1
}

if (-not (Test-Path -LiteralPath $CSV)) {
    Write-Host "ERROR: no features extracted -> $CSV missing"
    exit 1
}

Write-Host "[2/3] Training classifier on $CSV ..."
& $VENV (Join-Path $PIPE 'train_classifier.py') `
    --features-csv $CSV `
    --model-out (Join-Path $RPPG 'rppg_classifier.pkl') `
    --metadata-out (Join-Path $RPPG 'rppg_classifier_metadata.json')
if ($LASTEXITCODE -ne 0) { Write-Host 'Training FAILED'; exit 1 }

Write-Host '[3/3] Verifying saved model ...'
& $VENV -c "import pickle; m=pickle.load(open(r'$(Join-Path $RPPG 'rppg_classifier.pkl')','rb')); print('OK:', type(m).__name__, list(m.named_steps))"
if ($LASTEXITCODE -ne 0) { Write-Host 'Model verification FAILED'; exit 1 }

Write-Host 'Done.'