# setup_and_run.ps1
# ==================
# One-shot setup + full pipeline runner for the Deepfake-rPPG KYC project.
# Skips steps whose outputs already exist (idempotent).
#
# Usage:
#   .\setup_and_run.ps1                  full run (all missing stages)
#   .\setup_and_run.ps1 -Quick           smoke test (max 20 videos for extraction)
#   .\setup_and_run.ps1 -SkipExtract     skip rPPG feature extraction
#   .\setup_and_run.ps1 -SkipFrontend    skip npm install + frontend build
#   .\setup_and_run.ps1 -Video path.mp4  run inference on a sample video at the end
param(
    [switch]$Quick,
    [switch]$SkipExtract,
    [switch]$SkipFrontend,
    [string]$Video = ""
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv     = Join-Path $RepoRoot 'venv\Scripts\python.exe'
$Pip      = Join-Path $RepoRoot 'venv\Scripts\pip.exe'
$Working  = Join-Path $RepoRoot 'WORKING'
$Frontend = Join-Path $RepoRoot 'frontend'

# ---------- output paths (check these to decide what to skip) ----------
$RppgCsv      = Join-Path $Working 'output\rppg\dataset_features.csv'
$RppgPkl      = Join-Path $Working 'output\rppg\rppg_classifier.pkl'
$QuantumData  = Join-Path $Working 'output\quantum\data.npz'
$QuantumVqc   = Join-Path $Working 'output\quantum\hybrid_vqc.pt'
$QuantumSel   = Join-Path $Working 'output\quantum\qaoa_selection.json'
$QuantumScaler= Join-Path $Working 'output\quantum\feature_scaler.json'
$FrontendDist = Join-Path $Frontend 'dist'

function Test-Artifact($path) {
    if (Test-Path -LiteralPath $path) {
        Write-Host "  [skip] exists: $path" -ForegroundColor DarkGray
        return $true
    }
    return $false
}

function Step($num, $total, $label) {
    Write-Host ""
    Write-Host "===== [$num/$total] $label =====" -ForegroundColor Cyan
}

# =====================================================================
$TotalSteps = 7
if (-not $SkipFrontend) { $TotalSteps += 1 }   # npm install + build
if ($Video -ne "")       { $TotalSteps += 1 }   # sample inference

$step = 0

# ----- Step 1: Python venv + pip install -----
$step++
Step $step $TotalSteps "Python environment"
if (Test-Artifact $Venv) {
    Write-Host "  venv OK" -ForegroundColor Green
} else {
    Write-Host "  Creating venv..."
    & python -m venv (Join-Path $RepoRoot 'venv')
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: python -m venv"; exit 1 }
    Write-Host "  venv created" -ForegroundColor Green
}

# Check if key packages are already installed (avoid slow reinstall)
$pkgsInstalled = & $Venv -c "import torch, numpy, scipy, sklearn; print('ok')" 2>$null
if ($pkgsInstalled -eq 'ok') {
    Write-Host "  Core packages already installed" -ForegroundColor Green
} else {
    Write-Host "  Installing Python packages (this may take a few minutes)..."
    $stderr_log = Join-Path $Working 'output\pip_install_stderr.log'
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $Working 'output')
    & $Pip install -r (Join-Path $RepoRoot 'requirements.txt') 2> $stderr_log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: pip install. See $stderr_log" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Packages installed" -ForegroundColor Green
}

# ----- Step 2: npm install (frontend) -----
if (-not $SkipFrontend) {
    $step++
    Step $step $TotalSteps "Frontend npm install"
    if (Test-Path (Join-Path $Frontend 'node_modules')) {
        Write-Host "  [skip] node_modules exists" -ForegroundColor DarkGray
    } else {
        Write-Host "  Running npm install..."
        Push-Location $Frontend
        & npm install
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: npm install"; Pop-Location; exit 1 }
        Pop-Location
        Write-Host "  npm install done" -ForegroundColor Green
    }
}

# ----- Step 3: rPPG feature extraction -----
$step++
Step $step $TotalSteps "rPPG feature extraction"
if ($SkipExtract) {
    Write-Host "  [skip] -SkipExtract flag" -ForegroundColor DarkGray
} elseif (Test-Artifact $RppgCsv) {
    Write-Host "  Dataset already extracted" -ForegroundColor Green
} else {
    $extractArgs = @('--method', 'POS', '--output', $RppgCsv)
    if ($Quick) {
        $extractArgs += @('--max-per-class', '20')
        Write-Host "  Quick mode: max 20 per class" -ForegroundColor Yellow
    }
    $extractArgs += '--include-ffpp'
    $extractArgs += @('--workers', '0')

    $stderr_log = Join-Path $Working 'output\rppg\extract_stderr.log'
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $Working 'output\rppg')

    Write-Host "  Extracting features (workers=0=all cores, includes FF++)..."
    Write-Host "  stderr log: $stderr_log"
    & $Venv (Join-Path $Working 'RPPG\rppg-pipeline\extract_dataset_features.py') @extractArgs 2> $stderr_log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: feature extraction (exit $LASTEXITCODE). See $stderr_log" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path -LiteralPath $RppgCsv)) {
        Write-Host "FAILED: extraction finished but $RppgCsv not found" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Features extracted: $RppgCsv" -ForegroundColor Green
}

# ----- Step 4: Train rPPG classifier -----
$step++
Step $step $TotalSteps "Train rPPG classifier"
if (Test-Artifact $RppgPkl) {
    Write-Host "  Classifier already trained" -ForegroundColor Green
} else {
    $trainArgs = @(
        '--features-csv', $RppgCsv,
        '--model-out', $RppgPkl,
        '--metadata-out', (Join-Path $Working 'output\rppg\rppg_classifier_metadata.json')
    )
    & $Venv (Join-Path $Working 'RPPG\rppg-pipeline\train_classifier.py') @trainArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: classifier training"; exit 1 }
    Write-Host "  Classifier trained: $RppgPkl" -ForegroundColor Green
}

# ----- Step 5: Quantum pipeline (build data + QAOA select + train VQC + evaluate + baselines) -----
$step++
Step $step $TotalSteps "Quantum pipeline (--all)"
if ((Test-Artifact $QuantumData) -and (Test-Artifact $QuantumVqc) -and (Test-Artifact $QuantumSel) -and (Test-Artifact $QuantumScaler)) {
    Write-Host "  All quantum artifacts exist" -ForegroundColor Green
} else {
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $Working 'output\quantum')
    Write-Host "  Running full quantum flow (build + select + train + evaluate + baselines)..."
    Push-Location $Working
    & $Venv -m quantum.pipeline --all
    $exit = $LASTEXITCODE
    Pop-Location
    if ($exit -ne 0) { Write-Host "FAILED: quantum pipeline (exit $exit)"; exit 1 }
    Write-Host "  Quantum pipeline complete" -ForegroundColor Green
}

# ----- Step 6: Quantum self-tests -----
$step++
Step $step $TotalSteps "Quantum self-tests"
Push-Location $Working
& $Venv -m quantum.tests
$exit = $LASTEXITCODE
Pop-Location
if ($exit -ne 0) { Write-Host "FAILED: quantum self-tests (exit $exit)"; exit 1 }
Write-Host "  All tests passed" -ForegroundColor Green

# ----- Step 7: Frontend build -----
if (-not $SkipFrontend) {
    $step++
    Step $step $TotalSteps "Frontend build"
    if (Test-Path -LiteralPath $FrontendDist) {
        Write-Host "  [skip] dist/ exists" -ForegroundColor DarkGray
    } else {
        Write-Host "  Building frontend..."
        Push-Location $Frontend
        & npm run build
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: frontend build"; Pop-Location; exit 1 }
        Pop-Location
        Write-Host "  Frontend built: $FrontendDist" -ForegroundColor Green
    }
}

# ----- Step 8 (optional): sample inference -----
if ($Video -ne "") {
    $step++
    Step $step $TotalSteps "Sample inference: $Video"
    $videoPath = Resolve-Path -LiteralPath $Video -ErrorAction SilentlyContinue
    if (-not $videoPath) {
        Write-Host "  [warn] Video not found: $Video -- skipping inference" -ForegroundColor Yellow
    } else {
        $outJson = Join-Path $Working 'output\pipeline\pipeline_result.json'
        $null = New-Item -ItemType Directory -Force -Path (Join-Path $Working 'output\pipeline')
        Push-Location $Working
        & "$Venv" run_pipeline.py --source "$videoPath" --method POS --out "$outJson"
        $exit = $LASTEXITCODE
        Pop-Location
        if ($exit -eq 0) {
            Write-Host "  Verdict saved: $outJson" -ForegroundColor Green
        } elseif ($exit -eq 3) {
            Write-Host "  INCONCLUSIVE (insufficient usable frames)" -ForegroundColor Yellow
        } else {
            Write-Host "  Pipeline exited with code $exit" -ForegroundColor Yellow
        }
    }
}

# ----- Summary -----
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " SETUP + PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Generated artifacts:"
Write-Host "  Stage 1 (frames):   $Working\output\frames\"
Write-Host "  Stage 2 (rPPG):     $Working\output\rppg\"
Write-Host "  Stage 3 (quantum):  $Working\output\quantum\"
Write-Host "  Pipeline result:    $Working\output\pipeline\pipeline_result.json"
Write-Host ""
Write-Host "To run the web UI:"
Write-Host "  Terminal 1:  cd frontend; python server.py"
Write-Host "  Terminal 2:  cd frontend; npm run dev"
Write-Host ""
Write-Host "To run inference on any video:"
Write-Host "  cd WORKING"
Write-Host "  python run_pipeline.py --source VIDEO.mp4 --method POS"
