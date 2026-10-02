# setup_and_run.ps1
# ==================
# One-shot setup + full pipeline runner for the Deepfake-rPPG KYC project.
# Skips steps whose outputs already exist (idempotent).
# Supports GPU-accelerated rPPG feature extraction via YuNet (ONNX Runtime CUDA).
#
# Usage:
#   .\setup_and_run.ps1                          # full run (all missing stages)
#   .\setup_and_run.ps1 -Quick                   # smoke test (max 20 videos per class)
#   .\setup_and_run.ps1 -SkipExtract             # skip rPPG feature extraction
#   .\setup_and_run.ps1 -SkipFrontend            # skip npm install + frontend build
#   .\setup_and_run.ps1 -Video path.mp4          # run inference on a sample video at the end
#   .\setup_and_run.ps1 -CpuOnly                 # force CPU extraction (disable GPU)
#   .\setup_and_run.ps1 -Resume                  # resume rPPG extraction from checkpoint
#   .\setup_and_run.ps1 -GpuWorkers 4            # set GPU worker count (default: 8)
#   .\setup_and_run.ps1 -Start 100 -End 200      # process a specific index range
param(
    [switch]$Quick,
    [switch]$SkipExtract,
    [switch]$SkipFrontend,
    [switch]$CpuOnly,
    [switch]$Resume,
    [string]$Video = "",
    [int]$GpuWorkers = 0,      # 0 = auto (min(8, CPU cores))
    [int]$Start = -1,
    [int]$End = -1
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv     = Join-Path $RepoRoot 'venv\Scripts\python.exe'
$Pip      = Join-Path $RepoRoot 'venv\Scripts\pip.exe'
$Working  = Join-Path $RepoRoot 'WORKING'
$Frontend = Join-Path $RepoRoot 'frontend'

# Disable torch.compile (inductor/triton not available on Windows)
$env:TORCH_COMPILE_DISABLE = '1'

# Load .env file if present
$EnvFile = Join-Path $RepoRoot '.env'
if (Test-Path -LiteralPath $EnvFile) {
    Write-Host "  Loading .env from $EnvFile" -ForegroundColor Cyan
    $trimChars = " `t`r`n`"'"
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $name = $matches[1]
            $value = $matches[2].Trim($trimChars)
            $existing = Get-Item -Path "Env:$name" -ErrorAction SilentlyContinue
            if ($existing -and -not [string]::IsNullOrEmpty($existing.Value)) {
                Write-Host "    $name already set in environment, skipping" -ForegroundColor DarkGray
            } else {
                Set-Item -Path "Env:$name" -Value $value
                Write-Host "    Set $name=$value" -ForegroundColor DarkGray
            }
        }
    }
}

# Default DFDC dataset path
if (-not $env:DFDC_DATASET_PATH) {
    $env:DFDC_DATASET_PATH = Join-Path $RepoRoot 'DFDC_Dataset'
}
Write-Host "  DFDC_DATASET_PATH = $env:DFDC_DATASET_PATH" -ForegroundColor DarkGray

# Output root (MAJ_OUTPUT_ROOT env or Scrape/output)
$OutRoot = $env:MAJ_OUTPUT_ROOT
if (-not $OutRoot) {
    $OutRoot = Join-Path $RepoRoot 'Scrape\output'
    $env:MAJ_OUTPUT_ROOT = $OutRoot
}
Write-Host "  MAJ_OUTPUT_ROOT   = $OutRoot" -ForegroundColor DarkGray

# ---------- output paths (check these to decide what to skip) ----------
$RppgCsv       = Join-Path $OutRoot 'rppg\dataset_features.csv'
$RppgPkl       = Join-Path $OutRoot 'rppg\rppg_classifier.pkl'
$FusedCsv      = Join-Path $OutRoot 'visual\fused_features.csv'
$QuantumData   = Join-Path $OutRoot 'quantum\data_fused.npz'
$QuantumVqc    = Join-Path $OutRoot 'quantum\hybrid_vqc_fused.pt'
$QuantumSel    = Join-Path $OutRoot 'quantum\qaoa_selection_fused.json'
$QuantumScaler = Join-Path $OutRoot 'quantum\feature_scaler_fused.json'
$FrontendDist  = Join-Path $Frontend 'dist'

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
# Calculate total steps up front for progress display
$TotalSteps = 7   # venv, extract, train, visual fuse, quantum, frontend build, summary
if (-not $SkipFrontend) { $TotalSteps += 1 }   # npm install
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
    $stderr_log = Join-Path $OutRoot 'pip_install_stderr.log'
    $null = New-Item -ItemType Directory -Force -Path $OutRoot
    & $Pip install -r (Join-Path $RepoRoot 'requirements.txt') 2> $stderr_log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: pip install. See $stderr_log" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Packages installed" -ForegroundColor Green
}

# Verify GPU stack (CUDA, cuDNN, ORT GPU)
Write-Host "  Verifying CUDA/ORT GPU stack..."
$gpuCheck = & $Venv -c "
import torch
import onnxruntime as ort
print(f'Torch CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  cuDNN: {torch.backends.cudnn.version()}')
print(f'ORT providers: {ort.get_available_providers()}')
" 2>&1
Write-Host "  $gpuCheck" -ForegroundColor DarkGray

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

# ----- Step 3: rPPG feature extraction (GPU-accelerated) -----
$step++
Step $step $TotalSteps "rPPG feature extraction (POS, DFDC, GPU-accelerated)"
if ($SkipExtract) {
    Write-Host "  [skip] -SkipExtract flag" -ForegroundColor DarkGray
} elseif (Test-Artifact $RppgCsv) {
    Write-Host "  Dataset already extracted" -ForegroundColor Green
} else {
    $extractArgs = @('--method', 'POS', '--output', $RppgCsv)

    # GPU settings (default: enabled, 8 workers)
    $useGpu = -not $CpuOnly
    if ($useGpu) {
        # Validate GPU stack: onnxruntime-gpu 1.20.2 requires CUDA 12.x
        # Check if CUDA 12.x is available (CUDA 13.0 is not compatible)
        $cuda12Bin = & $Venv -c "
import os
for p in [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin']:
    if os.path.isdir(p):
        print(p)
        break
" 2>$null
        if (-not $cuda12Bin -or [string]::IsNullOrEmpty($cuda12Bin)) {
            Write-Host "  WARNING: CUDA 12.x not found (onnxruntime-gpu 1.20.2 requires CUDA 12.x)" -ForegroundColor Yellow
            Write-Host "  Falling back to CPU mode (MediaPipe). Use -CpuOnly to suppress this warning." -ForegroundColor Yellow
            $useGpu = $false
        }
    }

    if ($useGpu) {
        $extractArgs += @('--gpu')
        if ($GpuWorkers -gt 0) {
            $extractArgs += @('--gpu-workers', $GpuWorkers.ToString())
        }
        Write-Host "  GPU mode: YuNet ONNX Runtime CUDA" -ForegroundColor Green

        # Ensure PyTorch's cuDNN DLLs are on PATH for ONNX Runtime CUDA provider
        $torchLib = & $Venv -c "import torch, os; lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(lib) if os.path.isdir(lib) else print('')" 2>$null
        if ($torchLib -and -not [string]::IsNullOrEmpty($torchLib)) {
            $env:PATH = $torchLib + ";" + $env:PATH
            Write-Host "  Added torch lib to PATH: $torchLib" -ForegroundColor DarkGray
        }
        # Also ensure CUDA 12.x bin is on PATH (for cudart, cublas, etc.)
        if ($cuda12Bin -and -not [string]::IsNullOrEmpty($cuda12Bin)) {
            $env:PATH = $cuda12Bin + ";" + $env:PATH
            Write-Host "  Added CUDA 12.x bin to PATH: $cuda12Bin" -ForegroundColor DarkGray
        }
    } else {
        $extractArgs += @('--no-gpu')
        $extractArgs += @('--workers', '0')  # 0 = all CPU cores
        Write-Host "  CPU-only mode (MediaPipe)" -ForegroundColor Yellow
    }

    # Quality gates
    $extractArgs += @('--min-sqi', '0.05')
    $extractArgs += @('--max-nan-features', '2')

    # Quick mode
    if ($Quick) {
        $extractArgs += @('--max-per-class', '20')
        Write-Host "  Quick mode: max 20 per class" -ForegroundColor Yellow
    }

    # Resume support
    if ($Resume) {
        $extractArgs += @('--resume')
        Write-Host "  Resume mode: will continue from checkpoint" -ForegroundColor Yellow
    }
    if ($Start -ge 0) {
        $extractArgs += @('--start', $Start.ToString())
    }
    if ($End -ge 0) {
        $extractArgs += @('--end', $End.ToString())
    }

    $stderr_log = Join-Path $OutRoot 'rppg\extract_stderr.log'
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'rppg')

    Write-Host "  Extracting features..."
    Write-Host "  stderr log: $stderr_log"
    & $Venv (Join-Path $Working 'RPPG\extract_dataset_features.py') @extractArgs 2> $stderr_log
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
Step $step $TotalSteps "Train rPPG RandomForest classifier"
if (Test-Artifact $RppgPkl) {
    Write-Host "  Classifier already trained" -ForegroundColor Green
} else {
    $trainArgs = @(
        '--features-csv', $RppgCsv,
        '--model-out', $RppgPkl,
        '--metadata-out', (Join-Path $OutRoot 'rppg\rppg_classifier_metadata.json')
    )
    & $Venv (Join-Path $Working 'RPPG\train_classifier.py') @trainArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: classifier training"; exit 1 }
    Write-Host "  Classifier trained: $RppgPkl" -ForegroundColor Green
}

# ----- Step 5: Visual extraction + fusion -----
$step++
Step $step $TotalSteps "Visual extraction + fusion (ResNet50 + handcrafted -> fused CSV)"
if (Test-Artifact $FusedCsv) {
    Write-Host "  Fused features already present" -ForegroundColor Green
} elseif (-not (Test-Path -LiteralPath $RppgCsv)) {
    Write-Host "FAILED: $RppgCsv not found (run extraction first)" -ForegroundColor Red
    exit 1
} else {
    $visualDir = Join-Path $OutRoot 'visual'
    $framesRoot = Join-Path $OutRoot 'frames\frame_sequences'
    $null = New-Item -ItemType Directory -Force -Path $visualDir

    # Check if stage-1 frames exist
    if (Test-Path $framesRoot) {
        Write-Host "  Using stage-1 frames from $framesRoot" -ForegroundColor Green
        Write-Host "  Extracting visual features for rPPG-CSV videos, fusing, creating splits..."
        & $Venv (Join-Path $Working 'visual\pipeline.py') `
            --frames-root $framesRoot `
            --rppg-csv $RppgCsv `
            --output-dir $visualDir `
            --fuse --create-splits
    } else {
        Write-Host "  Stage-1 frames not found; extracting visual features directly from videos" -ForegroundColor Yellow
        Write-Host "  Extracting visual features from videos, fusing, creating splits..."
        & $Venv (Join-Path $Working 'visual\pipeline.py') `
            --video-root $env:DFDC_DATASET_PATH `
            --rppg-csv $RppgCsv `
            --output-dir $visualDir `
            --fuse --create-splits
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: visual extraction/fusion (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path -LiteralPath $FusedCsv)) {
        Write-Host "FAILED: $FusedCsv not found" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Fused features: $FusedCsv" -ForegroundColor Green
}

# ----- Step 6: Quantum pipeline (build data + QAOA select + train VQC + evaluate + baselines) -----
$step++
Step $step $TotalSteps "Quantum pipeline (--all: data + QAOA select + VQC train + eval + baselines)"
$quantumArtifacts = @($QuantumData, $QuantumVqc, $QuantumSel, $QuantumScaler)
$allQuantumExist = $true
foreach ($a in $quantumArtifacts) {
    if (-not (Test-Artifact $a)) { $allQuantumExist = $false; break }
}
if ($allQuantumExist) {
    Write-Host "  All quantum artifacts exist" -ForegroundColor Green
} else {
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'quantum')
    Write-Host "  Running full quantum flow (build + select + train + evaluate + baselines)..."
    Push-Location $Working
    & $Venv -m quantum.pipeline --all
    $exit = $LASTEXITCODE
    Pop-Location
    if ($exit -ne 0) { Write-Host "FAILED: quantum pipeline (exit $exit)"; exit 1 }
    Write-Host "  Quantum pipeline complete" -ForegroundColor Green
}

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
        $outJson = Join-Path $OutRoot 'pipeline\pipeline_result.json'
        $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'pipeline')
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
Write-Host "  Stage 1 (frames):   $OutRoot\frames\"
Write-Host "  Stage 2 (rPPG):     $OutRoot\rppg\"
Write-Host "  Stage 3 (visual):   $OutRoot\visual\"
Write-Host "  Stage 4 (quantum):  $OutRoot\quantum\"
Write-Host "  Pipeline result:    $OutRoot\pipeline\pipeline_result.json"
Write-Host ""
Write-Host "Project structure:"
Write-Host "  WORKING/"
Write-Host "    frame/           Stage 1: YOLO face detection + quality gating (30 fps)"
Write-Host "    RPPG/            Stage 2: MediaPipe/YuNet -> POS/CHROM -> 24 features"
Write-Host "    visual/          Stage 3: ResNet50 + handcrafted -> 39 visual features"
Write-Host "    quantum/         Stage 4: QAOA(63->3) -> Hybrid VQC -> P(real)"
Write-Host "    run_pipeline.py  End-to-end orchestrator"
Write-Host "  frontend/          React + Vite UI (server.py API on :8000)"
Write-Host ""
Write-Host "GPU-accelerated rPPG extraction:"
Write-Host "  Default: YuNet ONNX Runtime CUDA (8 workers)"
Write-Host "  Override: -CpuOnly for MediaPipe CPU, -GpuWorkers N to set parallelism"
Write-Host "  Checkpoint/resume: -Resume, -Start N, -End N"
Write-Host ""
Write-Host "To run the web UI:"
Write-Host "  Terminal 1:  cd frontend; python server.py"
Write-Host "  Terminal 2:  cd frontend; npm run dev"
Write-Host ""
Write-Host "To run inference on any video:"
Write-Host "  cd WORKING"
Write-Host "  python run_pipeline.py --source VIDEO.mp4 --method POS"
Write-Host ""
Write-Host "To re-run full rPPG extraction with GPU:"
Write-Host "  cd WORKING"
Write-Host "  python RPPG/extract_dataset_features.py --method POS --gpu --gpu-workers 8"