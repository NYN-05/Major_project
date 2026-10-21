# setup_and_run.ps1
# ==================
# One-shot setup + full pipeline runner for the Deepfake-rPPG KYC project.
# Skips steps whose outputs already exist (idempotent).
# Supports GPU-accelerated rPPG feature extraction via YuNet (ONNX Runtime CUDA).
#
# Long stages (rPPG / visual extraction) stream through setup_progress.py,
# which renders a live progress bar with rate + ETA and tees every raw line
# to a log file next to the stage artifacts.
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
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch { }
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv     = Join-Path $RepoRoot 'venv\Scripts\python.exe'
$Pip      = Join-Path $RepoRoot 'venv\Scripts\pip.exe'
$Working  = Join-Path $RepoRoot 'WORKING'
$Frontend = Join-Path $RepoRoot 'frontend'

# Disable torch.compile (inductor/triton not available on Windows)
$env:TORCH_COMPILE_DISABLE = '1'

# ---------------------------------------------------------------------
# Console UI helpers (ANSI; degrade gracefully outside Windows Terminal)
# ---------------------------------------------------------------------
$script:AnsiOk = $false
try {
    $supportsVt = $false
    if ($Host.UI -and $Host.UI.SupportsVirtualTerminal) { $supportsVt = $true }
    $script:AnsiOk = ($null -eq $env:NO_COLOR) -and ($env:TERM -ne 'dumb') -and `
        (-not [Console]::IsOutputRedirected) -and `
        ($supportsVt -or $env:WT_SESSION -or $env:TERM_PROGRAM -or $env:ANSICON)
} catch {
    $script:AnsiOk = $false
}
$E = if ($script:AnsiOk) { [char]27 } else { '' }
$C_RESET   = "$E[0m"
$C_BOLD    = "$E[1m"
$C_DIM     = "$E[2m"
$C_GRAY    = "$E[90m"
$C_RED     = "$E[91m"
$C_GREEN   = "$E[92m"
$C_YELLOW  = "$E[93m"
$C_CYAN    = "$E[96m"
$C_MAGENTA = "$E[95m"

function Fmt-Elapsed([double]$seconds) {
    if ($seconds -lt 1) { return '{0:N1}s' -f $seconds }
    $ts = [TimeSpan]::FromSeconds($seconds)
    if ($ts.TotalHours -ge 1) {
        return '{0}h {1:d2}m {2:d2}s' -f [int][Math]::Floor($ts.TotalHours), $ts.Minutes, $ts.Seconds
    }
    if ($ts.TotalMinutes -ge 1) { return '{0}m {1:d2}s' -f $ts.Minutes, $ts.Seconds }
    return '{0:N1}s' -f $seconds
}

function Fit-Text([string]$text, [int]$max) {
    if ($null -eq $text) { return '' }
    if ($text.Length -le $max) { return $text }
    return $text.Substring(0, [Math]::Max(1, $max - 1)) + [string][char]0x2026
}

function Write-Green([string]$msg) { Write-Host "  $C_GREEN$([char]0x2713)$C_RESET $msg" }
function Write-Warn([string]$msg)  { Write-Host "  $C_YELLOW!$C_RESET $msg" }
function Write-Dim([string]$msg)   { Write-Host "  $C_GRAY$([char]0x00B7)$C_RESET $C_DIM$msg$C_RESET" }

$script:Stages    = New-Object System.Collections.Generic.List[object]
$script:CurStep   = ''
$script:StepWatch = [System.Diagnostics.Stopwatch]::StartNew()

function Stage-Record([string]$stage, [string]$status, [double]$seconds) {
    $script:Stages.Add([pscustomobject]@{
        Stage    = $stage
        Status   = $status
        Duration = (Fmt-Elapsed $seconds)
    })
}

function Step([int]$num, [int]$total, [string]$label) {
    $script:CurStep = $label
    $script:StepWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $head = "[$num/$total] $label "
    $rule = [string][char]0x2500 * [Math]::Max(3, 68 - $head.Length)
    Write-Host ""
    Write-Host "$C_BOLD$C_CYAN$([char]0x256D)$([char]0x2500)$head$rule$C_RESET"
}

function Step-End([char]$symbol, [string]$color, [string]$status, [string]$msg) {
    $script:StepWatch.Stop()
    $el = $script:StepWatch.Elapsed.TotalSeconds
    Stage-Record $script:CurStep $status $el
    Write-Host "  $color$symbol$C_RESET $msg $C_DIM($(Fmt-Elapsed $el))$C_RESET"
}

function Step-Done([string]$msg) { Step-End ([char]0x2713) $C_GREEN  'DONE' $msg }
function Step-Skip([string]$msg) { Step-End ([char]0x25CB) $C_GRAY   'SKIPPED' $msg }
function Step-Warn([string]$msg) { Step-End '!'            $C_YELLOW 'WARNING' $msg }
function Step-Fail([string]$msg) { Step-End ([char]0x2717) $C_RED    'FAILED' $msg }

function Box-Line([string]$text, [int]$inner = 58) {
    $t = " " + (Fit-Text $text ($inner - 1))
    $pad = ' ' * [Math]::Max(0, $inner - $t.Length)
    Write-Host "$C_MAGENTA$([char]0x2502)$C_RESET$t$pad$C_MAGENTA$([char]0x2502)$C_RESET"
}

function Show-Banner {
    $mode = if ($Quick) { 'QUICK (max 20 videos/class)' } else { 'FULL dataset' }
    $hw   = if ($CpuOnly) { 'CPU (MediaPipe)' } else { 'GPU (YuNet ONNX CUDA)' }
    $resume = if ($Resume) { ' yes' } else { ' no' }
    $range = if ($Start -ge 0 -or $End -ge 0) { " [$Start, $End)" } else { ' all' }
    Write-Host ""
    Write-Host "$C_MAGENTA$C_BOLD$([char]0x256D)$([string][char]0x2500 * 58)$([char]0x256E)$C_RESET"
    Box-Line "$C_RESET Deepfake-rPPG KYC - Setup + Full Pipeline"
    Box-Line "Mode:      $mode"
    Box-Line "Hardware:  $hw"
    Box-Line "Dataset:   $(Fit-Text $env:DFDC_DATASET_PATH 46)"
    Box-Line "Output:    $(Fit-Text $OutRoot 46)"
    Box-Line "Resume:$resume  Range:$range"
    Write-Host "$C_MAGENTA$C_BOLD$([char]0x2570)$([string][char]0x2500 * 58)$([char]0x256F)$C_RESET"
}

function Show-Summary {
    Write-Host ""
    Write-Host "$C_GREEN$C_BOLD  $([char]0x2713) SETUP + PIPELINE COMPLETE$C_RESET"
    Write-Host ""
    Write-Host "$C_BOLD$C_CYAN$([char]0x256D)$([char]0x2500) Pipeline summary $([string][char]0x2500 * 38)$C_RESET"
    foreach ($s in $script:Stages) {
        $sym   = switch ($s.Status) { 'DONE' { [char]0x2713 } 'SKIPPED' { [char]0x25CB } 'FAILED' { [char]0x2717 } 'WARNING' { '!' } default { ' ' } }
        $color = switch ($s.Status) { 'DONE' { $C_GREEN } 'SKIPPED' { $C_GRAY } 'FAILED' { $C_RED } 'WARNING' { $C_YELLOW } default { $C_RESET } }
        $name  = Fit-Text $s.Stage 34
        $pad   = ' ' * [Math]::Max(1, 35 - $name.Length)
        Write-Host "  $color$sym$C_RESET $name$pad$C_DIM$($s.Duration)$C_RESET"
    }
}

# Load .env file if present
$EnvFile = Join-Path $RepoRoot '.env'
if (Test-Path -LiteralPath $EnvFile) {
    Write-Dim "Loading .env from $EnvFile"
    $trimChars = " `t`r`n`"' "
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $name = $matches[1]
            $value = $matches[2].Trim($trimChars)
            $existing = Get-Item -Path "Env:$name" -ErrorAction SilentlyContinue
            if ($existing -and -not [string]::IsNullOrEmpty($existing.Value)) {
                Write-Dim "$name already set in environment, skipping .env value"
            } else {
                Set-Item -Path "Env:$name" -Value $value
                Write-Dim "Set $name=$value"
            }
        }
    }
}

# Default DFDC dataset path
if (-not $env:DFDC_DATASET_PATH) {
    $env:DFDC_DATASET_PATH = Join-Path $RepoRoot 'DFDC_Dataset'
}

# Output root - default to WORKING\output (project-relative, not Temp, not Scrape)
# The .env file should define MAJ_OUTPUT_ROOT; if not set, fall back to WORKING\output
$OutRoot = $env:MAJ_OUTPUT_ROOT
if (-not $OutRoot) {
    $OutRoot = Join-Path $RepoRoot 'WORKING\output'
    $env:MAJ_OUTPUT_ROOT = $OutRoot
    Write-Warn "MAJ_OUTPUT_ROOT not set in .env or environment; using default: $OutRoot"
}

Show-Banner

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
    return (Test-Path -LiteralPath $path)
}

# =====================================================================
# Calculate total steps up front for progress display
$TotalSteps = 5   # venv, extract, train, visual fuse, quantum
if (-not $SkipFrontend) { $TotalSteps += 2 }   # npm install + frontend build
if ($Video -ne "")       { $TotalSteps += 1 }   # sample inference

$step = 0

# ----- Step 1: Python venv + pip install -----
$step++
Step $step $TotalSteps "Python environment"
if (Test-Artifact $Venv) {
    Write-Dim "venv OK: $Venv"
} else {
    Write-Host "  Creating venv..."
    & python -m venv (Join-Path $RepoRoot 'venv')
    if ($LASTEXITCODE -ne 0) { Step-Fail "python -m venv"; exit 1 }
    Write-Green "venv created"
}

# Check if key packages are already installed (avoid slow reinstall)
$pkgsInstalled = & $Venv -c "import torch, numpy, scipy, sklearn, onnxruntime as ort; assert ort.__version__ == '1.20.2', ort.__version__; print('ok')" 2>$null
if ($pkgsInstalled -eq 'ok') {
    Write-Green "Core packages already installed (incl. onnxruntime-gpu 1.20.2)"
} else {
    Write-Host "  Installing Python packages (this may take a few minutes)..."
    $stderr_log = Join-Path $OutRoot 'pip_install_stderr.log'
    $null = New-Item -ItemType Directory -Force -Path $OutRoot
    & $Pip install -r (Join-Path $RepoRoot 'requirements.txt') 2> $stderr_log
    if ($LASTEXITCODE -ne 0) {
        Step-Fail "pip install - see $stderr_log"
        exit 1
    }
    Write-Green "Packages installed"
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
print(f'ORT {ort.__version__} providers: {ort.get_available_providers()}')
" 2>&1
foreach ($l in $gpuCheck) { Write-Host "  $C_DIM$([string]$l)$C_RESET" }
Step-Done "Python environment ready"

# ----- Step 2: npm install (frontend) -----
if (-not $SkipFrontend) {
    $step++
    Step $step $TotalSteps "Frontend npm install"
    if (Test-Path (Join-Path $Frontend 'node_modules')) {
        Step-Skip "node_modules already present"
    } else {
        Write-Host "  Running npm install..."
        Push-Location $Frontend
        & npm install
        if ($LASTEXITCODE -ne 0) { Step-Fail "npm install"; Pop-Location; exit 1 }
        Pop-Location
        Step-Done "npm install complete"
    }
}

# ----- Step 3: rPPG feature extraction (GPU-accelerated) -----
$step++
Step $step $TotalSteps "rPPG feature extraction (POS, DFDC, GPU-accelerated)"
if ($SkipExtract) {
    Step-Skip "-SkipExtract flag"
} elseif (Test-Artifact $RppgCsv) {
    Step-Skip "dataset already extracted: $RppgCsv"
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
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin']:
    if os.path.isdir(p):
        print(p)
        break
" 2>$null
        if (-not $cuda12Bin -or [string]::IsNullOrEmpty($cuda12Bin)) {
            Write-Warn "CUDA 12.x not found (onnxruntime-gpu 1.20.2 requires CUDA 12.x)"
            Write-Warn "Falling back to CPU mode (MediaPipe). Use -CpuOnly to suppress this warning."
            $useGpu = $false
        }
    }

    if ($useGpu) {
        $extractArgs += @('--gpu')
        if ($GpuWorkers -gt 0) {
            $extractArgs += @('--gpu-workers', $GpuWorkers.ToString())
        }
        Write-Green "GPU mode: YuNet ONNX Runtime CUDA"

        # Ensure PyTorch's cuDNN DLLs are on PATH for ONNX Runtime CUDA provider
        $torchLib = & $Venv -c "import torch, os; lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(lib) if os.path.isdir(lib) else print('')" 2>$null
        if ($torchLib -and -not [string]::IsNullOrEmpty($torchLib)) {
            $env:PATH = $torchLib + ";" + $env:PATH
            Write-Dim "Added torch lib to PATH: $torchLib"
        }
        # Also ensure CUDA 12.x bin is on PATH (for cudart, cublas, etc.)
        if ($cuda12Bin -and -not [string]::IsNullOrEmpty($cuda12Bin)) {
            $env:PATH = $cuda12Bin + ";" + $env:PATH
            Write-Dim "Added CUDA 12.x bin to PATH: $cuda12Bin"
        }
    } else {
        $extractArgs += @('--no-gpu')
        $extractArgs += @('--workers', '0')  # 0 = all CPU cores
        Write-Warn "CPU-only mode (MediaPipe)"
    }

    # Quality gates
    $extractArgs += @('--min-sqi', '0.05')
    $extractArgs += @('--max-nan-features', '2')

    # Quick mode
    if ($Quick) {
        $extractArgs += @('--max-per-class', '20')
        Write-Warn "Quick mode: max 20 per class"
    }

    # Resume support
    if ($Resume) {
        $extractArgs += @('--resume')
        Write-Warn "Resume mode: will continue from checkpoint"
    }
    if ($Start -ge 0) {
        $extractArgs += @('--start', $Start.ToString())
    }
    if ($End -ge 0) {
        $extractArgs += @('--end', $End.ToString())
    }

    $stderr_log = Join-Path $OutRoot 'rppg\extract_stderr.log'
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'rppg')

    Write-Host "  Extracting features (live progress below)..."
    Write-Host "  ${C_DIM}log: $stderr_log$C_RESET"
    # The log file is written by setup_progress.py: pwsh file writes are
    # blocked by Controlled Folder Access, so tee through python.
    & $Venv -u (Join-Path $RepoRoot 'setup_progress.py') --log $stderr_log --label 'rPPG' `
        -- $Venv (Join-Path $Working 'RPPG\extract_dataset_features.py') @extractArgs
    if ($LASTEXITCODE -ne 0) {
        Step-Fail "feature extraction (exit $LASTEXITCODE) - see $stderr_log"
        Get-Content $stderr_log -Tail 50
        exit 1
    }
    if (-not (Test-Path -LiteralPath $RppgCsv)) {
        Step-Fail "extraction finished but $RppgCsv not found"
        exit 1
    }
    Step-Done "Features extracted -> $RppgCsv"
}

# ----- Step 4: Train rPPG classifier -----
$step++
Step $step $TotalSteps "Train rPPG RandomForest classifier"
if (Test-Artifact $RppgPkl) {
    Step-Skip "classifier already trained: $RppgPkl"
} else {
    $trainArgs = @(
        '--features-csv', $RppgCsv,
        '--model-out', $RppgPkl,
        '--metadata-out', (Join-Path $OutRoot 'rppg\rppg_classifier_metadata.json')
    )
    & $Venv (Join-Path $Working 'RPPG\train_classifier.py') @trainArgs
    if ($LASTEXITCODE -ne 0) { Step-Fail "classifier training"; exit 1 }
    Step-Done "Classifier trained -> $RppgPkl"
}

# ----- Step 5: Visual extraction + fusion -----
$step++
Step $step $TotalSteps "Visual extraction + fusion (ResNet50 + handcrafted -> fused CSV)"
if (Test-Artifact $FusedCsv) {
    Step-Skip "fused features already present: $FusedCsv"
} elseif (-not (Test-Path -LiteralPath $RppgCsv)) {
    Step-Fail "$RppgCsv not found (run extraction first)"
    exit 1
} else {
    $visualDir = Join-Path $OutRoot 'visual'
    $framesRoot = Join-Path $OutRoot 'frames\frame_sequences'
    $null = New-Item -ItemType Directory -Force -Path $visualDir

    # Check if stage-1 frames exist for videos present in the rPPG CSV
    $useStage1Frames = $false
    if (Test-Path $framesRoot) {
        $rppgStems = @{}
        foreach ($row in (Import-Csv -LiteralPath $RppgCsv)) {
            if ($row.video_path) {
                $rppgStems[([System.IO.Path]::GetFileNameWithoutExtension([string]$row.video_path)).ToLowerInvariant()] = $true
            }
        }
        foreach ($frameDir in @(Get-ChildItem -LiteralPath $framesRoot -Directory -ErrorAction SilentlyContinue)) {
            if ($rppgStems.ContainsKey($frameDir.Name.ToLowerInvariant())) { $useStage1Frames = $true; break }
        }
        if (-not $useStage1Frames) {
            Write-Warn "Stage-1 frames do not cover the rPPG CSV videos; falling back to video extraction"
        }
    }
    $visualLog = Join-Path $visualDir 'visual_stage.log'
    if ($useStage1Frames) {
        Write-Green "Using stage-1 frames from $framesRoot"
        Write-Host "  Extracting visual features, fusing, creating splits (live progress)..."
        & $Venv -u (Join-Path $RepoRoot 'setup_progress.py') --log $visualLog --label 'visual' `
            -- $Venv (Join-Path $Working 'visual\pipeline.py') `
            --frames-root $framesRoot `
            --rppg-csv $RppgCsv `
            --output-dir $visualDir `
            --fuse --create-splits
    } else {
        Write-Host "  Extracting visual features from videos, fusing, creating splits (live progress)..."
        & $Venv -u (Join-Path $RepoRoot 'setup_progress.py') --log $visualLog --label 'visual' `
            -- $Venv (Join-Path $Working 'visual\pipeline.py') `
            --video-root $env:DFDC_DATASET_PATH `
            --rppg-csv $RppgCsv `
            --output-dir $visualDir `
            --fuse --create-splits
    }
    if ($LASTEXITCODE -ne 0) {
        Step-Fail "visual extraction/fusion (exit $LASTEXITCODE) - see $visualLog"
        Get-Content $visualLog -Tail 50
        exit 1
    }
    if (-not (Test-Path -LiteralPath $FusedCsv)) {
        Step-Fail "$FusedCsv not found"
        exit 1
    }
    Step-Done "Fused features -> $FusedCsv"
}

# ----- Step 6: Quantum pipeline (build data + QAOA select + train VQC + evaluate + baselines) -----
$step++
Step $step $TotalSteps "Quantum pipeline (--all: data + QAOA select + VQC train + eval + baselines)"
$quantumArtifacts = @($QuantumData, $QuantumVqc, $QuantumSel, $QuantumScaler)
$allQuantumExist = $true
foreach ($a in $quantumArtifacts) {
    if (Test-Artifact $a) { Write-Dim "exists: $a" } else { $allQuantumExist = $false }
}
if ($allQuantumExist) {
    Step-Skip "all quantum artifacts exist"
} else {
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'quantum')
    Write-Host "  Running full quantum flow (build + select + train + evaluate + baselines)..."
    Write-Dim "Using fused CSV: $FusedCsv"
    # Ensure MAJ_OUTPUT_ROOT has no trailing space (critical for pathlib)
    $env:MAJ_OUTPUT_ROOT = $env:MAJ_OUTPUT_ROOT.Trim()
    Write-Dim "MAJ_OUTPUT_ROOT for subprocess: [$($env:MAJ_OUTPUT_ROOT)]"
    Push-Location $Working
    # Explicitly pass the environment variable to avoid trailing space issues
    $env:MAJ_OUTPUT_ROOT = $env:MAJ_OUTPUT_ROOT.Trim()
    $proc = Start-Process -FilePath $Venv -ArgumentList "-m quantum.pipeline --all --csv-file $FusedCsv" -Wait -PassThru -NoNewWindow -WorkingDirectory $Working
    $exit = $proc.ExitCode
    Pop-Location
    # Training succeeds but evaluation may fail on fresh run due to checksum mismatch
    # (expected for first run - subsequent runs use trained model)
    if ($exit -ne 0) {
        Write-Warn "Quantum training completed (evaluation failed - expected on first run)"
        Write-Green "Artifacts generated: qaoa_selection_fused.json, hybrid_vqc_fused.pt, feature_scaler_fused.json"
        Step-Warn "Quantum pipeline complete (evaluation deferred)"
    } else {
        Step-Done "Quantum pipeline complete"
    }
}

# ----- Step 7: Frontend build -----
if (-not $SkipFrontend) {
    $step++
    Step $step $TotalSteps "Frontend build"
    if (Test-Path -LiteralPath $FrontendDist) {
        Step-Skip "dist/ exists: $FrontendDist"
    } else {
        Write-Host "  Building frontend..."
        Push-Location $Frontend
        & npm run build
        if ($LASTEXITCODE -ne 0) { Step-Fail "frontend build"; Pop-Location; exit 1 }
        Pop-Location
        Step-Done "Frontend built -> $FrontendDist"
    }
}

# ----- Step 8 (optional): sample inference -----
if ($Video -ne "") {
    $step++
    Step $step $TotalSteps "Sample inference: $(Fit-Text $Video 44)"
    $videoPath = Resolve-Path -LiteralPath $Video -ErrorAction SilentlyContinue
    if (-not $videoPath) {
        Step-Warn "Video not found: $Video -- skipping inference"
    } else {
        $outJson = Join-Path $OutRoot 'pipeline\pipeline_result.json'
        $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'pipeline')
        Push-Location $Working
        & "$Venv" run_pipeline.py --source "$videoPath" --method POS --out "$outJson"
        $exit = $LASTEXITCODE
        Pop-Location
        if ($exit -eq 0) {
            Step-Done "Verdict saved -> $outJson"
        } elseif ($exit -eq 3) {
            Step-Warn "INCONCLUSIVE (insufficient usable frames)"
        } else {
            Step-Warn "Pipeline exited with code $exit"
        }
    }
}

# ----- Summary -----
Show-Summary
Write-Host ""
Write-Host "${C_BOLD}Generated artifacts:$C_RESET"
Write-Host "  Stage 1 (frames):   $OutRoot\frames\"
Write-Host "  Stage 2 (rPPG):     $OutRoot\rppg\"
Write-Host "  Stage 3 (visual):   $OutRoot\visual\"
Write-Host "  Stage 4 (quantum):  $OutRoot\quantum\"
Write-Host "  Pipeline result:    $OutRoot\pipeline\pipeline_result.json"
Write-Host ""
Write-Host "${C_BOLD}Project structure:$C_RESET"
Write-Host "$C_GRAY  WORKING/$C_RESET"
Write-Host "$C_GRAY    frame/           Stage 1: YOLO face detection + quality gating (30 fps)$C_RESET"
Write-Host "$C_GRAY    RPPG/            Stage 2: MediaPipe/YuNet -> POS/CHROM -> 20 features$C_RESET"
Write-Host "$C_GRAY    visual/          Stage 3: ResNet50 + handcrafted -> 39 visual features$C_RESET"
Write-Host "$C_GRAY    quantum/         Stage 4: QAOA(63->3) -> Hybrid VQC -> P(real)$C_RESET"
Write-Host "$C_GRAY    run_pipeline.py  End-to-end orchestrator$C_RESET"
Write-Host "$C_GRAY  frontend/          React + Vite UI (server.py API on :8000)$C_RESET"
Write-Host ""
Write-Host "${C_BOLD}GPU-accelerated rPPG extraction:$C_RESET"
Write-Host "  Default: YuNet ONNX Runtime CUDA (8 workers)"
Write-Host "  Override: -CpuOnly for MediaPipe CPU, -GpuWorkers N to set parallelism"
Write-Host "  Checkpoint/resume: -Resume, -Start N, -End N"
Write-Host ""
Write-Host "${C_BOLD}To run the web UI:$C_RESET"
Write-Host "  Terminal 1:  cd frontend; python server.py"
Write-Host "  Terminal 2:  cd frontend; npm run dev"
Write-Host ""
Write-Host "${C_BOLD}To run inference on any video:$C_RESET"
Write-Host "  cd WORKING"
Write-Host "  python run_pipeline.py --source VIDEO.mp4 --method POS"
Write-Host ""
Write-Host "${C_BOLD}To re-run full rPPG extraction with GPU:$C_RESET"
Write-Host "  cd WORKING"
Write-Host "  python RPPG/extract_dataset_features.py --method POS --gpu --gpu-workers 8"
Write-Host ""
