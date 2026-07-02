# install.ps1 — idempotent Windows installer for Quipu.
#
# Steps:
#   1. Resolve QUIPU_HOME (default %USERPROFILE%\.quipu) and venv path.
#   1b. Read saved model from $QUIPU_HOME\config.
#   2. Find repo root (directory of this script).
#   3. Create venv if absent.
#   4. Upgrade pip; pip install -e <repo_root>. onnxruntime wheel failure -> WARN + continue.
#   5. Ensure huggingface_hub in venv (best-effort).
#   6. Prompt model picker (interactive) or use saved/default; fetch chosen model if absent.
#   7. Validate: python -m quipu --version must succeed.
#   8. Print .mcp.json snippet.
#
# Re-run is safe: no re-download if model already present.
#
# Usage:
#   .\install.ps1
#   .\install.ps1 -QuipuHome C:\custom\quipu\home
#
param(
    [string]$QuipuHome = ""
)

# Step 1 — resolve QUIPU_HOME
if ([string]::IsNullOrEmpty($QuipuHome)) {
    $QuipuHome = if ($env:QUIPU_HOME) { $env:QUIPU_HOME } else { Join-Path $env:USERPROFILE ".quipu" }
}
$Venv = Join-Path $QuipuHome "venv"

# Step 1b — read saved model from config
$ConfigFile = Join-Path $QuipuHome "config"
$SavedModel = ""
if (Test-Path $ConfigFile) {
    $SavedModel = (Get-Content $ConfigFile | Where-Object { $_ -match '^MODEL=' } | Select-Object -First 1) -replace '^MODEL=', ''
}

# Model table
$ModelTable = @{
    1 = @{ Key = "nomic-embed-v2";        Repo = "nomic-ai/nomic-embed-v2" }
    2 = @{ Key = "nomic-embed-text-v1.5"; Repo = "nomic-ai/nomic-embed-text-v1.5" }
    3 = @{ Key = "bge-small-en-v1.5";     Repo = "BAAI/bge-small-en-v1.5" }
    4 = @{ Key = "bge-m3";                Repo = "BAAI/bge-m3" }
    5 = @{ Key = "embeddinggemma-300m";   Repo = "google/embeddinggemma-300m" }
}

$DefaultNum = 1
if ($SavedModel) {
    foreach ($n in $ModelTable.Keys) {
        if ($ModelTable[$n].Key -eq $SavedModel) { $DefaultNum = $n; break }
    }
}

function Resolve-ModelChoice([string]$raw, [int]$defaultNum) {
    $n = 0
    if ([string]::IsNullOrEmpty($raw)) { $n = $defaultNum }
    elseif ([int]::TryParse($raw.Trim(), [ref]$n)) { } else { $n = -1 }
    if ($n -ge 1 -and $n -le 5) { return $n }
    return -1
}

$ChosenModel = ""
$ChosenHfRepo = ""

if ([System.Console]::IsInputRedirected) {
    # Non-interactive — use saved or default
    if ($SavedModel -and ($ModelTable.Values | Where-Object { $_.Key -eq $SavedModel })) {
        $ChosenModel   = $SavedModel
        $ChosenHfRepo  = ($ModelTable.Values | Where-Object { $_.Key -eq $SavedModel } | Select-Object -First 1).Repo
    } else {
        $ChosenModel  = $ModelTable[$DefaultNum].Key
        $ChosenHfRepo = $ModelTable[$DefaultNum].Repo
    }
    Write-Host "Using model: $ChosenModel"
} else {
    # Interactive — show picker
    Write-Host ""
    Write-Host "==> Select embedding model:"
    Write-Host "    1) nomic-embed-text-v1.5   (nomic-ai/nomic-embed-text-v1.5)  [recommended]"
    Write-Host "    2) nomic-embed-text-v1.5   (nomic-ai/nomic-embed-text-v1.5)"
    Write-Host "    3) bge-small-en-v1.5       (BAAI/bge-small-en-v1.5)"
    Write-Host "    4) bge-m3                  (BAAI/bge-m3)"
    Write-Host "    5) embeddinggemma-300m     (google/embeddinggemma-300m)       [gated]"
    Write-Host ""
    if ($SavedModel) { Write-Host "    Current saved model: $SavedModel" }

    $raw1 = Read-Host "    Enter number [default: $DefaultNum]"
    $choice = Resolve-ModelChoice $raw1 $DefaultNum
    if ($choice -eq -1) {
        Write-Host "    Invalid choice '$raw1'. Please enter a number 1-5."
        $raw2 = Read-Host "    Enter number [default: $DefaultNum]"
        $choice = Resolve-ModelChoice $raw2 $DefaultNum
        if ($choice -eq -1) { $choice = $DefaultNum }
    }

    $ChosenModel  = $ModelTable[$choice].Key
    $ChosenHfRepo = $ModelTable[$choice].Repo
}

$ModelDir = Join-Path $QuipuHome "models\$ChosenModel"

# Persist chosen model to config (idempotent, atomic)
$null = New-Item -ItemType Directory -Force -Path $QuipuHome
if (Test-Path $ConfigFile) {
    $lines = Get-Content $ConfigFile | Where-Object { $_ -notmatch '^MODEL=' }
    $tmpFile = "$ConfigFile.tmp"
    $lines | Set-Content $tmpFile
    "MODEL=$ChosenModel" | Add-Content $tmpFile
    Move-Item -Force $tmpFile $ConfigFile
} else {
    "MODEL=$ChosenModel" | Set-Content $ConfigFile
}

# Step 2 — repo root = directory of this script
$RepoRoot = $PSScriptRoot

Write-Host "==> Quipu install"
Write-Host "    QUIPU_HOME : $QuipuHome"
Write-Host "    VENV       : $Venv"
Write-Host "    REPO_ROOT  : $RepoRoot"

# Step 3 — create venv if absent
if (Test-Path $Venv) {
    Write-Host "==> Venv already present at $Venv"
} else {
    Write-Host "==> Creating venv at $Venv"
    $null = New-Item -ItemType Directory -Force -Path $QuipuHome
    & python -m venv "$Venv"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: failed to create venv at $Venv. Is Python 3.10+ installed and on PATH?"
        exit 1
    }
}

$PY = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $PY)) {
    Write-Error "ERROR: cannot find python at $PY"
    exit 1
}

# Step 4 — upgrade pip and install quipu (editable)
Write-Host "==> Upgrading pip"
& $PY -m pip install --upgrade pip

Write-Host "==> Installing quipu (editable) from $RepoRoot"
# onnxruntime may fail on newer Python versions; that is acceptable.
& $PY -m pip install -e "$RepoRoot"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "WARNING: pip install returned non-zero (likely onnxruntime wheel unavailable)."
    Write-Warning "WARNING: Installing non-onnxruntime deps (mcp, tokenizers, numpy) explicitly, then quipu with --no-deps."
    & $PY -m pip install mcp tokenizers numpy
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "WARNING: runtime deps install failed; CLI may be broken."
    }
    & $PY -m pip install -e "$RepoRoot" --no-deps
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "WARNING: quipu editable install failed even with --no-deps; continuing (CLI will be broken)."
    }
}

# Step 5 — ensure huggingface_hub in venv (best-effort)
Write-Host "==> Checking huggingface_hub"
& $PY -c "import huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Installing huggingface_hub into venv"
    & $PY -m pip install huggingface_hub
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "WARNING: huggingface_hub install failed; model fetch may not work."
    }
} else {
    Write-Host "    huggingface_hub already present"
}

# Step 6 — fetch chosen model if absent
$ModelOnnx = Join-Path $ModelDir "model.onnx"
if (Test-Path $ModelOnnx) {
    Write-Host "==> Model already present at $ModelDir (skipping download)"
} else {
    Write-Host "==> Fetching $ChosenModel to $ModelDir"
    $null = New-Item -ItemType Directory -Force -Path $ModelDir

    $downloadFailed = $false
    & $PY -c "import huggingface_hub" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "WARNING: huggingface_hub not installed; model download skipped."
        Write-Host "  Install manually: pip install huggingface_hub"
        Write-Host "  Then run: hf download $ChosenHfRepo --local-dir `"$ModelDir`""
        $downloadFailed = $true
    } else {
        try {
            Write-Host "    Downloading from Hugging Face ($ChosenHfRepo) ..."
            & $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$ChosenHfRepo', local_dir=r'$ModelDir', local_dir_use_symlinks=False, resume_download=True)"
            if ($LASTEXITCODE -ne 0) { $downloadFailed = $true }
        } catch {
            $downloadFailed = $true
        }
    }

    if ($downloadFailed) {
        Write-Host ""
        if ($ChosenModel -eq "embeddinggemma-300m") {
            Write-Warning "WARNING: model download failed (likely gated - see below)."
            Write-Host ""
            Write-Host "  EmbeddingGemma-300m is a gated model and requires:" -ForegroundColor Yellow
            Write-Host "  1. Accept the license at https://huggingface.co/google/embeddinggemma-300m" -ForegroundColor Yellow
            Write-Host "  2. Run: hf auth login   (token from https://huggingface.co/settings/tokens)" -ForegroundColor Yellow
            Write-Host "  3. Re-run this installer." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  To fetch manually after login:" -ForegroundColor Yellow
            Write-Host "  hf download google/embeddinggemma-300m --local-dir `"$ModelDir`"" -ForegroundColor Yellow
            Write-Host ""
        } else {
            Write-Warning "WARNING: model download failed."
            Write-Host "  Check your internet connection and retry the installer." -ForegroundColor Yellow
            Write-Host "  To fetch manually: hf download $ChosenHfRepo --local-dir `"$ModelDir`"" -ForegroundColor Yellow
            Write-Host ""
        }
        exit 1
    }
}

# Step 7 — validate
Write-Host "==> Validating install"
& $PY -m quipu --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: quipu --version failed after install"
    exit 1
}

# Step 8 — print .mcp.json snippet
$AbsVenvPy = (Resolve-Path $PY).Path
$AbsVenvPyJson = $AbsVenvPy -replace '\\','/'

Write-Host ""
Write-Host "==> Install complete."
Write-Host ""
Write-Host "Add this to your .mcp.json to register Quipu with an MCP client:"
Write-Host ""
$Snippet = @"
{
  "mcpServers": {
    "quipu": {
      "command": "$AbsVenvPyJson",
      "args": ["-m", "quipu", "serve"],
      "env": {
        "QUIPU_MODE": "project",
        "QUIPU_PROJECT_ROOT": "<your-project-root>",
        "QUIPU_EMBEDDING_MODEL": "$ChosenModel"
      }
    }
  }
}
"@
Write-Host $Snippet
Write-Host ""
Write-Host "For global mode, omit QUIPU_PROJECT_ROOT and set QUIPU_MODE=global."
Write-Host "Run ``$AbsVenvPy -m quipu init`` inside your project to initialise the store."
