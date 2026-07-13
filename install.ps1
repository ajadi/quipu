# install.ps1 - idempotent Windows installer for Quipu.
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

# Step 1 - resolve QUIPU_HOME
if ([string]::IsNullOrEmpty($QuipuHome)) {
    $QuipuHome = if ($env:QUIPU_HOME) { $env:QUIPU_HOME } else { Join-Path $env:USERPROFILE ".quipu" }
}
$Venv = Join-Path $QuipuHome "venv"

# Step 1b - read saved model from config
$ConfigFile = Join-Path $QuipuHome "config"
$SavedModel = ""
if (Test-Path $ConfigFile) {
    $SavedModel = (Get-Content $ConfigFile | Where-Object { $_ -match '^MODEL=' } | Select-Object -First 1) -replace '^MODEL=', ''
}

# Model table
$ModelTable = @{
    1 = @{ Key = "nomic-embed-text-v1.5"; Repo = "nomic-ai/nomic-embed-text-v1.5" }
    2 = @{ Key = "bge-small-en-v1.5";     Repo = "BAAI/bge-small-en-v1.5" }
    3 = @{ Key = "bge-m3";                Repo = "BAAI/bge-m3" }
    4 = @{ Key = "embeddinggemma-300m";   Repo = "google/embeddinggemma-300m" }
}

function Resolve-ModelChoice([string]$raw) {
    # Returns 1-4 for a valid numeric choice, "none" for keyword-only
    # (case-insensitive), or $null for empty/invalid input (caller re-prompts
    # - no default is auto-accepted).
    if ([string]::IsNullOrEmpty($raw)) { return $null }
    if ($raw.Trim().ToLowerInvariant() -eq "none") { return "none" }
    $n = 0
    if ([int]::TryParse($raw.Trim(), [ref]$n) -and $n -ge 1 -and $n -le 4) { return $n }
    return $null
}

$ChosenModel = ""
$ChosenHfRepo = ""

if ([System.Console]::IsInputRedirected) {
    # Non-interactive / piped stdin - honor a previously saved choice only.
    # Never silently substitute a specific model when unset/unrecognized:
    # resolve cleanly to keyword-only mode instead (informational, exit 0).
    if ($SavedModel -and ($ModelTable.Values | Where-Object { $_.Key -eq $SavedModel })) {
        $ChosenModel   = $SavedModel
        $ChosenHfRepo  = ($ModelTable.Values | Where-Object { $_.Key -eq $SavedModel } | Select-Object -First 1).Repo
    } else {
        $ChosenModel  = "none"
        $ChosenHfRepo = ""
    }

    if ($ChosenModel -eq "none") {
        Write-Host "==> No embedding model configured - running in keyword-only mode (QUIPU_EMBEDDING_MODEL=none)."
        Write-Host "    To use semantic search instead, set QUIPU_EMBEDDING_MODEL=<key> (e.g. nomic-embed-text-v1.5) before installing."
    } else {
        Write-Host "Using model: $ChosenModel"
    }
} else {
    # Interactive - show picker, loop until a valid choice. Empty/invalid
    # input re-prompts; there is no auto-accepted default. 'none'
    # (keyword-only) is always a valid answer and resolves the loop
    # immediately.
    Write-Host ""
    Write-Host "==> Select embedding model:"
    Write-Host "    1) nomic-embed-text-v1.5   (nomic-ai/nomic-embed-text-v1.5)  [recommended]"
    Write-Host "       dim=768, ~270MB, English-focused, balanced quality/speed, open"
    Write-Host "    2) bge-small-en-v1.5       (BAAI/bge-small-en-v1.5)"
    Write-Host "       dim=384, ~130MB, English-only, fastest/smallest, open"
    Write-Host "    3) bge-m3                  (BAAI/bge-m3)"
    Write-Host "       dim=1024, ~2.2GB, multilingual, highest quality/slower, open"
    Write-Host "    4) embeddinggemma-300m     (google/embeddinggemma-300m)       [gated]"
    Write-Host "       dim=768, ~300MB, multilingual, high quality, GATED (HF login)"
    Write-Host "    none) keyword-only BM25 - no download, reduced semantic recall"
    Write-Host ""
    if ($SavedModel) { Write-Host "    Current saved model: $SavedModel" }

    $choice = $null
    while ($null -eq $choice) {
        $raw = Read-Host "    Enter number (1-4), or 'none' for keyword-only (no embedding model)"
        $choice = Resolve-ModelChoice $raw
        if ($null -eq $choice) {
            Write-Host "    Invalid choice '$raw'. Please enter a number 1-4, or 'none'."
        }
    }

    if ($choice -eq "none") {
        $ChosenModel  = "none"
        $ChosenHfRepo = ""
    } else {
        $ChosenModel  = $ModelTable[$choice].Key
        $ChosenHfRepo = $ModelTable[$choice].Repo
    }
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

# Test hook - used by tests/scripts/test_install_model_select.py to exercise
# the model-selection logic above without running the full install (venv/pip/
# network). Not used by real installs.
if ($env:QUIPU_TEST_MODEL_SELECT_ONLY -eq "1") {
    Write-Host "CHOSEN_MODEL=$ChosenModel"
    exit 0
}

# Step 2 - repo root = directory of this script
$RepoRoot = $PSScriptRoot

Write-Host "==> Quipu install"
Write-Host "    QUIPU_HOME : $QuipuHome"
Write-Host "    VENV       : $Venv"
Write-Host "    REPO_ROOT  : $RepoRoot"

# Step 3 - create venv if absent
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

# Step 4 - upgrade pip and install quipu (editable)
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

# Step 5 - ensure huggingface_hub in venv (best-effort)
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

# Step 6 - fetch chosen model if absent
$ModelOnnx = Join-Path $ModelDir "model.onnx"
if ($ChosenModel -eq "none") {
    Write-Host "==> Keyword-only mode (QUIPU_EMBEDDING_MODEL=none) - no model to download."
} elseif (Test-Path $ModelOnnx) {
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

# Step 7 - validate
Write-Host "==> Validating install"
& $PY -m quipu --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: quipu --version failed after install"
    exit 1
}

# Step 8 - print .mcp.json snippet
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
