$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment not found. Follow README.md to install dependencies first."
}

$env:HF_HOME = Join-Path $ProjectDir ".cache\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"
& $PythonExe (Join-Path $ProjectDir "app.py")
