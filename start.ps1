$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "尚未创建虚拟环境，请先按照 README.md 完成一次安装。"
}

$env:HF_HOME = Join-Path $ProjectDir ".cache\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"
& $PythonExe (Join-Path $ProjectDir "app.py")
