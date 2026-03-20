$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "logs"
$logFile = Join-Path $logDir "hf_backend.log"
$errFile = Join-Path $logDir "hf_backend.err.log"

if (-not (Test-Path $python)) {
    throw "Python 虚拟环境不存在：$python"
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*uvicorn app.main:app*"
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HOME = Join-Path $env:LOCALAPPDATA "huggingface"
$env:HF_HUB_DISABLE_XET = "1"

Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $root `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -WindowStyle Hidden | Out-Null

Write-Host "HF Constraint backend started. Logs: $logFile"
