param(
    [string]$ModelId = "Qwen/Qwen3.5-4B-Base",
    [string]$ModelDir = "",
    [int]$MaxSeedRows = 250000,
    [int]$MaxExamples = 300000,
    [switch]$SkipModelDownload,
    [switch]$Train
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python 虚拟环境不存在：$python"
}

if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    if (Test-Path "D:\models") {
        $ModelDir = "D:\models\" + ($ModelId.Split("/")[-1])
    } else {
        $ModelDir = Join-Path $env:USERPROFILE ("models\" + ($ModelId.Split("/")[-1]))
    }
}

Push-Location $root
try {
    & $python -m pip install -r ".\requirements.txt"

    if (-not $SkipModelDownload) {
        & $python ".\scripts\download_model.py" --model-id $ModelId --local-dir $ModelDir
    }

    & $python ".\scripts\download_seed_data.py" --dataset-id "Duyu/Pinyin-Hanzi" --max-rows $MaxSeedRows
    & $python ".\scripts\prepare_training_data.py" --max-examples $MaxExamples
    & $python ".\scripts\export_feedback_dataset.py"

    if ($Train) {
        & $python ".\scripts\train_qlora.py"
    } else {
        Write-Host "数据已准备完毕。需要启动 QLoRA 训练时，请追加 -Train。"
    }
}
finally {
    Pop-Location
}
