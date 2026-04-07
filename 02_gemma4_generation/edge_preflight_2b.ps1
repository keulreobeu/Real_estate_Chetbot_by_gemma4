param(
    [int]$CheckpointEvery = 25,
    [int]$LogEvery = 10
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ModelConfig = Join-Path $PSScriptRoot "config\models.local.json"
$OutputCsv = Join-Path $RepoRoot "data\eval\gemma4_generation_edge_predictions_gemma4_2b.csv"
$PythonExe = "C:\Users\lwwde\miniconda3\envs\py312\python.exe"

Write-Host "=== 5-minute preflight (edge 2b) ==="

if (-not (Test-Path $ModelConfig)) {
    throw "models.local.json not found: $ModelConfig"
}

$inspectCode = @"
import json
from pathlib import Path
p = Path(r'$ModelConfig')
data = json.loads(p.read_text(encoding='utf-8'))
model = data.get('models', {}).get('gemma4_2b')
if not isinstance(model, dict):
    raise SystemExit('missing_model')
runtime = str(model.get('runtime', ''))
device_map = model.get('device_map', '')
max_output_tokens = int(model.get('max_output_tokens', 0) or 0)
print(runtime)
print(json.dumps(device_map, ensure_ascii=False))
print(max_output_tokens)
"@
$inspectOut = & $PythonExe -c $inspectCode
if ($LASTEXITCODE -ne 0 -or $inspectOut.Count -lt 3) {
    throw "Failed to parse gemma4_2b fields from models.local.json"
}

$runtime = [string]$inspectOut[0]
$deviceMap = [string]$inspectOut[1]
$maxOutputTokens = [int]$inspectOut[2]
Write-Host "runtime=$runtime"
Write-Host "device_map=$deviceMap"
Write-Host "max_output_tokens=$maxOutputTokens"

if ($runtime -ne "transformers") {
    Write-Warning "Expected runtime=transformers for edge runbook."
}
if ($CheckpointEvery -lt 20 -or $CheckpointEvery -gt 50) {
    Write-Warning "Recommended checkpoint range is 20~50. current=$CheckpointEvery"
}
if ($LogEvery -le 0) {
    Write-Warning "log-every should be positive. current=$LogEvery"
}

Write-Host ""
Write-Host "[Power profile]"
powercfg /GETACTIVESCHEME

Write-Host ""
Write-Host "[Background heavy processes (top CPU)]"
Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 ProcessName,Id,CPU

Write-Host ""
Write-Host "[GPU snapshot]"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,pstate --format=csv,noheader

Write-Host ""
if (Test-Path $OutputCsv) {
    $item = Get-Item $OutputCsv
    Write-Host "existing_output=$($item.FullName)"
    Write-Host "last_write=$($item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "size_bytes=$($item.Length)"
} else {
    Write-Host "existing_output=none"
}

Write-Host ""
Write-Host "Preflight complete."
