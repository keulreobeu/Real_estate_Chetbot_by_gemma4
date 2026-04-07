param(
    [int]$CheckpointEvery = 10
)

$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\lwwde\miniconda3\envs\py312\python.exe"
$runScript = Join-Path $PSScriptRoot "run_generation_mvp.py"
$evalScript = Join-Path $PSScriptRoot "evaluate_generation_mvp.py"
$validateScript = Join-Path $PSScriptRoot "validate_generation_outputs.py"
$compareScript = Join-Path $PSScriptRoot "compare_generation_runs.py"

$edgeCsv = Join-Path $repo "data\eval\gemma4_generation_edge_predictions_gemma4_2b.csv"
$evalCsv = Join-Path $repo "data\eval\gemma4_generation_eval_predictions_gemma4_2b.csv"
$edgeMetrics = Join-Path $repo "data\eval\gemma4_generation_edge_metrics_gemma4_2b.json"
$evalMetrics = Join-Path $repo "data\eval\gemma4_generation_eval_metrics_gemma4_2b.json"

$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $logDir "pipeline_until_1400_$stamp.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logFile -Append
}

function Invoke-Py {
    param([string[]]$Args)
    & $py @Args 2>&1 | Tee-Object -FilePath $logFile -Append
    return $LASTEXITCODE
}

function Get-RowCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    $code = "import pandas as pd; p=r'$Path'; print(len(pd.read_csv(p, encoding='utf-8-sig')))"
    try {
        $value = & $py -c $code
        return [int]$value
    }
    catch {
        return 0
    }
}

Set-Location $repo
$cutoff = Get-Date -Hour 14 -Minute 0 -Second 0
Write-Log "Pipeline worker started. cutoff=$($cutoff.ToString('yyyy-MM-dd HH:mm:ss'))"

if ((Get-Date) -ge $cutoff) {
    Write-Log "Cutoff already passed. Exiting."
    exit 0
}

$stage04Done = $false

while ((Get-Date) -lt $cutoff) {
    $edgeRows = Get-RowCount -Path $edgeCsv
    $evalRows = Get-RowCount -Path $evalCsv
    Write-Log "Progress edge_rows=$edgeRows/2000 eval_rows=$evalRows/1000"

    if ($edgeRows -lt 2000) {
        Write-Log "Running edge generation resume loop."
        $code = Invoke-Py @(
            $runScript,
            "--mode", "edge",
            "--backend", "transformers",
            "--model", "gemma4_2b",
            "--offset", "0",
            "--resume",
            "--append",
            "--checkpoint-every", "$CheckpointEvery",
            "--startup-check"
        )
        Write-Log "edge generation exit_code=$code"
        if ($code -ne 0) { Start-Sleep -Seconds 60 }
        continue
    }

    if ($evalRows -lt 1000) {
        Write-Log "Running eval generation resume loop."
        $code = Invoke-Py @(
            $runScript,
            "--mode", "eval",
            "--backend", "transformers",
            "--model", "gemma4_2b",
            "--offset", "0",
            "--resume",
            "--append",
            "--checkpoint-every", "$CheckpointEvery",
            "--startup-check"
        )
        Write-Log "eval generation exit_code=$code"
        if ($code -ne 0) { Start-Sleep -Seconds 60 }
        continue
    }

    if (-not $stage04Done) {
        Write-Log "Stage 04 generation completeness reached. Running evaluation metrics."
        Invoke-Py @($evalScript, "--mode", "edge", "--model", "gemma4_2b") | Out-Null
        Invoke-Py @($evalScript, "--mode", "eval", "--model", "gemma4_2b") | Out-Null
        if (Test-Path $compareScript) {
            Invoke-Py @($compareScript, "--mode", "eval", "--left-model", "gemma4_2b", "--right-model", "gemma4_4b") | Out-Null
        }
        $stage04Done = $true
        Write-Log "Stage 04 finished."
    }

    Write-Log "Starting next tasks (post-04, non-GPU validation track)."
    Invoke-Py @($validateScript, "--mode", "edge", "--model", "gemma4_2b") | Out-Null
    Invoke-Py @($validateScript, "--mode", "eval", "--model", "gemma4_2b") | Out-Null
    Write-Log "Post-04 tasks executed. Worker exiting."
    break
}

Write-Log "Worker ended. edge_rows=$(Get-RowCount $edgeCsv) eval_rows=$(Get-RowCount $evalCsv) stage04Done=$stage04Done"

