param(
    [int]$CheckpointEvery = 25,
    [int]$LogEvery = 10,
    [int]$SampleSize = 20,
    [switch]$AbortOnGateFail
)

$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\lwwde\miniconda3\envs\py312\python.exe"
$runScript = Join-Path $PSScriptRoot "run_generation_mvp.py"
$validateScript = Join-Path $PSScriptRoot "validate_generation_outputs.py"
$benchmarkScript = Join-Path $PSScriptRoot "benchmark_edge_2b.py"
$targetCsv = Join-Path $repo "data\eval\gemma4_generation_edge_predictions_gemma4_2b.csv"
$heartbeatPath = Join-Path $repo "data\eval\gemma4_generation_edge_predictions_gemma4_2b.heartbeat.json"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$mainLog = Join-Path $logDir "edge_2b_until_1400_$stamp.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $mainLog -Append
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

function Get-LatestFilePath {
    param(
        [string]$Directory,
        [string]$Filter
    )
    if (-not (Test-Path $Directory)) {
        return ""
    }
    $item = Get-ChildItem -Path $Directory -File -Filter $Filter | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $item) {
        return ""
    }
    return $item.FullName
}

Set-Location $repo
$cutoff = Get-Date -Hour 14 -Minute 0 -Second 0
if ((Get-Date) -ge $cutoff) {
    Write-Log "Cutoff already passed for today. Exiting."
    exit 0
}

Write-Log "Worker started. cutoff=$($cutoff.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Log "Running benchmark gate before full loop."
& $py $benchmarkScript --sample-size $SampleSize 2>&1 | Tee-Object -FilePath $mainLog -Append
if ($LASTEXITCODE -ne 0) {
    if ($AbortOnGateFail) {
        Write-Log "RUN_STATE: BLOCKED_BY_GATE benchmark_failed=true"
        Write-Log "Benchmark gate failed (exit=$LASTEXITCODE). Abort full run."
        exit $LASTEXITCODE
    }
    Write-Log "RUN_STATE: GATE_FAILED_CONTINUE benchmark_failed=true"
    Write-Log "Benchmark gate failed (exit=$LASTEXITCODE). Continue anyway because AbortOnGateFail is not set."
}

$restartCount = 0
$consecutiveRunFailures = 0

while ((Get-Date) -lt $cutoff) {
    $rowsBefore = Get-RowCount -Path $targetCsv
    Write-Log "Current rows=$rowsBefore"
    if ($rowsBefore -ge 2000) {
        Write-Log "RUN_STATE: TARGET_REACHED"
        Write-Log "Target reached (rows >= 2000). Running finalize validation."
        & $py $validateScript --mode edge --model gemma4_2b 2>&1 | Tee-Object -FilePath $mainLog -Append
        break
    }

    Write-Log "RUN_STATE: RUNNING resume_chunk_start"
    Write-Log "Launching resume chunk run."
    & $py $runScript --mode edge --backend transformers --model gemma4_2b --offset 0 --resume --append --checkpoint-every $CheckpointEvery --log-every $LogEvery --no-startup-check --profile fast_edge --heartbeat-path $heartbeatPath 2>&1 | Tee-Object -FilePath $mainLog -Append
    $exitCode = $LASTEXITCODE
    $rowsAfter = Get-RowCount -Path $targetCsv
    $deltaRows = $rowsAfter - $rowsBefore
    Write-Log "Run exit_code=$exitCode delta_rows=$deltaRows rows_after=$rowsAfter"

    if ($exitCode -ne 0) {
        $restartCount += 1
        $consecutiveRunFailures += 1
        Write-Log "RUN_STATE: RUN_EXIT_NONZERO"
        Write-Log "Non-zero exit detected. Sleeping 60s then retry."
        if ($consecutiveRunFailures -ge 2) {
            Write-Log "AUTO_CHECKLIST: fast restart checklist triggered (2 consecutive non-zero exits)."
            Write-Log "1) Confirm --profile fast_edge and --no-startup-check."
            Write-Log "2) Confirm checkpoint/log are 25/10."
            Write-Log "3) Check GPU utilization and top CPU background processes."
            Write-Log "4) Resume via edge_runbook_2b.ps1 -Action resume -Execute."
        }
        Start-Sleep -Seconds 60
    }
    else {
        $consecutiveRunFailures = 0
        if ($deltaRows -le 0) {
            Write-Log "RUN_STATE: RUN_EXIT_OK_NO_PROGRESS"
        }
        else {
            Write-Log "RUN_STATE: RUN_EXIT_OK_PROGRESS"
        }
        Write-Log "Run completed without process error. Rechecking progress."
        Start-Sleep -Seconds 5
    }
}

$finalRows = Get-RowCount -Path $targetCsv
Write-Log "RUN_STATE: WORKER_FINISHED"
Write-Log "Worker finished. final_rows=$finalRows restart_count=$restartCount benchmark_json=$(Get-LatestFilePath -Directory $logDir -Filter 'edge_2b_benchmark_*.json')"
