param(
    [ValidateSet("print", "precheck", "benchmark", "monitor", "status", "start", "resume", "stop", "finalize")]
    [string]$Action = "print",
    [int]$Offset = 0,
    [int]$Limit = 0,
    [int]$CheckpointEvery = 25,
    [int]$LogEvery = 10,
    [int]$SampleSize = 20,
    [int]$MonitorIntervalMinutes = 10,
    [int]$TargetRows = 2000,
    [switch]$FastProfile,
    [switch]$NoStartupCheck,
    [switch]$SkipBenchmark,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lwwde\miniconda3\envs\py312\python.exe"
$RunScript = Join-Path $PSScriptRoot "run_generation_mvp.py"
$ValidateScript = Join-Path $PSScriptRoot "validate_generation_outputs.py"
$VerifyScript = Join-Path $PSScriptRoot "verify_local_inference_setup.py"
$EvalScript = Join-Path $PSScriptRoot "evaluate_generation_mvp.py"
$PreflightScript = Join-Path $PSScriptRoot "edge_preflight_2b.ps1"
$BenchmarkScript = Join-Path $PSScriptRoot "benchmark_edge_2b.py"
$MonitorScript = Join-Path $PSScriptRoot "monitor_edge_progress.py"
$OutputCsv = Join-Path $RepoRoot "data\eval\gemma4_generation_edge_predictions_gemma4_2b.csv"
$StopSignalPath = Join-Path $RepoRoot "data\eval\gemma4_generation_edge_predictions_gemma4_2b.stop"
$HeartbeatPath = Join-Path $RepoRoot "data\eval\gemma4_generation_edge_predictions_gemma4_2b.heartbeat.json"
$LogsDir = Join-Path $RepoRoot "logs"

function Invoke-Py {
    param([string[]]$PyArgs)
    & $PythonExe @PyArgs
}

function Get-RowCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    $code = "import pandas as pd; p=r'$Path'; print(len(pd.read_csv(p, encoding='utf-8-sig')))"
    $value = & $PythonExe -c $code
    return [int]$value
}

function Get-MaxSourceRowIndex {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return ""
    }
    $code = @"
import pandas as pd
p = r'$Path'
df = pd.read_csv(p, encoding='utf-8-sig')
if 'source_row_index' not in df.columns:
    print('')
else:
    s = pd.to_numeric(df['source_row_index'], errors='coerce').dropna()
    print('' if s.empty else int(s.max()))
"@
    $value = & $PythonExe -c $code
    return "$value".Trim()
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

function Show-Runbook {
    Write-Host "=== Edge Runbook (gemma4_2b) ==="
    Write-Host "1) Precheck  : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action precheck"
    Write-Host "2) Benchmark : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action benchmark -SampleSize 20 -Execute"
    Write-Host "3) Start     : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action start -Offset 0 -Limit 0 -CheckpointEvery 25 -LogEvery 10 -FastProfile -NoStartupCheck -Execute"
    Write-Host "4) Resume    : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action resume -Offset 0 -Limit 0 -CheckpointEvery 25 -LogEvery 10 -FastProfile -NoStartupCheck -Execute"
    Write-Host "5) Stop      : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action stop -Execute"
    Write-Host "6) Monitor   : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action monitor -MonitorIntervalMinutes 10 -TargetRows 2000 -Execute"
    Write-Host "7) Status    : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action status"
    Write-Host "8) Finalize  : .\02_gemma4_generation\edge_runbook_2b.ps1 -Action finalize -Execute"
    Write-Host ""
    Write-Host "Safety default: benchmark/start/resume/monitor/finalize prints commands only unless -Execute is set."
    Write-Host "Limit rule: -Limit 0 means process all remaining rows."
    Write-Host "Status now includes detector verdict, heartbeat freshness, benchmark state, and remaining rows."
}

function Do-Precheck {
    Set-Location $RepoRoot
    Invoke-Py @($VerifyScript, "--model", "gemma4_2b")
    & $PreflightScript -CheckpointEvery $CheckpointEvery -LogEvery $LogEvery
    Invoke-Py @($ValidateScript, "--mode", "edge", "--model", "gemma4_2b")
    nvidia-smi
}

function Do-Benchmark {
    Set-Location $RepoRoot
    $args = @($BenchmarkScript, "--sample-size", "$SampleSize")
    Write-Host "Command preview:"
    Write-Host ("& '$PythonExe' " + ($args -join " "))
    if ($Execute) {
        Invoke-Py $args
    }
}

function Do-Monitor {
    Set-Location $RepoRoot
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $monitorLog = Join-Path $RepoRoot "logs\edge_monitor_$stamp.log"
    $latestMainLog = Get-LatestFilePath -Directory $LogsDir -Filter "edge_2b_*.log"
    $latestBenchmarkJson = Get-LatestFilePath -Directory $LogsDir -Filter "edge_2b_benchmark_*.json"
    $args = @(
        $MonitorScript,
        "--output-csv", $OutputCsv,
        "--target-rows", "$TargetRows",
        "--interval-minutes", "$MonitorIntervalMinutes",
        "--main-log-path", $latestMainLog,
        "--heartbeat-path", $HeartbeatPath,
        "--benchmark-json-path", $latestBenchmarkJson,
        "--log-path", $monitorLog
    )
    Write-Host "Command preview:"
    Write-Host ("& '$PythonExe' " + ($args -join " "))
    if ($Execute) {
        Invoke-Py $args
    }
}

function Do-Status {
    Set-Location $RepoRoot
    $exists = Test-Path $OutputCsv
    $latestMainLog = Get-LatestFilePath -Directory $LogsDir -Filter "edge_2b_*.log"
    $latestBenchmarkJson = Get-LatestFilePath -Directory $LogsDir -Filter "edge_2b_benchmark_*.json"
    Write-Host "output_exists=$exists"
    if ($exists) {
        $item = Get-Item $OutputCsv
        $rows = Get-RowCount -Path $OutputCsv
        $maxSourceRowIndex = Get-MaxSourceRowIndex -Path $OutputCsv
        $remainingRows = [Math]::Max($TargetRows - $rows, 0)
        Write-Host ("output_path=" + $item.FullName)
        Write-Host ("length=" + $item.Length)
        Write-Host ("last_write=" + $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
        Write-Host ("rows=" + $rows)
        Write-Host ("max_source_row_index=" + $maxSourceRowIndex)
        Write-Host ("remaining_rows=" + $remainingRows)
        Write-Host ("heartbeat_path=" + $HeartbeatPath)
        Write-Host ("main_log_path=" + $latestMainLog)
        Write-Host ("benchmark_json_path=" + $latestBenchmarkJson)
        Write-Host ("stop_signal_exists=" + (Test-Path $StopSignalPath))
        Get-Content $OutputCsv | Select-Object -Last 2
    }
    else {
        Write-Host ("heartbeat_path=" + $HeartbeatPath)
        Write-Host ("main_log_path=" + $latestMainLog)
        Write-Host ("benchmark_json_path=" + $latestBenchmarkJson)
        Write-Host ("stop_signal_exists=" + (Test-Path $StopSignalPath))
    }
    $statusArgs = @(
        $MonitorScript,
        "--output-csv", $OutputCsv,
        "--target-rows", "$TargetRows",
        "--main-log-path", $latestMainLog,
        "--heartbeat-path", $HeartbeatPath,
        "--benchmark-json-path", $latestBenchmarkJson,
        "--one-shot",
        "--output-format", "text"
    )
    & $PythonExe @statusArgs
}

function Build-GenerationArgs {
    param([switch]$ResumeMode)
    $args = @(
        $RunScript,
        "--mode", "edge",
        "--backend", "transformers",
        "--model", "gemma4_2b",
        "--offset", "$Offset",
        "--checkpoint-every", "$CheckpointEvery",
        "--log-every", "$LogEvery",
        "--stop-signal-path", "$StopSignalPath",
        "--heartbeat-path", "$HeartbeatPath"
    )

    if ($Limit -gt 0) {
        $args += @("--limit", "$Limit")
    }

    $args += "--no-startup-check"

    if ($ResumeMode) {
        $args += @("--resume", "--append")
    }

    $args += @("--profile", "fast_edge")

    return ,$args
}

function Do-Start {
    Set-Location $RepoRoot
    if (-not $SkipBenchmark) {
        Do-Benchmark
        if (-not $Execute) {
            Write-Host "Benchmark gate preview completed. Run with -Execute to enforce gate."
        }
    }
    $args = Build-GenerationArgs
    Write-Host "Command preview:"
    Write-Host ("& '$PythonExe' " + ($args -join " "))
    if ($Execute) {
        Invoke-Py $args
    }
}

function Do-Resume {
    Set-Location $RepoRoot
    if (-not $SkipBenchmark) {
        Do-Benchmark
        if (-not $Execute) {
            Write-Host "Benchmark gate preview completed. Run with -Execute to enforce gate."
        }
    }
    $args = Build-GenerationArgs -ResumeMode
    Write-Host "Command preview:"
    Write-Host ("& '$PythonExe' " + ($args -join " "))
    if ($Execute) {
        Invoke-Py $args
    }
}

function Do-Finalize {
    Set-Location $RepoRoot
    $commands = @(
        @($ValidateScript, "--mode", "edge", "--model", "gemma4_2b"),
        @($EvalScript, "--mode", "edge", "--model", "gemma4_2b")
    )
    Write-Host "Command preview:"
    Write-Host ("& '$PythonExe' " + ($commands[0] -join " "))
    Write-Host ("& '$PythonExe' " + ($commands[1] -join " "))
    if ($Execute) {
        Invoke-Py $commands[0]
        Invoke-Py $commands[1]
    }
}

function Do-Stop {
    Write-Host "Command preview:"
    Write-Host ("New-Item -ItemType File -Path '" + $StopSignalPath + "' -Force")
    if ($Execute) {
        New-Item -ItemType File -Path $StopSignalPath -Force | Out-Null
        Write-Host ("Stop signal created: " + $StopSignalPath)
    }
}

switch ($Action) {
    "print" { Show-Runbook; break }
    "precheck" { Do-Precheck; break }
    "benchmark" { Do-Benchmark; break }
    "monitor" { Do-Monitor; break }
    "status" { Do-Status; break }
    "start" { Do-Start; break }
    "resume" { Do-Resume; break }
    "stop" { Do-Stop; break }
    "finalize" { Do-Finalize; break }
}
