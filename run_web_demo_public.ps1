param(
    [ValidateSet("transformers", "mock", "llama_cpp")]
    [string]$Backend = "transformers",
    [string]$Model = "gemma4_2b",
    [int]$Port = 8787,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\lwwde\miniconda3\envs\py312\python.exe"
$AppScript = Join-Path $RepoRoot "02_gemma4_generation\demo_chatbot_web_mvp.py"
$TunnelExe = Join-Path $RepoRoot "tools\cloudflared.exe"
$LogsDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

if (-not (Test-Path $AppScript)) {
    throw "Web demo script not found: $AppScript"
}
if (-not (Test-Path $TunnelExe)) {
    throw "cloudflared not found: $TunnelExe"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$appLog = Join-Path $LogsDir "web_demo_public_app_$stamp.log"
$tunnelLog = Join-Path $LogsDir "web_demo_public_tunnel_$stamp.log"

Write-Host "Starting web demo..." -ForegroundColor Cyan
$appCmd = "& '$PythonExe' '$AppScript' --backend $Backend --model $Model --host 127.0.0.1 --port $Port *> '$appLog'"
$appProc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-ExecutionPolicy", "Bypass", "-Command", $appCmd) -PassThru

Start-Sleep -Seconds 2
if (-not (Get-Process -Id $appProc.Id -ErrorAction SilentlyContinue)) {
    throw "Web demo process failed to start. Check log: $appLog"
}

Write-Host "Starting public tunnel..." -ForegroundColor Cyan
$tunnelCmd = "& '$TunnelExe' tunnel --no-autoupdate --url http://127.0.0.1:$Port *> '$tunnelLog'"
$tunnelProc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-ExecutionPolicy", "Bypass", "-Command", $tunnelCmd) -PassThru

$publicUrl = $null
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline -and -not $publicUrl) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
        $content = Get-Content -Path $tunnelLog -Raw -ErrorAction SilentlyContinue
        if ($content -match "https://[a-z0-9-]+\.trycloudflare\.com") {
            $publicUrl = $matches[0]
            break
        }
    }
    if (-not (Get-Process -Id $tunnelProc.Id -ErrorAction SilentlyContinue)) {
        break
    }
}

Write-Host ""
Write-Host "Local URL : http://127.0.0.1:$Port" -ForegroundColor Green
if ($publicUrl) {
    Write-Host "Public URL: $publicUrl" -ForegroundColor Green
    Write-Host "Share this Public URL from any network."
    if ($OpenBrowser) {
        Start-Process $publicUrl | Out-Null
    }
}
else {
    Write-Warning "Public URL could not be detected. Check tunnel log: $tunnelLog"
}

Write-Host ""
Write-Host "Running processes:"
Write-Host " - app PID: $($appProc.Id)"
Write-Host " - tunnel PID: $($tunnelProc.Id)"
Write-Host ""
Write-Host "Press Enter to stop both processes..."
[void](Read-Host)

foreach ($procId in @($tunnelProc.Id, $appProc.Id)) {
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
        Stop-Process -Id $procId -ErrorAction SilentlyContinue
    }
}

Write-Host "Stopped."
Write-Host "Logs:"
Write-Host " - $appLog"
Write-Host " - $tunnelLog"
