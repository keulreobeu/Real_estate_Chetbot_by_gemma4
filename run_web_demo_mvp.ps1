param(
    [ValidateSet("transformers", "mock", "llama_cpp")]
    [string]$Backend = "transformers",
    [string]$Model = "gemma4_2b",
    [string]$Host = "0.0.0.0",
    [int]$Port = 8787,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\lwwde\miniconda3\envs\py312\python.exe"
$ScriptPath = Join-Path $RepoRoot "02_gemma4_generation\demo_chatbot_web_mvp.py"
$LocalUrl = "http://127.0.0.1`:$Port"
$Url = if ($Host -eq "0.0.0.0") { $LocalUrl } else { "http://$Host`:$Port" }

if (-not (Test-Path $ScriptPath)) {
    throw "demo_chatbot_web_mvp.py not found: $ScriptPath"
}

Write-Host "Starting Web Demo MVP..." -ForegroundColor Cyan
Write-Host "backend=$Backend model=$Model url=$Url"
Write-Host "Stop server: Ctrl + C"
if ($Host -eq "0.0.0.0") {
    $ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.PrefixOrigin -ne "WellKnown" } |
        Select-Object -ExpandProperty IPAddress -Unique
    if ($ips) {
        Write-Host "LAN access URLs:"
        foreach ($ip in $ips) {
            Write-Host (" - http://{0}:{1}" -f $ip, $Port)
        }
    }
}

if ($OpenBrowser) {
    Start-Process $Url | Out-Null
}

& $PythonExe $ScriptPath --backend $Backend --model $Model --host $Host --port $Port
