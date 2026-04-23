$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
        $name, $value = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($name, $value)
        Set-Item -Path "Env:$name" -Value $value
    }
} else {
    Write-Host "Error: .env file not found"
    exit 1
}

$hostArg = if ($args.Count -ge 1) { $args[0] } else { "127.0.0.1" }
$portArg = if ($args.Count -ge 2) { $args[1] } else { "8080" }

Write-Host ""
Write-Host "SWE-Vision Web App"
Write-Host "Host: $hostArg"
Write-Host "Port: $portArg"
Write-Host "Open: http://${hostArg}:$portArg"
Write-Host ""

python apps/web_app.py --host $hostArg --port $portArg
