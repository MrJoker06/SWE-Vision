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

function Get-ChildProcessIds {
    param([int]$ParentId)

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Get-ChildProcessIds -ParentId $child.ProcessId
        $child.ProcessId
    }
}

function Stop-ProcessTree {
    param([int]$RootProcessId)

    $processIds = @(Get-ChildProcessIds -ParentId $RootProcessId)
    $processIds += $RootProcessId

    foreach ($processId in $processIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host ""
Write-Host "SWE-Vision Web App"
Write-Host "Host: $hostArg"
Write-Host "Port: $portArg"
Write-Host "Open: http://${hostArg}:$portArg"
Write-Host ""

$python = (Get-Command python).Source
$processInfo = [System.Diagnostics.ProcessStartInfo]::new()
$processInfo.FileName = $python
$processInfo.UseShellExecute = $false
[void]$processInfo.ArgumentList.Add("apps/web_app.py")
[void]$processInfo.ArgumentList.Add("--host")
[void]$processInfo.ArgumentList.Add($hostArg)
[void]$processInfo.ArgumentList.Add("--port")
[void]$processInfo.ArgumentList.Add($portArg)

$webProcess = [System.Diagnostics.Process]::Start($processInfo)
$exitCode = 0

try {
    $webProcess.WaitForExit()
    $exitCode = $webProcess.ExitCode
} finally {
    Stop-ProcessTree -RootProcessId $webProcess.Id
}

exit $exitCode
