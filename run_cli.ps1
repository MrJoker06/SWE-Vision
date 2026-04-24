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

if (-not $env:OPENAI_API_KEY) { throw "OPENAI_API_KEY is not set" }
if (-not $env:OPENAI_MODEL) { $env:OPENAI_MODEL = "gpt-5.4" }

$imagePath = if ($args.Count -ge 1) { $args[0] } else { "./assets/dogs.png" }
$query = if ($args.Count -ge 2) { $args[1] } else { "图中有几只小狗，不包含黄色那只？" }

python -m swe_vision.cli `
  --image $imagePath `
  --model $env:OPENAI_MODEL `
  --reasoning `
  $query
