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

if (-not $env:DEEPSEEK_API_KEY) { throw "DEEPSEEK_API_KEY is not set" }
if (-not $env:DEEPSEEK_BASE_URL) { $env:DEEPSEEK_BASE_URL = "https://api.deepseek.com" }
if (-not $env:DEEPSEEK_MODEL) { $env:DEEPSEEK_MODEL = "deepseek-v4-pro" }
if (-not $env:VLM_REASONING_EFFORT) { $env:VLM_REASONING_EFFORT = "high" }

$imagePath = if ($args.Count -ge 1) { $args[0] } else { "./assets/冷冽谷的伊鲁席尔.png" }
$query = if ($args.Count -ge 2) { $args[1] } else { "图中有哪些颜色" }
$model = if ($args.Count -ge 3) { $args[2] } else { $env:DEEPSEEK_MODEL }
$reasoningEffort = if ($args.Count -ge 4) { $args[3] } else { $env:VLM_REASONING_EFFORT }

python -m swe_vision.cli `
  --image $imagePath `
  --model $model `
  --api-key $env:DEEPSEEK_API_KEY `
  --base-url $env:DEEPSEEK_BASE_URL `
  --provider deepseek `
  --reasoning `
  --reasoning-effort $reasoningEffort `
  $query
