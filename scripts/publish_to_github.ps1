# Publish this backup to GitHub (run once after: gh auth login)
$ErrorActionPreference = "Stop"
$gh = "$env:LOCALAPPDATA\gh-cli\bin\gh.exe"
if (-not (Test-Path $gh)) {
    Write-Error "gh not found. Install GitHub CLI or re-run tools/create_reproducibility_backup setup."
}
$repoName = "3d-marl-uav-coverage-benchmark"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

& $gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Login required. Complete browser flow:"
    & $gh auth login -h github.com -p https -w
}

& $gh repo create $repoName --public --source=. --remote=origin --description "Reproducible 3D MARL multi-UAV cooperative coverage benchmark (GridWorld3DEnv, QMIX+PBRS)" --push
Write-Host "Done: https://github.com/$( & $gh api user -q .login )/$repoName"
