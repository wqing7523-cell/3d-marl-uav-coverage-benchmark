# Publish reproducibility bundle to GitHub
# Usage (from repo root):
#   python tools/publish_to_github.py
#   python tools/publish_to_github.py --dry-run
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
python tools/publish_to_github.py @args
