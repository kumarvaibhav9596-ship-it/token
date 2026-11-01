# Keeps the Gnani API key in this PowerShell process only, then starts the demo.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pythonExe = if ($pythonCommand) { $pythonCommand.Source } else {
    'C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
}
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Python was not found. Install Python 3.10+ or ask Codex to locate its bundled Python runtime.'
}

Set-Location -LiteralPath $projectRoot
& $pythonExe -c 'import gnani' *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing the optional official Gnani client…' -ForegroundColor Cyan
    & $pythonExe -m pip install -r .\requirements-voice.txt
    if ($LASTEXITCODE -ne 0) { throw 'Gnani client installation failed.' }
}

Write-Host 'Enter the Gnani API key at the hidden prompt. It will not be saved in a project file.' -ForegroundColor Cyan
$secureKey = Read-Host 'Gnani API key (hidden)' -AsSecureString
if ($secureKey.Length -eq 0) { throw 'An API key is required to start voice mode.' }
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:GNANI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secureKey.Dispose()
}

Write-Host 'Starting voice-enabled ZentroMart at http://127.0.0.1:8767. Stop with Ctrl+C to clear the key.' -ForegroundColor Green
try {
    & $pythonExe .\agent.py --port 8767
} finally {
    Remove-Item Env:GNANI_API_KEY -ErrorAction SilentlyContinue
}
