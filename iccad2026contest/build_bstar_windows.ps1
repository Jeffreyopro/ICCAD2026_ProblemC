# Build the optional bstar_core C extension on Windows with MSYS2 UCRT64 gcc.
#
# Usage (from iccad2026contest/):
#   powershell -ExecutionPolicy Bypass -File .\build_bstar_windows.ps1
#
# The resulting bstar_core.cp311-win_amd64.pyd must match the running Python
# version (the file name is generated automatically below).  If you use a
# different Python, change $PyTag or build with `python setup.py build_ext`.
$ErrorActionPreference = 'Stop'

$Gcc = 'C:\msys64\ucrt64\bin\gcc.exe'
if (-not (Test-Path $Gcc)) {
    throw "gcc not found at $Gcc. Install MSYS2 UCRT64 or edit `$Gcc."
}

$Py = (Get-Command python).Source
$PyDir = Split-Path $Py
$PyInclude = Join-Path $PyDir 'Include'
$PyLibs = Join-Path $PyDir 'libs'
$PyTag = (& python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')").Trim()
$Out = "bstar_core.$PyTag-win_amd64.pyd"

& $Gcc -shared -O3 -DMS_WIN64 -D_CRT_SECURE_NO_WARNINGS `
    "-I$PyInclude" bstar_core.c -o $Out `
    "-L$PyLibs" -lpython$($PyTag.Substring(2))

if ($LASTEXITCODE -ne 0) { throw "gcc failed with exit code $LASTEXITCODE" }
Write-Host "Built $Out"
& python -c "import bstar_core; print('bstar_core import OK:', bstar_core.__file__)"
