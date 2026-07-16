$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Installer = Join-Path $Root "install.py"

if (Get-Command py -ErrorAction SilentlyContinue) {
    $Installed = $false
    foreach ($Version in @("3.13", "3.12", "3.11")) {
        & py "-$Version" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) {
            & py "-$Version" $Installer @args
            $Installed = $true
            break
        }
    }
    if (-not $Installed) {
        Write-Error "未找到 Python 3.11、3.12 或 3.13。"
        exit 2
    }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $Installer @args
} else {
    Write-Error "未找到 Python。请先安装 Python 3.11、3.12 或 3.13。"
    exit 2
}

exit $LASTEXITCODE
