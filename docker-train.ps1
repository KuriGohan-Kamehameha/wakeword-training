[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThruArgs
)

$ErrorActionPreference = "Stop"

function Quote-BashArg {
    param([string]$Value)
    if ($null -eq $Value) {
        return "''"
    }
    return "'" + ($Value -replace "'", "'\"'\"'") + "'"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bashScript = Join-Path $scriptDir "docker-train.sh"

if (-not (Test-Path $bashScript)) {
    Write-Error "Missing script: $bashScript"
    exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "docker command not found. Install Docker Desktop and retry."
    exit 1
}

# Prefer native bash when available (Git Bash, MSYS2, WSL shell in PATH).
$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($bash) {
    & $bash.Source $bashScript @PassThruArgs
    exit $LASTEXITCODE
}

# Fallback to WSL when bash is not in Windows PATH.
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Error "Neither bash nor wsl was found. Install Git Bash or enable WSL."
    exit 1
}

$resolvedScriptDir = (Resolve-Path $scriptDir).Path
$wslScriptDir = & $wsl.Source "wslpath" "-a" $resolvedScriptDir
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslScriptDir)) {
    Write-Error "Unable to translate Windows path to WSL path: $resolvedScriptDir"
    exit 1
}

$joinedArgs = ($PassThruArgs | ForEach-Object { Quote-BashArg $_ }) -join " "
$cmd = "cd $(Quote-BashArg $wslScriptDir) && ./docker-train.sh $joinedArgs"
& $wsl.Source "bash" "-lc" $cmd
exit $LASTEXITCODE
