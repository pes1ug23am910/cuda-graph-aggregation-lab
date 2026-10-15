# Collect diagnostics separately from ordinary benchmark timings.
# Sources: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
# https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
[CmdletBinding()]
param(
    [string]$Executable = (Join-Path $PSScriptRoot '../build/cuda/Release/graphlab.exe'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../results/profiles'),
    [ValidateRange(1, 2147483647)][int]$Nodes = 4096,
    [ValidateRange(1, 2147483647)][int]$Features = 64,
    [ValidateSet('regular8', 'regular32', 'skewed')][string]$Family = 'skewed',
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [string]$Nsys = 'nsys',
    [string]$Ncu = 'ncu',
    [string]$Sanitizer = 'compute-sanitizer'
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable does not exist: $Executable" }
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$runDirectory = Join-Path $OutputDirectory ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '-' + $PID)
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$runDirectory = (Resolve-Path -LiteralPath $runDirectory).Path
$statusFile = Join-Path $runDirectory 'status.jsonl'
$script:hadFailure = $false
$script:hadMissing = $false
$script:lastRunCode = 0

function Add-Status {
    param([string]$Label, [string]$Status, [object]$Code, [string]$ToolPath, [string[]]$ToolArguments, [string]$Reason = '')
    [ordered]@{
        type = 'diagnostic_command'; label = $Label; status = $Status; exit_code = $Code
        utc = (Get-Date).ToUniversalTime().ToString('o'); executable = $ToolPath
        arguments = @($ToolArguments); reason = $Reason
    } | ConvertTo-Json -Compress -Depth 5 | Add-Content -LiteralPath $statusFile -Encoding utf8
}
function Invoke-Diagnostic {
    param([string]$Label, [string]$Tool, [string[]]$ToolArguments)
    $found = Get-Command -Name $Tool -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $found) {
        Add-Status $Label 'not_run' $null $Tool $ToolArguments 'Tool is unavailable'
        $script:hadMissing = $true; $script:lastRunCode = 77; return
    }
    $logPath = Join-Path $runDirectory ($Label + '.log')
    try {
        & $found.Source @ToolArguments *> $logPath
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
        $script:lastRunCode = $code
        if ($code -eq 0) { Add-Status $Label 'completed' $code $found.Source $ToolArguments }
        else { Add-Status $Label 'failed' $code $found.Source $ToolArguments; $script:hadFailure = $true }
    } catch {
        $_ | Out-String | Add-Content -LiteralPath $logPath -Encoding utf8
        Add-Status $Label 'failed' 1 $found.Source $ToolArguments $_.Exception.Message
        $script:hadFailure = $true; $script:lastRunCode = 1
    }
}

Invoke-Diagnostic 'environment' $Executable @('info')
Invoke-Diagnostic 'nsys-version' $Nsys @('--version')
Invoke-Diagnostic 'ncu-version' $Ncu @('--version')
Invoke-Diagnostic 'sanitizer-version' $Sanitizer @('--version')
foreach ($backend in @('naive', 'row_split', 'cusparse')) {
    $appArguments = @('bench', '--backend', $backend, '--nodes', "$Nodes", '--features', "$Features", '--family', $Family, '--seed', "$Seed", '--groups', '1', '--calls', '10', '--warmup', '3', '--smoke')
    $prefix = Join-Path $runDirectory $backend
    Invoke-Diagnostic ($backend + '-nsys') $Nsys (@('profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none', '--stats=true', '--force-overwrite=true', '--output', ($prefix + '-timeline'), $Executable) + $appArguments)
    # Profile the first few kernels, including library kernels, without claiming
    # profiler replay time is an ordinary latency measurement.
    Invoke-Diagnostic ($backend + '-ncu') $Ncu (@('--set', 'basic', '--target-processes', 'all', '--launch-count', '3', '--force-overwrite', '--export', ($prefix + '-kernels'), $Executable) + $appArguments)
    $sanitizerArguments = @('bench', '--backend', $backend, '--nodes', "$Nodes", '--features', "$Features", '--family', $Family, '--seed', "$Seed", '--groups', '1', '--calls', '2', '--warmup', '1', '--smoke')
    Invoke-Diagnostic ($backend + '-memcheck') $Sanitizer (@('--tool', 'memcheck', '--error-exitcode', '99', '--leak-check', 'full', $Executable) + $sanitizerArguments)
    $memcheckCode = $script:lastRunCode
    foreach ($check in @('initcheck', 'synccheck', 'racecheck')) {
        $checkArguments = @('--tool', $check, '--error-exitcode', '99', $Executable) + $sanitizerArguments
        if ($memcheckCode -eq 0) { Invoke-Diagnostic ($backend + '-' + $check) $Sanitizer $checkArguments }
        else { Add-Status ($backend + '-' + $check) 'not_run' $null $Sanitizer $checkArguments 'Memcheck did not complete successfully'; $script:hadMissing = $true }
    }
}
Write-Output "Diagnostic logs and command statuses: $runDirectory"
Write-Output 'Inspect report files and logs before interpreting completed commands as usable evidence.'
if ($script:hadFailure) { exit 1 }
if ($script:hadMissing) { exit 77 }
exit 0
