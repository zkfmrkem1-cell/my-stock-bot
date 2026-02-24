[CmdletBinding()]
param(
    [Nullable[int]]$Port
)

if ($PSBoundParameters.ContainsKey("Port")) {
    & (Join-Path $PSScriptRoot "Run-MarketDashboard.ps1") -Market US -Port $Port
}
else {
    & (Join-Path $PSScriptRoot "Run-MarketDashboard.ps1") -Market US
}
exit $LASTEXITCODE
