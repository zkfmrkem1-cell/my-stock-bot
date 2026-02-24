[CmdletBinding()]
param(
    [string[]]$Tickers,
    [string]$StartDate,
    [string]$EndDate,
    [switch]$FullRefresh,
    [switch]$SkipQC,
    [switch]$SkipLabels,
    [switch]$SkipNews,
    [switch]$SkipStatusReport,
    [switch]$SkipDiscord
)

& (Join-Path $PSScriptRoot "Run-MarketPipeline.ps1") `
    -Market US `
    -Tickers $Tickers `
    -StartDate $StartDate `
    -EndDate $EndDate `
    -FullRefresh:$FullRefresh `
    -SkipQC:$SkipQC `
    -SkipLabels:$SkipLabels `
    -SkipNews:$SkipNews `
    -SkipStatusReport:$SkipStatusReport `
    -SkipDiscord:$SkipDiscord

exit $LASTEXITCODE

