Set-StrictMode -Version Latest

function Set-DotEnvFromFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Env file not found: $Path"
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line) { return }
        if ($line.StartsWith("#")) { return }

        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }

        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }

        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()

        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            if ($value.Length -ge 2) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }

        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

function Import-MarketEnv {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("US", "KR")]
        [string]$Market,

        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $shared = Join-Path $RepoRoot ".ai/.env.shared"
    $marketFile = Join-Path $RepoRoot ".ai/.env.$($Market.ToLowerInvariant())"

    if (Test-Path -LiteralPath $shared) {
        Set-DotEnvFromFile -Path $shared
    }

    Set-DotEnvFromFile -Path $marketFile
}

