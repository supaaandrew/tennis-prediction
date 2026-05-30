# ops/lib.ps1 -- shared helpers for the Tennis Prediction Bot ops wrappers.
#
# Dot-source this file, then call the helpers:
#     . "$PSScriptRoot\lib.ps1"
#     Import-DotEnv
#     Invoke-Tennis -Command run
#
# It contains NO secrets -- every secret is read from the gitignored .env at
# runtime by Import-DotEnv. Safe to commit. ASCII-only on purpose: Windows
# PowerShell 5.1 reads BOM-less scripts as ANSI, so non-ASCII chars mojibake.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    # lib.ps1 lives in <repo>/ops; the repo root is its parent.
    return (Split-Path -Parent $PSScriptRoot)
}

function Import-DotEnv {
    <#
    .SYNOPSIS
        Load KEY=VALUE pairs from a .env file into the current process env.
    .DESCRIPTION
        The app reads os.environ directly (core/config.py) and does NOT autoload
        a .env, so every invocation (alembic, train, run) must export it first.

        Robust parser -- the project .env has inline comments and a value with
        internal spaces (a Gmail app password), so a naive parser corrupts it:
          - skips blank lines and whole-line comments (lines starting with #)
          - splits on the FIRST '=' only (values contain '=', ':', '@', '//')
          - strips one surrounding pair of single or double quotes
          - strips a trailing inline ' #comment' for UNQUOTED values only, so a
            '#' embedded in a value survives and an empty value + comment is ''
          - preserves internal spaces (e.g. 'aaaa bbbb cccc dddd')
    #>
    param(
        [string] $Path = (Join-Path (Get-RepoRoot) ".env")
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Import-DotEnv: env file not found at $Path"
    }
    foreach ($raw in (Get-Content -LiteralPath $Path)) {
        $line = $raw.TrimStart()
        if ($line -eq "" -or $line.StartsWith("#")) { continue }

        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { continue }

        $name = $line.Substring(0, $eq).Trim()
        if ($name.StartsWith("export ")) { $name = $name.Substring(7).Trim() }
        if ($name -eq "") { continue }

        $value = $line.Substring($eq + 1)
        $trimmed = $value.Trim()

        $isQuoted = $false
        if ($trimmed.Length -ge 2 -and
            ((($trimmed[0] -eq '"') -and ($trimmed[-1] -eq '"')) -or
             (($trimmed[0] -eq "'") -and ($trimmed[-1] -eq "'")))) {
            $value = $trimmed.Substring(1, $trimmed.Length - 2)
            $isQuoted = $true
        }

        if (-not $isQuoted) {
            # Inline comment = whitespace followed by '#'. Cut there.
            $m = [regex]::Match($value, '\s#')
            if ($m.Success) { $value = $value.Substring(0, $m.Index) }
            $value = $value.Trim()
        }

        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

function Get-VenvPython {
    $py = Join-Path (Get-RepoRoot) ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $py)) {
        throw "venv python not found at $py - create it and run: pip install -e .[dev]"
    }
    return $py
}

function Sync-SackmannMirror {
    # Thin wrapper over scripts/sync_sackmann_mirror.py (git clone/pull + mtime
    # refresh). A stale mirror dir mtime trips the Sackmann staleness halt (C5),
    # so the daily wrappers always sync before running.
    $root = Get-RepoRoot
    $py = Get-VenvPython
    & $py (Join-Path $root "scripts\sync_sackmann_mirror.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Sackmann mirror sync failed (exit $LASTEXITCODE)"
    }
}

function Invoke-Tennis {
    <#
    .SYNOPSIS
        Load .env, sync the Sackmann mirror, then run `python -m tennis <cmd>`.
    .PARAMETER Command
        'train' or 'run'.
    .PARAMETER SkipMirrorSync
        Skip the mirror sync (e.g. when the caller already synced).
    .PARAMETER ConfigPath
        Optional --config override (defaults to the CLI's config/config.yaml).
    .OUTPUTS
        The child process exit code (S8: 1 iff the aggregate run failed).
    #>
    param(
        [Parameter(Mandatory)][ValidateSet("train", "run")][string] $Command,
        [switch] $SkipMirrorSync,
        [string] $ConfigPath
    )
    $root = Get-RepoRoot
    Import-DotEnv -Path (Join-Path $root ".env")
    $py = Get-VenvPython

    if (-not $SkipMirrorSync) { Sync-SackmannMirror }

    $env:PYTHONPATH = Join-Path $root "src"
    $cliArgs = @("-m", "tennis", $Command)
    if ($ConfigPath) { $cliArgs += @("--config", $ConfigPath) }

    Push-Location $root
    try {
        # Route child stdout to the host so it does NOT leak into this
        # function's output stream; we return only the exit code. (stderr from
        # structlog goes straight to the console on its own.)
        & $py @cliArgs | Out-Host
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    return $exitCode
}
