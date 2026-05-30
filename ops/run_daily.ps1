# ops/run_daily.ps1 -- daily prediction wrapper (the 06:30 UTC cron entrypoint).
#
# Loads .env, syncs the Sackmann mirror, then runs `python -m tennis run`
# (Data -> Research -> Modeling -> Briefing -> Monitor). Train at least once
# first (run_train.ps1) or this fails fast with `no_active_model` (S6).
#
# Exits with the pipeline aggregate exit code (S8: 1 iff some agent that ran
# failed; 0 for succeeded/partial/lock-held no-op). Task Scheduler surfaces a
# non-zero exit as the task "Last Run Result", so a failed run is visible.

[CmdletBinding()]
param([string] $ConfigPath)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

if ($ConfigPath) {
    $code = Invoke-Tennis -Command run -ConfigPath $ConfigPath
} else {
    $code = Invoke-Tennis -Command run
}
exit $code
