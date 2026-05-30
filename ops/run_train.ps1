# ops/run_train.ps1 -- training wrapper (run on demand or a slow cron).
#
# Loads .env, syncs the Sackmann mirror, then runs `python -m tennis train`
# (Data -> Research(training) -> Modeling(training)). This is the bootstrap the
# daily chain needs: `run` fails fast with `no_active_model` until a model has
# been trained and activated (S6).
#
# Exits with the pipeline aggregate exit code (S8: 1 iff some agent that ran
# failed; 0 for succeeded/partial/lock-held no-op).

[CmdletBinding()]
param([string] $ConfigPath)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

if ($ConfigPath) {
    $code = Invoke-Tennis -Command train -ConfigPath $ConfigPath
} else {
    $code = Invoke-Tennis -Command train
}
exit $code
