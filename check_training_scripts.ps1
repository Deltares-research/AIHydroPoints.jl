#Requires -Version 5.1
# check_training_scripts.ps1
#
# Smoke-tests all training and inference scripts end-to-end.
# Runs jobs in parallel using PowerShell background jobs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File check_training_scripts.ps1
#   # or in PowerShell 7+:
#   pwsh check_training_scripts.ps1
#
# Exit code:
#   0 — all attempted scripts passed
#   1 — one or more scripts failed

Set-Location $PSScriptRoot

$script:pass    = 0
$script:fail    = 0
$script:results = [System.Collections.Generic.List[string]]::new()

# ── Verify bat scripts exist ───────────────────────────────────────────────────
if (-not (Test-Path "bin\train.bat") -or -not (Test-Path "bin\predict.bat")) {
    Write-Error "bin\train.bat or bin\predict.bat not found. Run from the repo root."
    exit 1
}

# ── Parallel group runner ──────────────────────────────────────────────────────
# $Jobs: array of @{Label="..."; Command="bin\train.bat examples\X.toml"}
# All jobs in a group are launched simultaneously; results are shown after all finish.
function Invoke-JobGroup {
    param([hashtable[]]$Jobs)

    $repoRoot = $PSScriptRoot

    # Launch all jobs as background PowerShell jobs
    $bgJobs = foreach ($job in $Jobs) {
        $label   = $job.Label
        $command = $job.Command
        Start-Job -Name $label -ScriptBlock {
            param($root, $cmd)
            Set-Location $root
            $output = cmd /c $cmd 2>&1
            @{ Output = $output; ExitCode = $LASTEXITCODE }
        } -ArgumentList $repoRoot, $command
    }

    # Wait for all jobs to complete
    $null = Wait-Job -Job $bgJobs

    # Display results in launch order
    foreach ($bg in $bgJobs) {
        $data    = Receive-Job -Job $bg
        $elapsed = [int](($bg.PSEndTime - $bg.PSBeginTime).TotalSeconds)
        Remove-Job -Job $bg

        Write-Host ""
        Write-Host ("-" * 65)
        Write-Host "--- $($bg.Name)"
        $data.Output | ForEach-Object { Write-Host "[$($bg.Name)] $_" }

        if ($data.ExitCode -eq 0) {
            Write-Host "  PASS -- $elapsed s"
            $script:results.Add("PASS  $($bg.Name)  ($elapsed s)")
            $script:pass++
        } else {
            Write-Host "  FAIL -- $elapsed s  (exit $($data.ExitCode))"
            $script:results.Add("FAIL  $($bg.Name)  ($elapsed s)")
            $script:fail++
        }
    }
}

# ── Jobs ───────────────────────────────────────────────────────────────────────
Invoke-JobGroup @(
    @{ Label = "train LinearSurgeModel";       Command = "bin\train.bat examples\LinearSurgeModel.toml" }
    @{ Label = "train ConvSurgeModel";          Command = "bin\train.bat examples\ConvSurgeModel.toml" }
    @{ Label = "train AttentionSurgeModel";     Command = "bin\train.bat examples\AttentionSurgeModel.toml" }
    @{ Label = "train DeepONetTideModel";       Command = "bin\train.bat examples\DeepONetTideModel.toml" }
    @{ Label = "train ProductTideModel";        Command = "bin\train.bat examples\ProductTideModel.toml" }
    @{ Label = "train ConvWaveModel";           Command = "bin\train.bat examples\ConvWaveModel.toml" }
    @{ Label = "train DeepONetWaveModel";       Command = "bin\train.bat examples\DeepONetWaveModel.toml" }
    @{ Label = "train ConvInteractionModel";    Command = "bin\train.bat examples\ConvInteractionModel.toml" }
)

Invoke-JobGroup @(
    @{ Label = "predict ConvSurgeModel";   Command = "bin\predict.bat examples\predict_ConvSurgeModel.toml" }
    @{ Label = "predict LinearSurgeModel"; Command = "bin\predict.bat examples\predict_LinearSurgeModel.toml" }
)

# ── Summary ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("-" * 65)
Write-Host "--- Summary"
Write-Host ("-" * 65)
foreach ($r in $script:results) { Write-Host "  $r" }
Write-Host ""
Write-Host "  Passed: $($script:pass)   Failed: $($script:fail)"
Write-Host ("-" * 65)

exit ([int]($script:fail -gt 0))
