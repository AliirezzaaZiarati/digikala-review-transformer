# Stage-2 launcher: waits for Stage 1, picks the variant, smoke-tests and trains
# each encoder, then packages the outputs. Safe to re-run after an interruption.
$ErrorActionPreference = "Continue"   # keep "Continue": "Stop" breaks on harmless native stderr in Windows PowerShell 5.1
$RunsRoot       = "artifacts\transformer_runs"
$PollSeconds    = 60
$MaxTrainTries  = 3
$Models         = @("roberta", "mdeberta")   # order matters: a finished model is kept if a later one fails

# ---- Section: wait for Stage 1 and obtain the chosen variant ----
$variant = $null
while ($true) {
    $output = uv run python scripts/pick_variant.py --runs-root $RunsRoot
    $code = $LASTEXITCODE
    if ($code -eq 0) { $variant = ($output | Select-Object -Last 1).ToString().Trim(); break }
    if ($code -ne 3) { Write-Host "[FAIL] pick_variant failed (exit $code). Send a screenshot of this window."; exit 1 }
    Write-Host "$(Get-Date -Format HH:mm) Stage 1 not finished yet, waiting..."
    Start-Sleep -Seconds $PollSeconds
}
if ($variant -notin @("B", "C")) { Write-Host "[FAIL] unexpected variant '$variant'."; exit 1 }
Write-Host "[INFO] Stage-2 variant: $variant"

# ---- Section: package Stage-1 outputs so they can be sent immediately ----
uv run python scripts/package_run_outputs.py --runs tooka_B tooka_C
if ($LASTEXITCODE -eq 0) {
    Write-Host "=============================================================="
    Write-Host "SEND NOW: outputs_to_send\tooka_B.zip and outputs_to_send\tooka_C.zip"
    Write-Host "=============================================================="
} else { Write-Host "[WARN] Stage-1 packaging failed; send a screenshot of this window." }

# ---- Section: smoke test, then train, one encoder at a time ----
$done = @()
foreach ($model in $Models) {
    $config = "configs/transformer/${model}_${variant}.yaml"
    uv run python scripts/train_transformer.py --smoke --config $config
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] smoke failed for $model, skipping it."; continue }
    $finished = $false
    for ($attempt = 1; $attempt -le $MaxTrainTries; $attempt++) {
        Write-Host "[INFO] training $model, attempt $attempt of $MaxTrainTries"
        uv run python scripts/train_transformer.py --config $config
        if ($LASTEXITCODE -eq 0) { $finished = $true; break }
        Start-Sleep -Seconds 30
    }
    if ($finished) { $done += "${model}_${variant}" } else { Write-Host "[FAIL] training failed for $model." }
}

# ---- Section: package Stage-2 outputs ----
if ($done.Count -gt 0) { uv run python scripts/package_run_outputs.py --runs $done }
Write-Host "[INFO] Finished runs: $($done -join ', ')"