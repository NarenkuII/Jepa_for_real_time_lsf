param(
    [string]$DownloadsDir = (Join-Path $HOME "Downloads"),
    [int]$TimeoutMinutes = 60,
    [string]$Python = ".venv-jepa\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$startedAt = Get-Date
$extensions = @(".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z")
$baseline = @{}

Get-ChildItem -LiteralPath $DownloadsDir -File -ErrorAction SilentlyContinue |
    Where-Object {
        $extensions -contains $_.Extension.ToLowerInvariant() -and
        $_.Length -gt 0
    } |
    ForEach-Object { $baseline[$_.FullName] = $_.Length }

Write-Output "Watching $DownloadsDir for a new Mediapi-RGB archive after $($startedAt.ToString('o'))"

$deadline = $startedAt.AddMinutes($TimeoutMinutes)
$stableSizes = @{}
while ((Get-Date) -lt $deadline) {
    $candidates = Get-ChildItem -LiteralPath $DownloadsDir -File -ErrorAction SilentlyContinue |
        Where-Object {
            $extensions -contains $_.Extension.ToLowerInvariant() -and
            -not $baseline.ContainsKey($_.FullName) -and
            $_.LastWriteTime -ge $startedAt -and
            $_.Length -gt 0
        } |
        Sort-Object LastWriteTime

    foreach ($candidate in $candidates) {
        if ($stableSizes[$candidate.FullName] -eq $candidate.Length) {
            Write-Output "Stable archive detected: $($candidate.FullName)"
            & $Python tools\bootstrap_lsf_datasets.py --mediapi-archive $candidate.FullName
            exit $LASTEXITCODE
        }
        $stableSizes[$candidate.FullName] = $candidate.Length
    }

    Start-Sleep -Seconds 10
}

throw "No new stable Mediapi-RGB archive appeared within $TimeoutMinutes minutes."
