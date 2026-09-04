[CmdletBinding()]
param(
    [string]$Repository = "zorosdrone/pi-zero-camera-gimbal-tracker",
    [string]$Tag = "v0.1.0-kcf-demo",
    [switch]$Prerelease
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mediaDir = Join-Path $projectRoot "media"
$notesFile = Join-Path $mediaDir "release-notes-v0.1.0-kcf-demo.md"
$assets = @(
    (Join-Path $mediaDir "gimbal-kcf-roi-browser-demo-v0.1.0.mp4"),
    (Join-Path $mediaDir "gimbal-kcf-servo-demo-v0.1.0.mp4")
)

foreach ($asset in $assets + $notesFile) {
    if (-not (Test-Path -LiteralPath $asset -PathType Leaf)) {
        throw "Release file not found: $asset"
    }
}

$releaseArgs = @(
    "release", "create", $Tag,
    "--repo", $Repository,
    "--title", "KCF ROI Tracking Demo v0.1.0",
    "--notes-file", $notesFile
)
if ($Prerelease) {
    $releaseArgs += "--prerelease"
}
$releaseArgs += $assets

& gh @releaseArgs
if ($LASTEXITCODE -ne 0) {
    throw "GitHub Release creation failed."
}
