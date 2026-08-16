[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$BackupRoot,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-NormalizedLfSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $text = [IO.File]::ReadAllText($Path).Replace("`r`n", "`n")
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "")
    }
    finally {
        $sha.Dispose()
    }
}

$bundleRoot = $PSScriptRoot
$overlayRoot = Join-Path $bundleRoot "offline_overlay"
$manifestPath = Join-Path $bundleRoot "IMPLEMENTATION_FILE_MANIFEST.csv"

if (-not (Test-Path -LiteralPath $overlayRoot -PathType Container)) {
    throw "Missing offline_overlay directory next to this script."
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing IMPLEMENTATION_FILE_MANIFEST.csv next to this script."
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "ProjectRoot does not exist: $ProjectRoot"
}

$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$projectDriveRoot = [IO.Path]::GetPathRoot($project).TrimEnd("\")
if ($project.TrimEnd("\") -eq $projectDriveRoot) {
    throw "Refusing to use a drive root as ProjectRoot: $project"
}
if (-not (Test-Path -LiteralPath (Join-Path $project "gr00t") -PathType Container)) {
    throw "ProjectRoot does not look like Isaac-GR00T: missing gr00t directory."
}

if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupRoot = Join-Path (Split-Path -Parent $project) "a2a_backup_$stamp"
}
if (Test-Path -LiteralPath $BackupRoot) {
    throw "BackupRoot already exists; choose a new empty path: $BackupRoot"
}

$manifest = Import-Csv -LiteralPath $manifestPath
if ($manifest.Count -eq 0) {
    throw "Manifest is empty."
}

# Validate the bundle itself before touching the target project.
foreach ($row in $manifest) {
    $source = Join-Path $overlayRoot $row.Path
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Overlay file is missing: $($row.Path)"
    }
    $actual = Get-NormalizedLfSha256 -Path $source
    if ($actual -ne $row.NormalizedLF_SHA256) {
        throw "Overlay SHA-256 mismatch for $($row.Path)"
    }
}

$operations = @()
foreach ($row in $manifest) {
    $target = Join-Path $project $row.Path
    $targetExists = Test-Path -LiteralPath $target -PathType Leaf
    $targetHash = if ($targetExists) { Get-NormalizedLfSha256 -Path $target } else { $null }

    if ($targetHash -eq $row.NormalizedLF_SHA256) {
        $operations += [pscustomobject]@{ Row = $row; State = "AlreadyCurrent" }
        continue
    }

    if ($row.ChangeType -eq "Modified") {
        if (-not $targetExists) {
            throw "Required baseline file is missing: $($row.Path)"
        }
        if (-not $Force -and $targetHash -ne $row.Baseline_NormalizedLF_SHA256) {
            throw (
                "Local drift detected in $($row.Path). " +
                "Use the exact N1.7 General Release base, merge manually with the patch, " +
                "or rerun with -Force after reviewing and backing up your changes."
            )
        }
    }
    elseif ($targetExists -and -not $Force) {
        throw "New A2A path already exists with different content: $($row.Path)"
    }

    $operations += [pscustomobject]@{ Row = $row; State = "Copy" }
}

$copyOperations = @($operations | Where-Object State -eq "Copy")
if ($copyOperations.Count -eq 0) {
    Write-Host "All $($manifest.Count) implementation files are already current."
    exit 0
}

if ($PSCmdlet.ShouldProcess($project, "Back up and install $($copyOperations.Count) A2A files")) {
    New-Item -ItemType Directory -Path $BackupRoot | Out-Null
    $newFiles = [Collections.Generic.List[string]]::new()

    foreach ($operation in $copyOperations) {
        $row = $operation.Row
        $source = Join-Path $overlayRoot $row.Path
        $target = Join-Path $project $row.Path
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $backup = Join-Path $BackupRoot $row.Path
            New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
            Copy-Item -LiteralPath $target -Destination $backup -Force
        }
        else {
            $newFiles.Add($row.Path)
        }

        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Force
    }

    $newFiles | Set-Content -LiteralPath (Join-Path $BackupRoot "NEW_FILES_TO_REMOVE_ON_ROLLBACK.txt") -Encoding UTF8

    foreach ($row in $manifest) {
        $target = Join-Path $project $row.Path
        $actual = Get-NormalizedLfSha256 -Path $target
        if ($actual -ne $row.NormalizedLF_SHA256) {
            throw "Post-copy verification failed for $($row.Path)"
        }
    }

    Write-Host "Installed and verified $($manifest.Count) implementation files."
    Write-Host "Backup: $BackupRoot"
}
