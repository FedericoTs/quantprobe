# Dev helper: rebuild cmcore (release) and report bpc + round-trip on enwik8 slices.
#   pwsh -File bench.ps1 [-label "..."] [-with4]
param([string]$label = "", [switch]$with4)
$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"
Push-Location $PSScriptRoot
$build = & $cargo build --release 2>&1
if ($LASTEXITCODE -ne 0) { $build | Select-String "error" | Select-Object -First 10; Pop-Location; exit 1 }
$exe = ".\target\release\cmcore.exe"
$gt = "..\data\corpora\generic-text"
function B($n) {
    $src = "$gt\$n"; $c = "$env:TEMP\$n.cmp"; $d = "$env:TEMP\$n.dec"
    & $exe c $src $c; & $exe d $c $d
    $o = (Get-Item $src).Length; $cl = (Get-Item $c).Length
    $ok = (Get-FileHash $src).Hash -eq (Get-FileHash $d).Hash
    "{0,-12} bpc={1:N4}  ratio={2:N3}  rt={3}" -f $n, (8.0 * $cl / $o), ($o / $cl), $(if ($ok) { 'OK' } else { 'FAIL' })
}
if ($label) { "[$label]" }
B "enwik8_1mb"
if ($with4) { B "enwik8_4mb" }
Pop-Location