$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "..\data\calibration\opencv_left"
New-Item -ItemType Directory -Force -Path $target | Out-Null
$imageNumbers = 1..9 + 11..14
$imageNumbers | ForEach-Object {
    $name = "left{0:D2}.jpg" -f $_
    $url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/$name"
    $destination = Join-Path $target $name
    if (Test-Path -LiteralPath $destination) {
        Write-Host "exists $name"
        return
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $destination
            Write-Host "downloaded $name"
            break
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds 2
        }
    }
}
Write-Host "Calibration images: $target"
