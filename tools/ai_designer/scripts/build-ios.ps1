#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Builds efficient-egui-template for iOS architectures and bundles .xcframework.
.DESCRIPTION
    Compiles crates/app for physical iOS devices (aarch64-apple-ios)
    and iOS simulators (aarch64-apple-ios-sim, x86_64-apple-ios).
    Combines simulator slices with lipo and packages into an XCFramework.
.PARAMETER OutputDir
    Destination directory for compiled iOS binaries and XCFramework. Defaults to "target/ios".
.EXAMPLE
    ./scripts/build-ios.ps1
#>
param(
    [string]$OutputDir = "target/ios"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " Building efficient-egui-template for iOS"
Write-Host "============================================================"

$targets = @("aarch64-apple-ios", "aarch64-apple-ios-sim", "x86_64-apple-ios")

$installedTargets = rustup target list --installed
foreach ($target in $targets) {
    if ($installedTargets -notcontains $target) {
        Write-Host "[*] Installing missing target: $target..."
        rustup target add $target
    }
}

New-Item -ItemType Directory -Force -Path "$OutputDir/device" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir/simulator" | Out-Null

foreach ($target in $targets) {
    Write-Host "[*] Compiling crates/app for $target..."
    cargo build --package app --release --target $target
}

Copy-Item -Force "target/aarch64-apple-ios/release/libapp.a" "$OutputDir/device/libEguiApp.a" -ErrorAction SilentlyContinue

if (Get-Command lipo -ErrorAction SilentlyContinue) {
    Write-Host "[*] Assembling universal simulator binary with lipo..."
    lipo -create `
        "target/aarch64-apple-ios-sim/release/libapp.a" `
        "target/x86_64-apple-ios/release/libapp.a" `
        -output "$OutputDir/simulator/libEguiApp.a"
} else {
    Write-Host "[i] lipo not available on this host. Preserving individual simulator slices."
    Copy-Item -Force "target/aarch64-apple-ios-sim/release/libapp.a" "$OutputDir/simulator/libEguiApp_arm64.a" -ErrorAction SilentlyContinue
    Copy-Item -Force "target/x86_64-apple-ios/release/libapp.a" "$OutputDir/simulator/libEguiApp_x86_64.a" -ErrorAction SilentlyContinue
}

if (Get-Command xcodebuild -ErrorAction SilentlyContinue) {
    Write-Host "[*] Generating EguiApp.xcframework via xcodebuild..."
    $xcframeworkPath = "$OutputDir/EguiApp.xcframework"
    if (Test-Path $xcframeworkPath) {
        Remove-Item -Recurse -Force $xcframeworkPath
    }
    xcodebuild -create-xcframework `
        -library "$OutputDir/device/libEguiApp.a" `
        -library "$OutputDir/simulator/libEguiApp.a" `
        -output $xcframeworkPath
    Write-Host "[+] Generated $xcframeworkPath"
} else {
    Write-Host "[i] xcodebuild not available on this host. Static libraries ready for Xcode import."
}

Write-Host "[+] iOS build completed successfully! Output: $OutputDir"
