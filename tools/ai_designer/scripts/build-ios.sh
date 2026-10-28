#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-target/ios}"

echo "============================================================"
echo " Building efficient-egui-template for iOS"
echo "============================================================"

TARGETS=("aarch64-apple-ios" "aarch64-apple-ios-sim" "x86_64-apple-ios")

for target in "${TARGETS[@]}"; do
    if ! rustup target list --installed | grep -q "^${target}\$"; then
        echo "[*] Installing missing target: ${target}..."
        rustup target add "$target"
    fi
done

mkdir -p "$OUTPUT_DIR/device"
mkdir -p "$OUTPUT_DIR/simulator"

for target in "${TARGETS[@]}"; do
    echo "[*] Compiling crates/app for ${target}..."
    cargo build --package app --release --target "$target"
done

cp -f target/aarch64-apple-ios/release/libapp.a "$OUTPUT_DIR/device/libEguiApp.a" 2>/dev/null || true

if command -v lipo &> /dev/null; then
    echo "[*] Assembling universal simulator binary with lipo..."
    lipo -create \
        target/aarch64-apple-ios-sim/release/libapp.a \
        target/x86_64-apple-ios/release/libapp.a \
        -output "$OUTPUT_DIR/simulator/libEguiApp.a"
else
    echo "[i] lipo not available. Preserving individual simulator slices."
    cp -f target/aarch64-apple-ios-sim/release/libapp.a "$OUTPUT_DIR/simulator/libEguiApp_arm64.a" 2>/dev/null || true
    cp -f target/x86_64-apple-ios/release/libapp.a "$OUTPUT_DIR/simulator/libEguiApp_x86_64.a" 2>/dev/null || true
fi

if command -v xcodebuild &> /dev/null; then
    echo "[*] Generating EguiApp.xcframework via xcodebuild..."
    rm -rf "$OUTPUT_DIR/EguiApp.xcframework"
    xcodebuild -create-xcframework \
        -library "$OUTPUT_DIR/device/libEguiApp.a" \
        -library "$OUTPUT_DIR/simulator/libEguiApp.a" \
        -output "$OUTPUT_DIR/EguiApp.xcframework"
    echo "[+] Generated $OUTPUT_DIR/EguiApp.xcframework"
fi

echo "[+] iOS build completed successfully! Output: $OUTPUT_DIR"
