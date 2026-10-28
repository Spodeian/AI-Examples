//! Native desktop application runner and headless CLI for the AI Designer Suite.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cli;

use app::TemplateApp;
use eframe::NativeOptions;
use eframe::egui;
use spodeian_telemetry::init_default;

fn main() -> eframe::Result<()> {
    // 1. Check CLI arguments
    let args: Vec<String> = std::env::args().collect();
    match cli::run_cli(&args) {
        Ok(true) => {
            // CLI command executed successfully
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("[Error] {}", err);
            std::process::exit(1);
        }
        Ok(false) => {
            // No CLI command given -> proceed to launch egui Desktop Studio
        }
    }

    // 2. Universal telemetry & logging initialization
    init_default();

    let viewport = egui::ViewportBuilder::default()
        .with_title("AI Designer Studio & Training Suite")
        .with_inner_size(egui::vec2(1280.0, 800.0))
        .with_min_inner_size(egui::vec2(800.0, 600.0));

    let options = NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "AI Designer Studio & Training Suite",
        options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
    )
}
