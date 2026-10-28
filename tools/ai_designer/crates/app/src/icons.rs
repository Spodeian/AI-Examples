//! Bootstrap Icons v1.13.2 font integration and semantic icon constants for `egui`.
//!
//! Embeds `bootstrap-icons.ttf` (unpacked from official Bootstrap Icons v1.13.2) into
//! `egui::FontDefinitions` so that all Private Use Area (`\u{F101}`..`\u{F91E}`) glyphs
//! render crisply across both Native Desktop (Winit) and WebAssembly (Canvas/WebGPU).

use eframe::egui::{Context, FontData, FontDefinitions, FontFamily};
use std::sync::Arc;

/// Embedded TrueType font bytes for Bootstrap Icons v1.13.2.
pub const BOOTSTRAP_ICONS_TTF: &[u8] = include_bytes!("../assets/bootstrap-icons.ttf");

/// Font key registered in `egui::FontDefinitions`.
pub const BOOTSTRAP_FONT_NAME: &str = "bootstrap-icons";

/// Registers the Bootstrap Icons v1.13.2 font as a fallback for both Proportional and Monospace families.
pub fn install_bootstrap_icons(ctx: &Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        BOOTSTRAP_FONT_NAME.to_owned(),
        Arc::new(FontData::from_static(BOOTSTRAP_ICONS_TTF)),
    );

    if let Some(prop) = fonts.families.get_mut(&FontFamily::Proportional)
        && !prop.iter().any(|name| name == BOOTSTRAP_FONT_NAME)
    {
        prop.push(BOOTSTRAP_FONT_NAME.to_owned());
    }

    if let Some(mono) = fonts.families.get_mut(&FontFamily::Monospace)
        && !mono.iter().any(|name| name == BOOTSTRAP_FONT_NAME)
    {
        mono.push(BOOTSTRAP_FONT_NAME.to_owned());
    }

    ctx.set_fonts(fonts);
}

/// Official Bootstrap Icons v1.13.2 Unicode codepoints matching `bootstrap-icons.min.css`.
pub mod bi {
    // Brand & Navigation
    pub const LIGHTNING_CHARGE_FILL: &str = "\u{f46c}";
    pub const LIGHTNING_CHARGE: &str = "\u{f46d}";
    pub const CPU_FILL: &str = "\u{f2d5}";
    pub const CPU: &str = "\u{f2d6}";
    pub const GPU_CARD: &str = "\u{f6e2}";
    pub const DIAGRAM_3_FILL: &str = "\u{f2ed}";
    pub const DIAGRAM_3: &str = "\u{f2ee}";
    pub const DIAGRAM_2: &str = "\u{f2ec}";
    pub const COLLECTION: &str = "\u{f2cc}";
    pub const GRID_1X2: &str = "\u{f3f4}";
    pub const INFO_CIRCLE: &str = "\u{f431}";
    pub const INFO_CIRCLE_FILL: &str = "\u{f430}";
    pub const QUESTION_CIRCLE: &str = "\u{f505}";
    pub const SPEEDOMETER2: &str = "\u{f580}";
    pub const HDD_STACK: &str = "\u{f411}";
    pub const DATABASE: &str = "\u{f8c4}";
    pub const PALETTE2: &str = "\u{f4b2}";
    pub const SUN_FILL: &str = "\u{f5a1}";
    pub const MOON_STARS_FILL: &str = "\u{f495}";
    pub const CIRCLE_HALF: &str = "\u{f288}";

    // Workflow & Canvas Modes
    pub const MAGIC: &str = "\u{f675}";
    pub const STARS: &str = "\u{f589}";
    pub const BOUNDING_BOX: &str = "\u{f1b6}";
    pub const LIST_NESTED: &str = "\u{f474}";
    pub const CODE_SLASH: &str = "\u{f2c6}";
    pub const CODE_SQUARE: &str = "\u{f2c7}";
    pub const BRACES: &str = "\u{f1c9}";
    pub const SHIELD_CHECK: &str = "\u{f52f}";
    pub const SHIELD_EXCLAMATION: &str = "\u{f530}";
    pub const EXCLAMATION_TRIANGLE_FILL: &str = "\u{f33a}";
    pub const EXCLAMATION_TRIANGLE: &str = "\u{f33b}";
    pub const CHECK_CIRCLE_FILL: &str = "\u{f26a}";
    pub const CHECK_CIRCLE: &str = "\u{f26b}";
    pub const CHECK_LG: &str = "\u{f633}";
    pub const GRAPH_UP_ARROW: &str = "\u{f673}";
    pub const BAR_CHART_LINE: &str = "\u{f17c}";
    pub const PLAY_FILL: &str = "\u{f4f4}";
    pub const STOP_FILL: &str = "\u{f592}";
    pub const TERMINAL: &str = "\u{f5c3}";

    // Canvas HUD & Node Manipulation
    pub const ZOOM_IN: &str = "\u{f62c}";
    pub const ZOOM_OUT: &str = "\u{f62d}";
    pub const PLUS_LG: &str = "\u{f64d}";
    pub const PLUS_CIRCLE: &str = "\u{f4fa}";
    pub const DASH_LG: &str = "\u{f63b}";
    pub const ARROWS_FULLSCREEN: &str = "\u{f14d}";
    pub const CROSSHAIR: &str = "\u{f769}";
    pub const ARROW_UP: &str = "\u{f148}";
    pub const ARROW_DOWN: &str = "\u{f128}";
    pub const ARROW_LEFT: &str = "\u{f12f}";
    pub const ARROW_RIGHT: &str = "\u{f138}";
    pub const ARROW_COUNTERCLOCKWISE: &str = "\u{f117}";
    pub const COPY: &str = "\u{f759}";
    pub const FILES: &str = "\u{f3c2}";
    pub const TRASH3: &str = "\u{f78b}";
    pub const X_LG: &str = "\u{f659}";

    // File Formats, Import/Export & Links
    pub const DOWNLOAD: &str = "\u{f30a}";
    pub const UPLOAD: &str = "\u{f603}";
    pub const FOLDER2_OPEN: &str = "\u{f3d8}";
    pub const FILE_EARMARK_CODE: &str = "\u{f362}";
    pub const FILE_EARMARK_BINARY: &str = "\u{f35c}";
    pub const FILE_EARMARK_TEXT: &str = "\u{f38b}";
    pub const BOX_SEAM: &str = "\u{f1c7}";
    pub const ARCHIVE: &str = "\u{f10d}";
    pub const CLIPBOARD_CHECK: &str = "\u{f28b}";
    pub const BOOK: &str = "\u{f194}";
    pub const JOURNAL_CODE: &str = "\u{f43f}";
    pub const BOX_ARROW_UP_RIGHT: &str = "\u{f1c5}";
    pub const LINK_45DEG: &str = "\u{f470}";

    // Catalog, Filtering & Archetypes
    pub const SEARCH: &str = "\u{f52a}";
    pub const FUNNEL: &str = "\u{f3e1}";
    pub const FUNNEL_FILL: &str = "\u{f3e0}";
    pub const SLIDERS: &str = "\u{f56b}";
    pub const LAYERS: &str = "\u{f45b}";
    pub const LAYERS_FILL: &str = "\u{f459}";
    pub const ROBOT: &str = "\u{f6b1}";
    pub const EYE: &str = "\u{f341}";
    pub const ACTIVITY: &str = "\u{f66b}";
    pub const TABLE: &str = "\u{f5aa}";
    pub const SHUFFLE: &str = "\u{f544}";
    pub const BOXES: &str = "\u{f685}";
    pub const COMPASS: &str = "\u{f2d1}";
    pub const KEYBOARD: &str = "\u{f451}";
    pub const SHIELD_LOCK: &str = "\u{f538}";
    pub const ROCKET_TAKEOFF: &str = "\u{f845}";
    pub const BOX_ARROW_IN_RIGHT: &str = "\u{f1be}";
    pub const BOX_ARROW_RIGHT: &str = "\u{f1c3}";
}
