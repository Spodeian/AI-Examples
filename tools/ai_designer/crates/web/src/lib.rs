//! Web entrypoint and WebRunner for the Serverless & Desktop Template.

#[cfg(target_arch = "wasm32")]
use app::TemplateApp;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, prelude::*};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    // Universal telemetry & logging initialization
    spodeian_telemetry::init_default();

    // Spawn the async eframe WebRunner natively
    wasm_bindgen_futures::spawn_local(async {
        let document = match web_sys::window().and_then(|win| win.document()) {
            Some(doc) => doc,
            None => {
                tracing::error!("Failed to obtain web window or document");
                return;
            }
        };
        let canvas = match document
            .get_element_by_id("egui_canvas")
            .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        {
            Some(canvas) => canvas,
            None => {
                tracing::error!("Canvas element 'egui_canvas' not found or invalid element type");
                return;
            }
        };

        let web_options = eframe::WebOptions::default();
        let runner = eframe::WebRunner::new();
        let _ = runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
            )
            .await;
    });
}
