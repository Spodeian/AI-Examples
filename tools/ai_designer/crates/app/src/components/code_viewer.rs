//! Multi-Backend Code Viewer modal: displays emitted code for PyTorch 2, JAX, Candle, WebGPU, and test harnesses.

use crate::icons::bi;
use crate::storage_manager::trigger_text_download;
use crate::studio_state::{CodeTab, StudioState};
use eframe::egui::{self, Color32, RichText, ScrollArea, Window};

pub struct CodeViewerView;

impl Default for CodeViewerView {
    fn default() -> Self {
        Self
    }
}

impl CodeViewerView {
    pub fn render(&mut self, ctx: &egui::Context, state: &mut StudioState) {
        if !state.show_code_modal {
            return;
        }

        let mut is_open = state.show_code_modal;

        Window::new("Multi-Backend Target Code Generator")
            .open(&mut is_open)
            .default_size([900.0, 600.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(state.active_code_tab == CodeTab::PyTorch2, "PyTorch 2.x")
                        .clicked()
                    {
                        state.active_code_tab = CodeTab::PyTorch2;
                    }
                    if ui
                        .selectable_label(
                            state.active_code_tab == CodeTab::JAXEquinox,
                            "JAX / Equinox",
                        )
                        .clicked()
                    {
                        state.active_code_tab = CodeTab::JAXEquinox;
                    }
                    if ui
                        .selectable_label(
                            state.active_code_tab == CodeTab::RustCandle,
                            "Rust Candle",
                        )
                        .clicked()
                    {
                        state.active_code_tab = CodeTab::RustCandle;
                    }
                    if ui
                        .selectable_label(
                            state.active_code_tab == CodeTab::WebGPUWGSL,
                            "WebGPU (WGSL)",
                        )
                        .clicked()
                    {
                        state.active_code_tab = CodeTab::WebGPUWGSL;
                    }
                    if ui
                        .selectable_label(
                            state.active_code_tab == CodeTab::TritonOps,
                            "OpenAI Triton",
                        )
                        .clicked()
                    {
                        state.active_code_tab = CodeTab::TritonOps;
                    }
                    if ui
                        .selectable_label(
                            state.active_code_tab == CodeTab::CrossBackendHarness,
                            "Parity Harness",
                        )
                        .clicked()
                    {
                        state.active_code_tab = CodeTab::CrossBackendHarness;
                    }
                });

                let (active_filename, code_content) = if let Some(ref bundle) = state.generated_bundle {
                    match state.active_code_tab {
                        CodeTab::PyTorch2 => (
                            "train.py",
                            bundle
                                .files
                                .get("pytorch/train.py")
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        CodeTab::JAXEquinox => (
                            "jax_train.py",
                            bundle
                                .files
                                .get("jax/train.py")
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        CodeTab::RustCandle => (
                            "model.rs",
                            bundle
                                .files
                                .get("candle/src/model.rs")
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        CodeTab::WebGPUWGSL => (
                            "shaders.wgsl",
                            bundle
                                .files
                                .get("kernels/shaders.wgsl")
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        CodeTab::TritonOps => (
                            "triton_ops.py",
                            bundle
                                .files
                                .get("kernels/triton_ops.py")
                                .cloned()
                                .unwrap_or_default(),
                        ),
                        CodeTab::CrossBackendHarness => (
                            "verify_and_bench.py",
                            bundle
                                .files
                                .get("verify_and_bench.py")
                                .cloned()
                                .unwrap_or_default(),
                        ),
                    }
                } else {
                    (
                        "README.txt",
                        "No code generated yet. Click 'Generate Target Code' in the studio toolbar."
                            .to_string(),
                    )
                };

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(format!("{} Copy File", bi::COPY))
                        .on_hover_text("Copy active source file to system clipboard")
                        .clicked()
                    {
                        ui.ctx().copy_text(code_content.clone());
                        state.code_status_message =
                            Some(format!("{} Copied '{active_filename}' to clipboard", bi::CHECK_LG));
                    }

                    if ui
                        .button(format!("{} Download File", bi::DOWNLOAD))
                        .on_hover_text("Download active source file")
                        .clicked()
                    {
                        let dl_name = format!(
                            "{}_{}",
                            state.spec.metadata.model_id, active_filename
                        );
                        trigger_text_download(&dl_name, &code_content, "text/plain;charset=utf-8");
                        state.code_status_message =
                            Some(format!("{} Triggered download for '{dl_name}'", bi::CHECK_LG));
                    }

                    if ui
                        .button(
                            RichText::new(format!("{} Save Bundle to Disk", bi::FOLDER2_OPEN))
                                .color(Color32::from_rgb(120, 220, 140))
                                .strong(),
                        )
                        .on_hover_text("Write complete multi-backend project directory to exported_models/<model_id>/")
                        .clicked()
                    {
                        if let Some(ref bundle) = state.generated_bundle {
                            let export_dir =
                                std::path::Path::new("exported_models").join(&bundle.model_id);
                            match bundle.save_to_disk(&export_dir) {
                                Ok(()) => {
                                    state.code_status_message = Some(format!(
                                        "{} Saved {} files to {}",
                                        bi::CHECK_LG,
                                        bundle.files.len(),
                                        export_dir.display()
                                    ));
                                }
                                Err(e) => {
                                    state.code_status_message = Some(format!(
                                        "{} Save failed (use Download File on Web): {e}",
                                        bi::EXCLAMATION_TRIANGLE_FILL
                                    ));
                                }
                            }
                        }
                    }

                    if let Some(ref msg) = state.code_status_message {
                        ui.label(
                            RichText::new(msg)
                                .size(11.5)
                                .color(Color32::from_rgb(120, 220, 140)),
                        );
                    }
                });

                ui.separator();

                ScrollArea::vertical().show(ui, |ui| {
                    let mut mut_code = code_content;
                    ui.add(
                        egui::TextEdit::multiline(&mut mut_code)
                            .font(egui::TextStyle::Monospace)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(30),
                    );
                });
            });

        state.show_code_modal = is_open;
    }
}
