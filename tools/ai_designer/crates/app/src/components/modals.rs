//! Modal dialogs, warning banners, and data transfer views.

use crate::{ExportFormat, TemplateApp, icons::bi, storage_manager::*, studio_state::StudioState};
use eframe::egui;
use shared::{
    ItemCollection, export_to_compressed_bson, import_from_compressed_bson, import_from_csv,
    import_from_json,
};

pub fn render_warning_banners(app: &mut TemplateApp, ctx: &egui::Context) {
    let is_ephemeral = app.storage_diag.is_persisted == Some(false);
    let is_quota = app.storage_diag.quota_exceeded;

    let show_combined = is_ephemeral && is_quota && !app.dismissed_combined_warning;
    let show_ephemeral = is_ephemeral && !app.dismissed_ephemeral_warning;
    let show_quota = is_quota && !app.dismissed_quota_warning;

    if show_combined || show_ephemeral || show_quota {
        let (msg, fill_color, stroke_color, text_color) = if show_combined {
            (
                "Storage Alert: Storage is Ephemeral AND Quota Limit Exceeded!",
                egui::Color32::from_rgba_premultiplied(35, 20, 20, 245),
                egui::Color32::from_rgb(240, 80, 80),
                egui::Color32::from_rgb(255, 120, 120),
            )
        } else if show_ephemeral {
            (
                "Ephemeral Storage: Browser may clear local data under storage pressure.",
                egui::Color32::from_rgba_premultiplied(35, 28, 15, 245),
                egui::Color32::from_rgb(220, 160, 30),
                egui::Color32::from_rgb(255, 200, 80),
            )
        } else {
            (
                "Storage Quota Exceeded: State migrated to IndexedDB fallback tier.",
                egui::Color32::from_rgba_premultiplied(35, 28, 15, 245),
                egui::Color32::from_rgb(220, 160, 30),
                egui::Color32::from_rgb(255, 200, 80),
            )
        };

        egui::Area::new(egui::Id::new("storage_warning_banner_area"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -20.0))
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(fill_color)
                    .stroke(egui::Stroke::new(1.0_f32, stroke_color))
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(16, 10))
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new(msg).strong().color(text_color));
                            ui.add_space(6.0);
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center)
                                    .with_main_align(egui::Align::Center),
                                |ui| {
                                    if ui.button("Save .bson Backup").clicked() {
                                        if let Ok(bytes) =
                                            export_to_compressed_bson(&app.state.collection)
                                        {
                                            trigger_binary_download(
                                                "data_backup.bson",
                                                &bytes,
                                                "application/octet-stream",
                                            );
                                        }
                                    }
                                    if ui.button("Request Persistence").clicked() {
                                        request_persistent_storage();
                                    }
                                    if ui.button("Dismiss").clicked() {
                                        if show_combined {
                                            app.dismissed_combined_warning = true;
                                        } else if show_ephemeral {
                                            app.dismissed_ephemeral_warning = true;
                                        } else {
                                            app.dismissed_quota_warning = true;
                                        }
                                    }
                                },
                            );
                        });
                    });
            });
    }
}

pub fn render_dialogs(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if app.show_help_dialog {
        let mut open = true;
        let win_w = (ui.available_width() - 24.0).clamp(340.0, 600.0);
        let win_h = (ui.available_height() - 32.0).clamp(440.0, 680.0);

        egui::Window::new(format!(
            "{} AI Designer Studio — Workflow & Help",
            bi::QUESTION_CIRCLE
        ))
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(320.0)
        .min_height(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(format!(
                        "{} AI Designer Studio & Training Suite",
                        bi::CPU_FILL
                    ));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                .strong()
                                .color(ui.visuals().hyperlink_color),
                        );
                    });
                });
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading(format!("{} 5-Stage Architecture Design Workflow", bi::COMPASS));
                ui.add_space(4.0);
                ui.label("1. Scaffold: Click 'New / Wizard' to choose from 9 Guided Archetypes, 20 Starter Blueprints across 8 scientific domains, or a Blank Canvas.");
                ui.label("2. Compose: Add operators from the Left Palette (with Category & Paradigm filters), inspect/reorder/duplicate layers in the Right Inspector, and view Input -> Layer -> Output dataflow on the Canvas.");
                ui.label("3. Diagnose & Pareto: Inspect 55+ mathematical/hardware rules with 1-click 'Quick Fix' in the Advisor dock, review VRAM breakdown in Roofline, and apply optimal scaling variants in Pareto Frontier.");
                ui.label("4. Verify & Train: Click 'Test Parity & Condition' for deterministic Seed-42 activation verification and 'Run Verification & Training' for live loss monitoring.");
                ui.label("5. Export & Codegen: Click 'Generate Target Code' to emit PyTorch 2, JAX/Equinox, Rust Candle, WebGPU WGSL, Triton, and Parity Harness files, or export native .aid (binary), .yaml, or .aidpkg.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading(format!("{} Canvas Controls & Keyboard Shortcuts", bi::KEYBOARD));
                ui.add_space(4.0);
                ui.label("• Pan Canvas: Middle-Mouse Drag or Alt + Left-Drag.");
                ui.label("• Zoom Canvas: Mouse Scroll Wheel or HUD - / 100% / + buttons.");
                ui.label("• Auto-Layout: Press Home or click 'Auto-Layout' in the Canvas HUD.");
                ui.label("• Layer Actions: Select a node to Move Up, Move Down, Duplicate, or Delete (Del / Backspace).");
                ui.label("• Close Dialogs / Deselect: Press Escape.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading(format!("{} Multi-Tier Storage & Offline Privacy", bi::HDD_STACK));
                ui.add_space(4.0);
                ui.label("• Tier 1: Fast synchronous local storage.");
                ui.label("• Tier 2: Asynchronous IndexedDB extended quota fallback.");
                ui.label(
                    egui::RichText::new("100% Client-Side: Runs natively on Desktop or offline in Browser WASM.")
                        .color(egui::Color32::from_rgb(80, 160, 90))
                        .strong(),
                );
            });
        });
        if !open {
            app.show_help_dialog = false;
        }
    }

    if app.show_reset_dialog {
        egui::Window::new("Reset Data?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label("Are you sure you want to reset all items to default sample data?");
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Yes, Reset").clicked() {
                        app.state.collection = ItemCollection::default();
                        app.show_reset_dialog = false;
                        app.persist_state();
                    }
                    if ui.button("Cancel").clicked() {
                        app.show_reset_dialog = false;
                    }
                });
            });
    }

    if let Some(format) = app.show_export_dialog {
        let mut open = true;
        egui::Window::new(format!("Export {}", format.label()))
            .open(&mut open)
            .default_size(egui::vec2(540.0, 380.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Copy to Clipboard").clicked() {
                        ui.ctx().copy_text(app.export_text_buffer.clone());
                        app.export_copied_notification = Some(ui.input(|i| i.time));
                    }
                    if let Some(t) = app.export_copied_notification
                        && ui.input(|i| i.time) - t < 3.0
                    {
                        ui.label(
                            egui::RichText::new("Copied to clipboard!").color(egui::Color32::GREEN),
                        );
                    }

                    ui.separator();

                    if ui.button("Download File").clicked() {
                        match format {
                            ExportFormat::Json => {
                                trigger_text_download(
                                    "data_export.json",
                                    &app.export_text_buffer,
                                    "application/json;charset=utf-8",
                                );
                            }
                            ExportFormat::Csv => {
                                trigger_text_download(
                                    "data_export.csv",
                                    &app.export_text_buffer,
                                    "text/csv;charset=utf-8",
                                );
                            }
                            ExportFormat::Bson => {
                                if let Ok(bytes) = export_to_compressed_bson(&app.state.collection)
                                {
                                    trigger_binary_download(
                                        "data_backup.bson",
                                        &bytes,
                                        "application/octet-stream",
                                    );
                                }
                            }
                        }
                    }
                });

                ui.add_space(8.0);
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.export_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .code_editor()
                            .lock_focus(true)
                            .desired_width(f32::INFINITY),
                    );
                });
            });

        if !open {
            app.show_export_dialog = None;
            app.export_text_buffer.clear();
        }
    }

    if app.show_import_dialog {
        let mut open = true;
        egui::Window::new(format!("{} Import Blueprint or Data", bi::UPLOAD))
            .open(&mut open)
            .default_size(egui::vec2(540.0, 380.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label(
                    "Paste a Blueprint Specification (.yaml, .json, .toml, .ron) or Collection Backup (JSON, CSV, Base64 BSON):",
                );
                ui.add_space(8.0);

                egui::ScrollArea::both().max_height(210.0).show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.import_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .hint_text("Paste .yaml / .json AI blueprint specification or backup data here...")
                            .desired_width(f32::INFINITY)
                            .desired_rows(9),
                    );
                });

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(format!("{} Apply Import", bi::CHECK_LG))
                        .clicked()
                    {
                        let input = app.import_text_buffer.trim();
                        if input.is_empty() {
                            app.import_result_message = Some(Err("Input is empty.".to_string()));
                        } else if let Ok(spec) =
                            shared::format::load_blueprint_from_bytes(input.as_bytes(), None)
                            && (!spec.inputs.is_empty() || !spec.graph.nodes.is_empty())
                        {
                            let name = spec.metadata.display_name.clone();
                            app.studio = StudioState::new_with_starter(spec);
                            app.state.config.active_page = shared::AppPage::Designer;
                            app.import_result_message = Some(Ok(format!(
                                "Successfully loaded AI Blueprint '{name}' into Studio!"
                            )));
                            app.persist_state();
                        } else if input.starts_with('{') {
                            match import_from_json(input) {
                                Ok(col) => {
                                    let count = col.total_count();
                                    app.state.collection = col;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported {} items from JSON!",
                                        count
                                    )));
                                    app.persist_state();
                                }
                                Err(e) => {
                                    app.import_result_message = Some(Err(e.to_string()));
                                }
                            }
                        } else if let Ok(decoded_bytes) = base64::Engine::decode(
                            &base64::engine::general_purpose::STANDARD,
                            input,
                        ) {
                            match import_from_compressed_bson(&decoded_bytes) {
                                Ok(col) => {
                                    let count = col.total_count();
                                    app.state.collection = col;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported {} items from compressed BSON!",
                                        count
                                    )));
                                    app.persist_state();
                                }
                                Err(e) => {
                                    app.import_result_message =
                                        Some(Err(format!("BSON import failed: {}", e)));
                                }
                            }
                        } else {
                            match import_from_csv(input) {
                                Ok(col) => {
                                    let count = col.total_count();
                                    app.state.collection = col;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported {} items from CSV!",
                                        count
                                    )));
                                    app.persist_state();
                                }
                                Err(e) => {
                                    app.import_result_message = Some(Err(e.to_string()));
                                }
                            }
                        }
                    }

                    if ui.button("Cancel").clicked() {
                        app.show_import_dialog = false;
                        app.import_text_buffer.clear();
                        app.import_result_message = None;
                    }
                });

                if let Some(ref result) = app.import_result_message {
                    ui.add_space(6.0);
                    match result {
                        Ok(msg) => {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::from_rgb(80, 180, 90))
                                    .strong(),
                            );
                        }
                        Err(msg) => {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::from_rgb(220, 70, 70))
                                    .strong(),
                            );
                        }
                    }
                }
            });

        if !open {
            app.show_import_dialog = false;
            app.import_text_buffer.clear();
            app.import_result_message = None;
        }
    }

    if app.show_storage_modal {
        render_storage_modal(app, ui);
    }
}

pub fn render_storage_modal(app: &mut TemplateApp, ui: &mut egui::Ui) {
    let mut open = true;
    egui::Window::new("Storage & Data Management")
        .open(&mut open)
        .default_size(egui::vec2(480.0, 380.0))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            ui.heading("Storage Diagnostics");
            ui.add_space(4.0);

            let status_color = match app.storage_diag.is_persisted {
                Some(true) => egui::Color32::from_rgb(80, 200, 100),
                Some(false) => egui::Color32::from_rgb(240, 160, 50),
                None => egui::Color32::GRAY,
            };
            let status_label = match app.storage_diag.is_persisted {
                Some(true) => "Persistent (Immune to browser eviction)",
                Some(false) => "Ephemeral (May be cleared if disk is low)",
                None => "Unknown / Querying...",
            };

            ui.horizontal(|ui| {
                ui.label("Persistence Status:");
                ui.colored_label(status_color, egui::RichText::new(status_label).strong());
            });

            ui.horizontal(|ui| {
                ui.label("Active Storage Tier:");
                ui.label(egui::RichText::new(app.storage_diag.backend.label()).strong());
            });

            ui.horizontal(|ui| {
                ui.label("In-Memory Working Set:");
                ui.label(format!(
                    "{} items (Capacity limit: {} items)",
                    app.state.collection.items.len(),
                    app.storage_diag.memory_capacity_limit
                ));
            });

            ui.horizontal(|ui| {
                ui.label("IndexedDB Fallback:");
                let fallback_label =
                    if app.storage_diag.idb_active || app.storage_diag.fallback_tier_active {
                        "Active (Prevents RAM Overfilling)"
                    } else {
                        "Synchronized & Available"
                    };
                ui.colored_label(
                    egui::Color32::from_rgb(80, 200, 100),
                    egui::RichText::new(fallback_label).strong(),
                );
            });

            ui.horizontal(|ui| {
                ui.label("PWA Installation:");
                if app.storage_diag.is_pwa_installed {
                    ui.colored_label(
                        egui::Color32::from_rgb(80, 200, 100),
                        "Installed (Permanent App)",
                    );
                } else if app.storage_diag.pwa_install_available {
                    ui.label("Available to Install");
                } else {
                    ui.label("Not Available in Browser Tab");
                }
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);

            ui.heading("Actions");
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if app.storage_diag.is_persisted != Some(true) {
                    if ui
                        .button("Request Persistent Storage")
                        .on_hover_text("Ask browser permission to protect state from auto-eviction")
                        .clicked()
                    {
                        request_persistent_storage();
                    }
                }

                if app.storage_diag.pwa_install_available && !app.storage_diag.is_pwa_installed {
                    if ui
                        .button("Install Web App")
                        .on_hover_text("Install to homescreen/desktop for highest durability")
                        .clicked()
                    {
                        trigger_pwa_install();
                    }
                }
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .button("Export Compressed BSON Backup")
                    .on_hover_text("Download compact, offline state backup (.bson)")
                    .clicked()
                {
                    if let Ok(bytes) = export_to_compressed_bson(&app.state.collection) {
                        trigger_binary_download(
                            "data_backup.bson",
                            &bytes,
                            "application/octet-stream",
                        );
                    }
                }

                if ui.button("Import Backup").clicked() {
                    app.show_storage_modal = false;
                    app.show_import_dialog = true;
                }
            });
        });

    if !open {
        app.show_storage_modal = false;
    }
}
