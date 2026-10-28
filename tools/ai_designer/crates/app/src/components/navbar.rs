//! Top navigation bar and global header controls using Bootstrap Icons v1.13.2.

use crate::{ScreenConstraints, TemplateApp, icons::bi, storage_manager::*};
use eframe::egui;

pub fn render_navbar(app: &mut TemplateApp, ui: &mut egui::Ui, constraints: &ScreenConstraints) {
    egui::Panel::top("top_panel").show(ui, |ui| {
        ui.add_space(4.0);
        let title_text = if constraints.is_mobile {
            format!("{} AI Designer", bi::CPU_FILL)
        } else {
            format!("{} AI Designer Studio", bi::CPU_FILL)
        };
        let header_row_height = if constraints.is_mobile { 44.0 } else { 32.0 };

        ui.horizontal(|ui| {
            ui.set_height(header_row_height);

            if constraints.is_mobile {
                ui.label(egui::RichText::new(title_text).size(17.0).strong());
            } else {
                ui.heading(title_text);
            }

            ui.add_space(8.0);
            if ui
                .selectable_label(
                    app.state.config.active_page == shared::AppPage::Designer,
                    format!("{} Studio Workbench", bi::DIAGRAM_3_FILL),
                )
                .on_hover_text("Interactive Dataflow Graph, Outliner, YAML, Advisor, Roofline & Codegen")
                .clicked()
            {
                app.state.config.active_page = shared::AppPage::Designer;
                app.persist_state();
            }

            if ui
                .selectable_label(
                    app.state.config.active_page == shared::AppPage::Dashboard,
                    format!("{} Blueprint Gallery (20)", bi::COLLECTION),
                )
                .on_hover_text("Browse all 20 Starter Blueprints across 18 AI-Examples Categories & Experiment Notes")
                .clicked()
            {
                app.state.config.active_page = shared::AppPage::Dashboard;
                app.persist_state();
            }

            if ui
                .selectable_label(
                    app.state.config.active_page == shared::AppPage::About,
                    format!("{} Docs & Architecture", bi::INFO_CIRCLE),
                )
                .on_hover_text("5-Stage Workflow Guide, System Architecture & Keyboard Shortcuts")
                .clicked()
            {
                app.state.config.active_page = shared::AppPage::About;
                app.persist_state();
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let theme_icon = if app.state.config.theme.is_high_contrast() {
                    bi::CIRCLE_HALF
                } else {
                    bi::PALETTE2
                };
                let theme_text = if constraints.is_mobile {
                    format!("{theme_icon} {}", app.state.config.theme.label())
                } else {
                    format!("{theme_icon} Theme: {}", app.state.config.theme.label())
                };

                if ui
                    .button(theme_text)
                    .on_hover_text("Cycle visual themes: Dark, Warm Light, High Contrast (Dark), and High Contrast (Light)")
                    .clicked()
                {
                    app.state.config.theme = app.state.config.theme.next();
                    app.persist_state();
                }

                if ui
                    .button(format!("{} Help", bi::QUESTION_CIRCLE))
                    .on_hover_text("5-Stage AI Architecture Workflow, shortcuts & documentation")
                    .clicked()
                {
                    app.show_help_dialog = true;
                }

                if ui
                    .button(format!("{} Telemetry", bi::SPEEDOMETER2))
                    .on_hover_text("View live studio subsystem performance & active blueprint telemetry")
                    .clicked()
                {
                    app.show_telemetry_modal = true;
                }

                // Storage diagnostics button
                let storage_text = match app.storage_diag.is_persisted {
                    Some(true) => {
                        if constraints.is_mobile {
                            format!("{} Storage (P)", bi::HDD_STACK)
                        } else {
                            format!("{} Storage: Persistent", bi::HDD_STACK)
                        }
                    }
                    Some(false) => {
                        if constraints.is_mobile {
                            format!("{} Storage (E)", bi::HDD_STACK)
                        } else {
                            format!("{} Storage: Ephemeral", bi::HDD_STACK)
                        }
                    }
                    None => format!("{} Storage", bi::HDD_STACK),
                };
                if ui
                    .button(storage_text)
                    .on_hover_text("Multi-tier storage persistence & backups")
                    .clicked()
                {
                    app.show_storage_modal = true;
                }

                // PWA Install Button
                if app.storage_diag.pwa_install_available
                    && !app.storage_diag.is_pwa_installed
                    && ui
                        .button(format!("{} Install App", bi::DOWNLOAD))
                        .on_hover_text("Install application for permanent offline storage")
                        .clicked()
                {
                    trigger_pwa_install();
                }

                if ui
                    .button(format!("{} Import", bi::UPLOAD))
                    .on_hover_text("Import YAML/JSON blueprint specification or backup data")
                    .clicked()
                {
                    app.show_import_dialog = true;
                    app.import_text_buffer.clear();
                    app.import_result_message = None;
                }
            });
        });
        ui.add_space(4.0);
    });
}
