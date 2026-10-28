//! Live debounced YAML blueprint source code editor with syntax error validation.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{Color32, RichText, ScrollArea, Ui};

pub struct CanvasYamlView;

impl Default for CanvasYamlView {
    fn default() -> Self {
        Self
    }
}

impl CanvasYamlView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ui.vertical(|ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new(format!("{} Blueprint YAML Specification (.yaml)", bi::BRACES))
                        .size(15.0)
                        .strong(),
                );
                ui.with_layout(
                    eframe::egui::Layout::right_to_left(eframe::egui::Align::Center),
                    |ui| {
                        if ui
                            .button(format!("{} Apply Changes", bi::CHECK_LG))
                            .clicked()
                        {
                            self.apply_yaml_changes(state);
                        }
                        if ui
                            .button(format!("{} Format / Revert", bi::ARROW_COUNTERCLOCKWISE))
                            .clicked()
                        {
                            state.yaml_buffer =
                                shared::format::aid_yaml::to_yaml_string(&state.spec)
                                    .unwrap_or_default();
                            state.yaml_error = None;
                        }
                    },
                );
            });

            if let Some(ref err) = state.yaml_error {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("{} {}", bi::EXCLAMATION_TRIANGLE_FILL, err))
                        .color(Color32::from_rgb(240, 100, 100)),
                );
            }

            ui.add_space(4.0);
            ui.separator();

            ScrollArea::vertical().show(ui, |ui| {
                ui.add(
                    eframe::egui::TextEdit::multiline(&mut state.yaml_buffer)
                        .font(eframe::egui::TextStyle::Monospace)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(35),
                );
            });
        });
    }

    fn apply_yaml_changes(&self, state: &mut StudioState) {
        match shared::format::aid_yaml::from_yaml_str(&state.yaml_buffer) {
            Ok(new_spec) => {
                state.spec = new_spec;
                state.yaml_error = None;
                state.auto_layout_nodes();
                state.refresh_analyses();
            }
            Err(e) => {
                state.yaml_error = Some(format!("YAML Syntax Error: {}", e));
            }
        }
    }
}
