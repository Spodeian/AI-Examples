//! Hierarchical Outliner Tree View: displays Inputs, Layers, Topology, Training, and Outputs in a structured list.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{Color32, RichText, ScrollArea, Ui};

pub struct CanvasTreeView;

impl Default for CanvasTreeView {
    fn default() -> Self {
        Self
    }
}

impl CanvasTreeView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(8.0);
            ui.heading(
                RichText::new(format!("{} Architecture Hierarchy Outliner", bi::LIST_NESTED))
                    .size(16.0)
                    .strong(),
            );
            ui.add_space(4.0);
            ui.label(
                RichText::new(
                    "Structured tree navigation of ports, layers, wiring, and loss pairings.",
                )
                .size(12.0)
                .color(Color32::from_rgb(180, 180, 180)),
            );
            ui.separator();

            // 1. Inputs Section
            eframe::egui::CollapsingHeader::new(
                RichText::new(format!(
                    "{} Input Streams ({})",
                    bi::BOX_ARROW_IN_RIGHT,
                    state.spec.inputs.len()
                ))
                .strong(),
            )
            .default_open(true)
            .show(ui, |ui| {
                for input in &state.spec.inputs {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(bi::ARROW_RIGHT).color(Color32::from_rgb(100, 180, 255)),
                        );
                        ui.label(RichText::new(&input.id).strong());
                        ui.label(format!(
                            "({:?} | Norm: {:?} | Enc: {:?})",
                            input.space, input.normalizer, input.coordinate_encoding
                        ));
                    });
                }
            });

            ui.add_space(8.0);

            // 2. Topology & Wiring
            ui.collapsing(RichText::new("Wiring Topology").strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Pattern:");
                    ui.label(format!("{:?}", state.spec.graph.topology));
                });
            });

            ui.add_space(8.0);

            // 3. Sequential / Graph Nodes
            eframe::egui::CollapsingHeader::new(
                RichText::new(format!(
                    "{} Computation Layers ({})",
                    bi::LAYERS,
                    state.spec.graph.nodes.len()
                ))
                .strong(),
            )
            .default_open(true)
            .show(ui, |ui| {
                let mut to_select = None;
                let mut action = None; // 1 = up, 2 = down, 3 = dup, 4 = del

                for (idx, node) in state.spec.graph.nodes.iter().enumerate() {
                    let is_selected = state.selected_node_id.as_deref() == Some(&node.id);
                    let prec = node.precision.unwrap_or(shared::PrecisionFormat::BF16);
                    let label = format!(
                        "{}. {} ({}) [{:?}]",
                        idx + 1,
                        node.name,
                        node.id,
                        prec
                    );

                    ui.horizontal(|ui| {
                        if ui.selectable_label(is_selected, label).clicked() {
                            to_select = Some(node.id.clone());
                        }
                        if is_selected {
                            if ui
                                .small_button(bi::ARROW_UP)
                                .on_hover_text("Move Layer Up")
                                .clicked()
                            {
                                action = Some(1);
                            }
                            if ui
                                .small_button(bi::ARROW_DOWN)
                                .on_hover_text("Move Layer Down")
                                .clicked()
                            {
                                action = Some(2);
                            }
                            if ui
                                .small_button(bi::COPY)
                                .on_hover_text("Duplicate Layer")
                                .clicked()
                            {
                                action = Some(3);
                            }
                            if ui
                                .small_button(
                                    RichText::new(bi::TRASH3).color(Color32::from_rgb(240, 100, 100)),
                                )
                                .on_hover_text("Delete Layer")
                                .clicked()
                            {
                                action = Some(4);
                            }
                        }
                    });
                }
                if let Some(id) = to_select {
                    state.select_node(Some(id));
                }
                match action {
                    Some(1) => {
                        state.move_selected_node(-1);
                    }
                    Some(2) => {
                        state.move_selected_node(1);
                    }
                    Some(3) => {
                        state.duplicate_selected_node();
                    }
                    Some(4) => {
                        state.delete_selected_node();
                    }
                    _ => {}
                }
            });

            ui.add_space(8.0);

            // 4. Output Ports
            eframe::egui::CollapsingHeader::new(
                RichText::new(format!(
                    "{} Output Streams ({})",
                    bi::BOX_ARROW_RIGHT,
                    state.spec.outputs.len()
                ))
                .strong(),
            )
            .default_open(true)
            .show(ui, |ui| {
                for output in &state.spec.outputs {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(bi::ARROW_LEFT).color(Color32::from_rgb(120, 220, 140)),
                        );
                        ui.label(RichText::new(&output.id).strong());
                        ui.label(format!(
                            "Loss: {:?} | Constraint: {:?}",
                            output.loss, output.constraint
                        ));
                    });
                }
            });

            ui.add_space(8.0);

            // 5. Training Regime
            ui.collapsing(
                RichText::new("Training Regime & Partitioning").strong(),
                |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Muon LR:");
                        ui.label(format!("{}", state.spec.training.muon.lr));
                        ui.label("| AdamW LR:");
                        ui.label(format!("{}", state.spec.training.adamw.lr));
                    });
                    ui.horizontal(|ui| {
                        ui.label("AutoClip:");
                        ui.label(format!("{}", state.spec.training.hygiene.auto_clip));
                        ui.label("| Grokfast:");
                        ui.label(format!("{}", state.spec.training.hygiene.grokfast));
                    });
                },
            );
        });
    }
}
