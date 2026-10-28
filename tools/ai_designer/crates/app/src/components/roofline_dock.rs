//! Roofline & VRAM Waterfall Dock: hardware ceilings, memory bandwidth, and operational intensity metrics.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{Color32, ProgressBar, RichText, ScrollArea, Ui};
use shared::analysis::roofline::RooflineRegime;

pub struct RooflineDockView;

impl Default for RooflineDockView {
    fn default() -> Self {
        Self
    }
}

impl RooflineDockView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().show(ui, |ui| {
            let r = &state.roofline_report;
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new("Analytical Hardware Roofline & VRAM Waterfall")
                        .size(15.0)
                        .strong(),
                );
                ui.label(format!("Target: {}", r.hardware_profile));
            });

            ui.add_space(4.0);
            ui.separator();

            // Key KPI metrics row
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.label(
                        RichText::new("Operational Intensity")
                            .size(11.0)
                            .color(Color32::from_rgb(180, 180, 180)),
                    );
                    ui.label(
                        RichText::new(format!("{:.1} FLOPs/B", r.operational_intensity))
                            .strong()
                            .size(14.0),
                    );
                    ui.label(format!("Machine Balance: {:.1} FLOPs/B", r.machine_balance));
                });

                ui.group(|ui| {
                    ui.label(
                        RichText::new("Execution Regime")
                            .size(11.0)
                            .color(Color32::from_rgb(180, 180, 180)),
                    );
                    let (regime_text, regime_color) = match r.regime {
                        RooflineRegime::ComputeBound => {
                            ("COMPUTE BOUND", Color32::from_rgb(120, 220, 140))
                        }
                        RooflineRegime::MemoryBandwidthBound => {
                            ("MEMORY BOUND", Color32::from_rgb(240, 160, 60))
                        }
                    };
                    ui.label(
                        RichText::new(regime_text)
                            .color(regime_color)
                            .strong()
                            .size(14.0),
                    );
                    ui.label(format!(
                        "Est. Step: {:.2} ms",
                        r.estimated_latency_per_step_ms
                    ));
                });

                ui.group(|ui| {
                    ui.label(
                        RichText::new("Critical Horizon (L*)")
                            .size(11.0)
                            .color(Color32::from_rgb(180, 180, 180)),
                    );
                    ui.label(
                        RichText::new(format!("{} tokens", r.critical_horizon_tokens))
                            .strong()
                            .size(14.0),
                    );
                    ui.label("Max sequence before OOM");
                });

                ui.group(|ui| {
                    ui.label(
                        RichText::new("VRAM Headroom")
                            .size(11.0)
                            .color(Color32::from_rgb(180, 180, 180)),
                    );
                    let headroom_gb = r.vram.headroom_bytes as f64 / 1e9;
                    let headroom_color = if r.vram.is_oom_risk {
                        Color32::from_rgb(240, 80, 80)
                    } else {
                        Color32::from_rgb(120, 220, 140)
                    };
                    ui.label(
                        RichText::new(format!("{:.2} GB", headroom_gb))
                            .color(headroom_color)
                            .strong()
                            .size(14.0),
                    );
                    ui.label(format!(
                        "Total Allocated: {:.2} GB",
                        r.vram.total_allocated_bytes as f64 / 1e9
                    ));
                });
            });

            ui.add_space(8.0);
            ui.label(
                RichText::new("VRAM Memory Allocation Breakdown")
                    .size(12.0)
                    .strong(),
            );

            let total_capacity = r.vram.max_hardware_vram_bytes as f32;
            let weights_frac = (r.vram.weights_bytes as f32 / total_capacity).clamp(0.0, 1.0);
            let opt_frac = (r.vram.optimizer_states_bytes as f32 / total_capacity).clamp(0.0, 1.0);
            let act_frac = (r.vram.activation_bytes as f32 / total_capacity).clamp(0.0, 1.0);

            let total_used_frac =
                (r.vram.total_allocated_bytes as f32 / total_capacity).clamp(0.0, 1.0);

            ui.add(
                ProgressBar::new(total_used_frac)
                    .show_percentage()
                    .animate(false),
            );

            ui.horizontal(|ui| {
                ui.label(format!(
                    "Weights: {:.2} GB ({:.1}%)",
                    r.vram.weights_bytes as f64 / 1e9,
                    weights_frac * 100.0
                ));
                ui.label("|");
                ui.label(format!(
                    "Optimizer: {:.2} GB ({:.1}%)",
                    r.vram.optimizer_states_bytes as f64 / 1e9,
                    opt_frac * 100.0
                ));
                ui.label("|");
                ui.label(format!(
                    "Activations: {:.2} GB ({:.1}%)",
                    r.vram.activation_bytes as f64 / 1e9,
                    act_frac * 100.0
                ));
            });
        });
    }

    pub fn render_pareto_explorer(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new("Multi-Objective Pareto Architecture Frontier")
                        .size(15.0)
                        .strong(),
                );
                ui.label(format!(
                    "(Target: {} • {} Candidates)",
                    state.spec.metadata.target_hardware.profile_name,
                    state.pareto_candidates.len()
                ));
            });
            ui.add_space(2.0);
            ui.label(
                RichText::new(
                    "Compare synthesized width, depth, and precision scaling variants across FLOPs/token, step latency, and VRAM budget:",
                )
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 210)),
            );
            ui.add_space(4.0);
            ui.separator();

            let mut candidate_to_apply: Option<shared::AiDesignSpec> = None;

            eframe::egui::Grid::new("pareto_frontier_grid")
                .striped(true)
                .num_columns(7)
                .spacing([18.0, 6.0])
                .show(ui, |ui| {
                    ui.label(RichText::new("Candidate Tier").strong());
                    ui.label(RichText::new("Parameters").strong());
                    ui.label(RichText::new("FLOPs / Token").strong());
                    ui.label(RichText::new("Est. Step Latency").strong());
                    ui.label(RichText::new("VRAM Footprint").strong());
                    ui.label(RichText::new("Pareto Score").strong());
                    ui.label(RichText::new("Action").strong());
                    ui.end_row();

                    for cand in &state.pareto_candidates {
                        ui.vertical(|ui| {
                            ui.label(RichText::new(&cand.label).strong());
                            ui.label(
                                RichText::new(&cand.description)
                                    .size(10.5)
                                    .color(Color32::from_rgb(165, 175, 190)),
                            );
                        });
                        ui.label(format!("{:.2} M", cand.total_param_count as f64 / 1e6));
                        ui.label(format!("{:.2} M", cand.flops_per_token as f64 / 1e6));
                        ui.label(format!("{:.2} ms", cand.latency_ms));

                        let vram_gb = cand.vram_bytes as f64 / 1e9;
                        let vram_color = if cand.is_oom_risk {
                            Color32::from_rgb(240, 80, 80)
                        } else {
                            Color32::from_rgb(120, 220, 140)
                        };
                        let vram_label = if cand.is_oom_risk {
                            format!("{:.2} GB (OOM)", vram_gb)
                        } else {
                            format!("{:.2} GB (OK)", vram_gb)
                        };
                        ui.label(RichText::new(vram_label).color(vram_color).strong());
                        ui.label(format!("{:.2}", cand.score));

                        if ui
                            .button(
                                RichText::new(format!("{} Apply Variant", bi::MAGIC))
                                    .color(Color32::from_rgb(100, 190, 255))
                                    .strong(),
                            )
                            .on_hover_text("Load this Pareto candidate into the active Studio canvas")
                            .clicked()
                        {
                            candidate_to_apply = Some(cand.spec.clone());
                        }
                        ui.end_row();
                    }
                });

            if let Some(new_spec) = candidate_to_apply {
                *state = StudioState::new_with_starter(new_spec);
            }
        });
    }
}
