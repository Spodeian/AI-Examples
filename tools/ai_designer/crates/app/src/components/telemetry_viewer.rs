//! System Telemetry Dashboard Viewer Component.
//!
//! Provides a properly proportioned telemetry dashboard modal surfacing real-time studio performance,
//! active architecture footprint, analytical engine latency, and subsystem health metrics.

use crate::{TemplateApp, icons::bi};
use eframe::egui::{self, Color32, RichText};
use spodeian_web_utils::copy_to_clipboard;

pub fn render_telemetry_modal(app: &mut TemplateApp, ctx: &egui::Context) {
    if !app.show_telemetry_modal {
        return;
    }

    let mut is_open = app.show_telemetry_modal;
    egui::Window::new(format!("{} Studio Performance & Architecture Telemetry", bi::SPEEDOMETER2))
        .open(&mut is_open)
        .collapsible(false)
        .resizable(true)
        .default_size([760.0, 480.0])
        .min_size([620.0, 380.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.add_space(4.0);

            // Header status banner
            ui.horizontal(|ui| {
                ui.heading(RichText::new(&app.telemetry_data.title).size(16.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (icon, badge_text, badge_color) = if app.telemetry_data.success_rate > 99.0 {
                        (
                            bi::CHECK_CIRCLE_FILL,
                            "System Operational",
                            Color32::from_rgb(70, 190, 110),
                        )
                    } else {
                        (
                            bi::EXCLAMATION_TRIANGLE_FILL,
                            "Degraded",
                            Color32::from_rgb(220, 150, 40),
                        )
                    };
                    ui.colored_label(badge_color, format!("{icon} {badge_text}"));
                });
            });

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(6.0);

            // Row 1: 4 Equal-Width KPI Cards for Runtime Health
            ui.label(
                RichText::new(format!("{} Runtime & Engine KPIs", bi::ACTIVITY))
                    .size(12.5)
                    .strong(),
            );
            ui.add_space(4.0);
            ui.columns(4, |cols| {
                render_kpi_card(
                    &mut cols[0],
                    bi::SPEEDOMETER2,
                    "Session Uptime",
                    &format!("{} s", app.telemetry_data.uptime_secs),
                    "Active Studio Session",
                );
                render_kpi_card(
                    &mut cols[1],
                    bi::LIGHTNING_CHARGE_FILL,
                    "Engine Passes",
                    &format!("{}", app.telemetry_data.total_operations),
                    "Shape + Advisor Evals",
                );
                render_kpi_card(
                    &mut cols[2],
                    bi::SHIELD_CHECK,
                    "Reliability SLA",
                    &format!("{:.2}%", app.telemetry_data.success_rate),
                    "Zero-Fault Execution",
                );
                render_kpi_card(
                    &mut cols[3],
                    bi::CPU,
                    "Throughput",
                    &format!("{:.0} ops/s", app.telemetry_data.throughput_ops_sec),
                    &format!("RAM: {:.1} MB", app.telemetry_data.memory_mb),
                );
            });

            ui.add_space(10.0);

            // Row 2: 4 Equal-Width KPI Cards for Active Model Architecture Telemetry
            ui.label(
                RichText::new(format!(
                    "{} Active Blueprint Telemetry ({})",
                    bi::DIAGRAM_3,
                    app.studio.spec.metadata.display_name
                ))
                .size(12.5)
                .strong(),
            );
            ui.add_space(4.0);
            let params_m = app.studio.partition_report.total_param_count as f64 / 1e6;
            let flops_m = app.studio.roofline_report.total_flops_per_token as f64 / 1e6;
            let vram_gb = app.studio.roofline_report.vram.total_allocated_bytes as f64 / 1e9;
            let nodes_len = app.studio.spec.graph.nodes.len();
            let findings_len = app.studio.advisor_findings.len();

            ui.columns(4, |cols| {
                render_kpi_card(
                    &mut cols[0],
                    bi::LAYERS,
                    "Graph Topology",
                    &format!("{nodes_len} Layers"),
                    &format!(
                        "{} In / {} Out Ports",
                        app.studio.spec.inputs.len(),
                        app.studio.spec.outputs.len()
                    ),
                );
                render_kpi_card(
                    &mut cols[1],
                    bi::BOXES,
                    "Parameter Count",
                    &format!("{params_m:.2} M"),
                    "Muon 2D + AdamW 1D",
                );
                render_kpi_card(
                    &mut cols[2],
                    bi::GRAPH_UP_ARROW,
                    "Compute Density",
                    &format!("{flops_m:.2} MFLOPs"),
                    &format!(
                        "{:.1} FLOPs/Byte",
                        app.studio.roofline_report.operational_intensity
                    ),
                );
                render_kpi_card(
                    &mut cols[3],
                    bi::GPU_CARD,
                    "Estimated VRAM",
                    &format!("{vram_gb:.2} GB"),
                    &format!("{findings_len} Advisor Findings"),
                );
            });

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(6.0);

            // Channel Telemetry Table spanning full modal width
            ui.label(
                RichText::new(format!(
                    "{} Subsystem Channels & Latency Breakdown",
                    bi::BAR_CHART_LINE
                ))
                .size(12.5)
                .strong(),
            );
            ui.add_space(4.0);

            egui::Frame::group(ui.style())
                .inner_margin(egui::Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let col_w = (ui.available_width() - 48.0).max(360.0);
                            egui::Grid::new("telemetry_channels_grid")
                                .striped(true)
                                .num_columns(4)
                                .min_col_width(col_w * 0.16)
                                .spacing([20.0, 8.0])
                                .show(ui, |ui| {
                                    ui.label(RichText::new("Subsystem Channel").strong());
                                    ui.label(RichText::new("Invocations").strong());
                                    ui.label(RichText::new("Avg Latency").strong());
                                    ui.label(RichText::new("Health Status").strong());
                                    ui.end_row();

                                    for ch in &app.telemetry_data.channels {
                                        ui.label(&ch.name);
                                        ui.monospace(format!("{}", ch.count));
                                        ui.monospace(format!("{:.2} ms", ch.latency_ms));
                                        let (status_icon, color) =
                                            if ch.status == "Healthy" || ch.status == "Optimal" {
                                                (
                                                    bi::CHECK_CIRCLE_FILL,
                                                    Color32::from_rgb(70, 190, 110),
                                                )
                                            } else {
                                                (
                                                    bi::EXCLAMATION_TRIANGLE_FILL,
                                                    Color32::from_rgb(220, 150, 40),
                                                )
                                            };
                                        ui.colored_label(
                                            color,
                                            format!("{status_icon} {}", ch.status),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                });

            ui.add_space(10.0);
            ui.separator();

            // Actions: Copy Snapshot JSON / Close
            ui.horizontal(|ui| {
                if ui
                    .button(format!("{} Copy Telemetry JSON", bi::COPY))
                    .on_hover_text("Copy structured telemetry snapshot to clipboard")
                    .clicked()
                {
                    if let Ok(json) = serde_json::to_string_pretty(&app.telemetry_data) {
                        copy_to_clipboard(&json);
                    }
                }

                ui.label(
                    RichText::new(format!("Storage Tier: {}", app.storage_diag.backend.label()))
                        .size(11.0)
                        .weak(),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(format!("{} Close", bi::X_LG)).clicked() {
                        app.show_telemetry_modal = false;
                    }
                });
            });
        });

    app.show_telemetry_modal = is_open;
}

fn render_kpi_card(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    value: &str,
    subtitle: &str,
) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(10, 8))
        .corner_radius(6.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(format!("{icon} {title}"))
                        .size(11.0)
                        .weak(),
                );
                ui.add_space(2.0);
                ui.label(RichText::new(value).size(16.0).strong());
                ui.add_space(2.0);
                ui.label(RichText::new(subtitle).size(10.5).weak());
            });
        });
}
