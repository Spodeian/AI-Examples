//! Advisory Engine Dock: displays 55+ mathematical and physical failure mode warnings with one-click quick-fixes
//! and clickable curriculum reference links.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{Color32, RichText, ScrollArea, Ui};
use shared::advisor::AdvisorySeverity;

pub struct AdvisorDockView;

impl Default for AdvisorDockView {
    fn default() -> Self {
        Self
    }
}

impl AdvisorDockView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new(format!(
                        "{} Intelligent Mathematical & Physical Advisor",
                        bi::SHIELD_CHECK
                    ))
                    .size(15.0)
                    .strong(),
                );
                ui.label(format!("({} active diagnostics)", state.advisor_findings.len()));
            });

            ui.add_space(4.0);
            ui.separator();

            if state.advisor_findings.is_empty() {
                ui.label(
                    RichText::new(format!(
                        "{} All 55+ mathematical invariants, physical conservation laws, and optimizer partitioning rules satisfied.",
                        bi::CHECK_CIRCLE_FILL
                    ))
                    .color(Color32::from_rgb(120, 220, 140)),
                );
                return;
            }

            let mut mutation_to_apply = None;

            for finding in &state.advisor_findings {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let (badge_icon, badge_text, badge_color) = match finding.severity {
                            AdvisorySeverity::CriticalMathematicalDefect => (
                                bi::EXCLAMATION_TRIANGLE_FILL,
                                "CRITICAL",
                                Color32::from_rgb(240, 80, 80),
                            ),
                            AdvisorySeverity::Warning => (
                                bi::EXCLAMATION_TRIANGLE,
                                "WARNING",
                                Color32::from_rgb(240, 180, 60),
                            ),
                            AdvisorySeverity::BestPractice => (
                                bi::LIGHTNING_CHARGE_FILL,
                                "BEST PRACTICE",
                                Color32::from_rgb(100, 180, 255),
                            ),
                            AdvisorySeverity::Info => (
                                bi::INFO_CIRCLE,
                                "INFO",
                                Color32::from_rgb(180, 180, 180),
                            ),
                        };

                        ui.label(
                            RichText::new(format!("{badge_icon} {badge_text}"))
                                .color(badge_color)
                                .strong()
                                .size(11.0),
                        );
                        ui.label(
                            RichText::new(format!("[{}] {}", finding.rule_id, finding.title))
                                .strong()
                                .size(13.0),
                        );

                        ui.with_layout(
                            eframe::egui::Layout::right_to_left(eframe::egui::Align::Center),
                            |ui| {
                                if let Some(ref mutation) = finding.quick_fix
                                    && ui
                                        .button(
                                            RichText::new(format!(
                                                "{} Quick Fix",
                                                bi::LIGHTNING_CHARGE_FILL
                                            ))
                                            .color(Color32::from_rgb(120, 220, 140))
                                            .strong(),
                                        )
                                        .clicked()
                                {
                                    mutation_to_apply = Some(mutation.clone());
                                }

                                let ref_slug = finding.category.curriculum_slug();
                                let ref_url = finding.category.curriculum_url();
                                ui.hyperlink_to(
                                    RichText::new(format!(
                                        "{} {ref_slug} {}",
                                        bi::BOOK,
                                        bi::BOX_ARROW_UP_RIGHT
                                    ))
                                    .size(11.0)
                                    .monospace()
                                    .color(Color32::from_rgb(110, 195, 250)),
                                    &ref_url,
                                )
                                .on_hover_text(format!(
                                    "Open AI-Examples curriculum reference: {ref_url}"
                                ));
                            },
                        );
                    });

                    ui.add_space(2.0);
                    ui.label(&finding.description);

                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(format!("Rationale: {}", finding.mathematical_rationale))
                            .size(11.0)
                            .color(Color32::from_rgb(180, 190, 210)),
                    );
                });
                ui.add_space(4.0);
            }

            if let Some(mutation) = mutation_to_apply {
                mutation.apply(&mut state.spec);
                state.refresh_analyses();
            }
        });
    }
}
