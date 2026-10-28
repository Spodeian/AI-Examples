//! Blueprint Gallery & Architecture Experiment Notes View.
//!
//! Replaces the generic template dashboard with a searchable gallery of all 20 Starter Blueprints
//! across the 18 AI-Examples curriculum categories (with clickable curriculum links and 1-click
//! loading into the Studio), followed by an Architecture Experiment Notes & Milestones tracker.

use crate::components::palette::TEMPLATES;
use crate::icons::bi;
use crate::storage_manager::{trigger_binary_download, trigger_text_download};
use crate::studio_state::StudioState;
use crate::{ScreenConstraints, TemplateApp};
use eframe::egui::{self, Color32, RichText};
use shared::{AppPage, Priority, REPO_CATEGORIES};

pub fn render_summary_cards(
    app: &mut TemplateApp,
    ui: &mut egui::Ui,
    constraints: &ScreenConstraints,
) {
    egui::Frame::group(ui.style())
        .inner_margin(if constraints.is_mobile { 10.0 } else { 14.0 })
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} AI-Examples Starter Blueprint Gallery (20 Blueprints • 18 Categories)",
                            bi::COLLECTION
                        ))
                        .strong()
                        .size(16.0),
                    );
                    ui.label(
                        RichText::new(
                            "Browse pre-engineered neural architectures across all 8 scientific domains, open any blueprint in the Studio Workbench, or export as native .aid or .yaml.",
                        )
                        .size(12.0)
                        .color(Color32::from_rgb(175, 185, 205)),
                    );
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button(
                            RichText::new(format!(
                                "{} Return to Studio Workbench",
                                bi::DIAGRAM_3_FILL
                            ))
                            .color(Color32::from_rgb(100, 195, 255))
                            .strong(),
                        )
                        .clicked()
                    {
                        app.state.config.active_page = AppPage::Designer;
                        app.persist_state();
                    }
                });
            });
        });

    ui.add_space(8.0);

    // Search filter for Blueprint Gallery
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{} Filter Blueprints & Notes:", bi::SEARCH)).strong());
        ui.add(
            egui::TextEdit::singleline(&mut app.search_query)
                .hint_text("Search by model name, category, operator, or note...")
                .desired_width(280.0),
        );
        if !app.search_query.is_empty()
            && ui.small_button(format!("{} Clear", bi::X_LG)).clicked()
        {
            app.search_query.clear();
        }
    });

    ui.add_space(6.0);

    let query = app.search_query.trim().to_lowercase();
    let mut loaded_yaml: Option<&'static str> = None;

    egui::ScrollArea::vertical()
        .id_salt("blueprint_gallery_scroll")
        .max_height(310.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (idx, tmpl) in TEMPLATES.iter().enumerate() {
                let clean_id = tmpl
                    .file_name
                    .trim_end_matches(".aid.yaml")
                    .trim_end_matches(".yaml");
                let linked_cat = REPO_CATEGORIES
                    .iter()
                    .find(|c| c.recommended_starter == Some(tmpl.file_name))
                    .unwrap_or(&REPO_CATEGORIES[idx % REPO_CATEGORIES.len()]);

                if !query.is_empty()
                    && !tmpl.display_name.to_lowercase().contains(&query)
                    && !clean_id.to_lowercase().contains(&query)
                    && !linked_cat.name.to_lowercase().contains(&query)
                    && !linked_cat.description.to_lowercase().contains(&query)
                {
                    continue;
                }

                egui::Frame::group(ui.style())
                    .inner_margin(10.0)
                    .corner_radius(6.0)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!(
                                            "{} {}",
                                            bi::DIAGRAM_3,
                                            tmpl.display_name
                                        ))
                                        .size(14.0)
                                        .strong(),
                                    );
                                    let cat_url = linked_cat.category_url();
                                    ui.hyperlink_to(
                                        RichText::new(format!(
                                            "{} [{}] {} {}",
                                            bi::BOOK,
                                            linked_cat.id,
                                            linked_cat.name,
                                            bi::BOX_ARROW_UP_RIGHT
                                        ))
                                        .size(11.0)
                                        .monospace()
                                        .color(Color32::from_rgb(110, 195, 250)),
                                        &cat_url,
                                    )
                                    .on_hover_text(format!("Open curriculum category: {cat_url}"));
                                });
                                ui.label(
                                    RichText::new(linked_cat.description)
                                        .size(11.5)
                                        .color(Color32::from_rgb(175, 185, 200)),
                                );
                            });

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button(
                                            RichText::new(format!(
                                                "{} Open in Studio",
                                                bi::ROCKET_TAKEOFF
                                            ))
                                            .color(Color32::from_rgb(120, 225, 150))
                                            .strong(),
                                        )
                                        .clicked()
                                    {
                                        loaded_yaml = Some(tmpl.yaml_content);
                                    }

                                    if ui
                                        .button(format!("{} .aid", bi::FILE_EARMARK_BINARY))
                                        .on_hover_text(
                                            "Download native binary .aid blueprint container",
                                        )
                                        .clicked()
                                        && let Ok(spec) =
                                            shared::format::aid_yaml::from_yaml_str(
                                                tmpl.yaml_content,
                                            )
                                        && let Ok(aid_bytes) =
                                            shared::format::aid_envelope::pack_aid(
                                                &spec,
                                                shared::format::AidFormatTag::Bson,
                                            )
                                    {
                                        let fname = format!("{clean_id}.aid");
                                        trigger_binary_download(
                                            &fname,
                                            &aid_bytes,
                                            "application/octet-stream",
                                        );
                                    }

                                    if ui
                                        .button(format!("{} .yaml", bi::FILE_EARMARK_CODE))
                                        .on_hover_text("Download human-readable .yaml blueprint")
                                        .clicked()
                                    {
                                        let fname = format!("{clean_id}.yaml");
                                        trigger_text_download(
                                            &fname,
                                            tmpl.yaml_content,
                                            "text/yaml;charset=utf-8",
                                        );
                                    }
                                },
                            );
                        });
                    });
                ui.add_space(4.0);
            }
        });

    if let Some(yaml) = loaded_yaml
        && let Ok(spec) = shared::format::aid_yaml::from_yaml_str(yaml)
    {
        app.studio = StudioState::new_with_starter(spec);
        app.state.config.active_page = AppPage::Designer;
        app.persist_state();
    }
}

pub fn render_new_item_form(
    app: &mut TemplateApp,
    ui: &mut egui::Ui,
    constraints: &ScreenConstraints,
) {
    let total = app.state.collection.total_count();
    let completed = app.state.collection.completed_count();
    let ratio = app.state.collection.completion_ratio();

    egui::Frame::group(ui.style())
        .inner_margin(if constraints.is_mobile { 10.0 } else { 14.0 })
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{} Architecture Experiment Notes & Milestones ({completed}/{total} completed)",
                        bi::JOURNAL_CODE
                    ))
                    .strong()
                    .size(14.0),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let bar_width = (ui.available_width() - 8.0).clamp(80.0, 220.0);
                    ui.add(
                        egui::ProgressBar::new(ratio)
                            .text(format!("{:.0}%", ratio * 100.0))
                            .desired_width(bar_width),
                    );
                });
            });
            ui.add_space(6.0);

            let mut submit_new_item = false;

            ui.horizontal_wrapped(|ui| {
                let title_resp = ui.add(
                    egui::TextEdit::singleline(&mut app.new_item_title)
                        .hint_text("Add architecture experiment note or milestone...")
                        .desired_width(if constraints.is_mobile { 180.0 } else { 240.0 }),
                );
                if title_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    submit_new_item = true;
                }
                let desc_resp = ui.add(
                    egui::TextEdit::singleline(&mut app.new_item_description)
                        .hint_text("Hypothesis, target hardware, or metric...")
                        .desired_width(if constraints.is_mobile { 180.0 } else { 300.0 }),
                );
                if desc_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    submit_new_item = true;
                }
                egui::ComboBox::from_id_salt("desktop_priority_dropdown")
                    .selected_text(app.new_item_priority.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut app.new_item_priority, Priority::Low, "Low");
                        ui.selectable_value(
                            &mut app.new_item_priority,
                            Priority::Medium,
                            "Medium",
                        );
                        ui.selectable_value(&mut app.new_item_priority, Priority::High, "High");
                    });

                if ui
                    .button(format!("{} Add Note", bi::PLUS_LG))
                    .clicked()
                {
                    submit_new_item = true;
                }
            });

            if submit_new_item && !app.new_item_title.trim().is_empty() {
                app.state.collection.add(
                    app.new_item_title.trim(),
                    app.new_item_description.trim(),
                    app.new_item_priority,
                );
                app.new_item_title.clear();
                app.new_item_description.clear();
                app.persist_state();
            }
        });
}

pub fn render_item_list(app: &mut TemplateApp, ui: &mut egui::Ui, _constraints: &ScreenConstraints) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{} Saved Experiment Notes", bi::LIST_NESTED))
                .size(14.0)
                .strong(),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button(format!("{} Clear Completed", bi::TRASH3))
                .on_hover_text("Remove all completed notes")
                .clicked()
            {
                app.state.collection.clear_completed();
                app.persist_state();
            }

            egui::ComboBox::from_id_salt("filter_priority_dropdown")
                .selected_text(match app.filter_priority {
                    None => "All Priorities",
                    Some(p) => p.label(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.filter_priority, None, "All Priorities");
                    ui.selectable_value(&mut app.filter_priority, Some(Priority::Low), "Low");
                    ui.selectable_value(&mut app.filter_priority, Some(Priority::Medium), "Medium");
                    ui.selectable_value(&mut app.filter_priority, Some(Priority::High), "High");
                });
        });
    });

    ui.add_space(6.0);

    let query = app.search_query.trim().to_lowercase();
    let priority_filter = app.filter_priority;

    let mut item_ids_to_toggle = Vec::new();
    let mut item_ids_to_remove = Vec::new();

    egui::ScrollArea::vertical()
        .id_salt("experiment_notes_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let matching_items: Vec<_> = app
                .state
                .collection
                .items
                .iter()
                .filter(|item| {
                    if let Some(pf) = priority_filter
                        && item.priority != pf
                    {
                        return false;
                    }
                    if !query.is_empty() {
                        let match_title = item.title.to_lowercase().contains(&query);
                        let match_desc = item.description.to_lowercase().contains(&query);
                        if !match_title && !match_desc {
                            return false;
                        }
                    }
                    true
                })
                .cloned()
                .collect();

            if matching_items.is_empty() {
                ui.add_space(12.0);
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("No experiment notes match your query or filter.")
                            .weak(),
                    );
                });
                return;
            }

            for item in matching_items {
                egui::Frame::group(ui.style())
                    .inner_margin(8.0)
                    .corner_radius(6.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let mut checked = item.completed;
                            if ui.checkbox(&mut checked, "").clicked() {
                                item_ids_to_toggle.push(item.id);
                            }

                            ui.vertical(|ui| {
                                let mut title_text =
                                    egui::RichText::new(&item.title).strong().size(13.5);
                                if item.completed {
                                    title_text = title_text.strikethrough().weak();
                                }
                                ui.label(title_text);

                                if !item.description.is_empty() {
                                    let mut desc_text =
                                        egui::RichText::new(&item.description).small();
                                    if item.completed {
                                        desc_text = desc_text.weak();
                                    }
                                    ui.label(desc_text);
                                }
                            });

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button(bi::TRASH3)
                                        .on_hover_text("Delete note")
                                        .clicked()
                                    {
                                        item_ids_to_remove.push(item.id);
                                    }

                                    let badge_color = match item.priority {
                                        Priority::High => egui::Color32::from_rgb(220, 70, 70),
                                        Priority::Medium => egui::Color32::from_rgb(230, 140, 50),
                                        Priority::Low => egui::Color32::from_rgb(100, 140, 180),
                                    };
                                    ui.colored_label(
                                        badge_color,
                                        egui::RichText::new(item.priority.label()).strong().small(),
                                    );
                                },
                            );
                        });
                    });
                ui.add_space(4.0);
            }
        });

    let mut collection_changed = false;
    for id in item_ids_to_toggle {
        app.state.collection.toggle(id);
        collection_changed = true;
    }
    for id in item_ids_to_remove {
        app.state.collection.remove(id);
        collection_changed = true;
    }
    if collection_changed {
        app.persist_state();
    }
}
