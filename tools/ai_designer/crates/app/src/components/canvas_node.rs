//! Interactive Dataflow Graph Canvas: renders Input Port nodes, Computation Layer nodes,
//! Output Port & Loss nodes, cubic Bézier dataflow wires, symbolic shape chips, and HUD controls.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{self, Color32, CornerRadius, Id, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use eframe::epaint::CubicBezierShape;
use shared::{PrecisionFormat, TensorSpace};

pub struct CanvasNodeView;

impl Default for CanvasNodeView {
    fn default() -> Self {
        Self
    }
}

impl CanvasNodeView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        // 1. Canvas Pan & Zoom handling
        if response.dragged_by(egui::PointerButton::Middle)
            || (response.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.alt))
        {
            state.pan_offset += response.drag_delta();
        }

        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta != 0.0 && response.hovered() {
            let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 0.9 };
            state.zoom_level = (state.zoom_level * zoom_factor).clamp(0.25, 2.5);
        }

        // Draw background canvas grid
        self.draw_grid(&painter, rect, state.pan_offset, state.zoom_level);

        let zoom = state.zoom_level;
        let to_screen =
            |pos: Pos2| -> Pos2 { rect.min + state.pan_offset + (pos.to_vec2() * zoom) };

        let card_width = 235.0 * zoom;
        let card_height = 92.0 * zoom;

        // Helper to draw a cubic Bézier wire and optional midpoint shape chip
        let draw_wire = |start_point: Pos2,
                         end_point: Pos2,
                         wire_color: Color32,
                         chip_text: Option<String>| {
            let dx = (end_point.x - start_point.x).abs().max(45.0) * 0.5;
            let control1 = start_point + Vec2::new(dx, 0.0);
            let control2 = end_point - Vec2::new(dx, 0.0);

            let wire_stroke = Stroke::new(2.5 * zoom.clamp(0.6, 1.5), wire_color);
            painter.add(CubicBezierShape::from_points_stroke(
                [start_point, control1, control2, end_point],
                false,
                Color32::TRANSPARENT,
                wire_stroke,
            ));

            if zoom >= 0.6
                && let Some(label) = chip_text
            {
                let mid = Pos2::new(
                    (start_point.x + end_point.x) * 0.5,
                    (start_point.y + end_point.y) * 0.5,
                );
                let chip_w = (label.len() as f32 * 6.5 + 18.0).max(72.0) * zoom;
                let chip_rect = Rect::from_center_size(mid, Vec2::new(chip_w, 18.0 * zoom));
                painter.rect_filled(chip_rect, 4.0, Color32::from_rgb(24, 30, 42));
                painter.rect_stroke(
                    chip_rect,
                    4.0,
                    Stroke::new(1.0, wire_color),
                    egui::epaint::StrokeKind::Inside,
                );
                painter.text(
                    mid,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::monospace(10.0 * zoom),
                    Color32::from_rgb(185, 215, 255),
                );
            }
        };

        // 2. Draw Dataflow Wires: Inputs -> Computation Layers -> Outputs
        let node_count = state.spec.graph.nodes.len();

        if node_count > 0 {
            // (a) Input Port(s) -> First Computation Layer
            let first_node = &state.spec.graph.nodes[0];
            let first_pos = state.get_node_pos(&first_node.id);
            let first_in_pin = to_screen(first_pos) + Vec2::new(0.0, card_height * 0.5);

            for (idx, input) in state.spec.inputs.iter().enumerate() {
                let in_key = format!("__input__:{}", input.id);
                let in_pos = state
                    .node_positions
                    .get(&in_key)
                    .copied()
                    .unwrap_or(Pos2::new(40.0, 110.0 + (idx as f32 * 140.0)));
                let start_pin = to_screen(in_pos) + Vec2::new(card_width, card_height * 0.5);
                let shape_str = state
                    .shape_report
                    .input_shapes
                    .get(&input.id)
                    .map(|s| s.display());
                draw_wire(
                    start_pin,
                    first_in_pin,
                    Color32::from_rgb(70, 185, 235),
                    shape_str,
                );
            }

            // (b) Computation Layer i -> Computation Layer i + 1
            for i in 0..node_count.saturating_sub(1) {
                let from_node = &state.spec.graph.nodes[i];
                let to_node = &state.spec.graph.nodes[i + 1];

                let from_pos = state.get_node_pos(&from_node.id);
                let to_pos = state.get_node_pos(&to_node.id);

                let start_point = to_screen(from_pos) + Vec2::new(card_width, card_height * 0.5);
                let end_point = to_screen(to_pos) + Vec2::new(0.0, card_height * 0.5);

                let chip_str = state
                    .shape_report
                    .node_shapes
                    .get(&from_node.id)
                    .and_then(|info| info.output_shape.as_ref().map(|s| s.display()));

                draw_wire(
                    start_point,
                    end_point,
                    Color32::from_rgb(100, 150, 240),
                    chip_str,
                );
            }

            // (c) Last Computation Layer -> Output Port(s)
            let last_node = &state.spec.graph.nodes[node_count - 1];
            let last_pos = state.get_node_pos(&last_node.id);
            let last_out_pin = to_screen(last_pos) + Vec2::new(card_width, card_height * 0.5);
            let last_shape_str = state
                .shape_report
                .node_shapes
                .get(&last_node.id)
                .and_then(|info| info.output_shape.as_ref().map(|s| s.display()));

            for (idx, output) in state.spec.outputs.iter().enumerate() {
                let out_key = format!("__output__:{}", output.id);
                let out_pos = state
                    .node_positions
                    .get(&out_key)
                    .copied()
                    .unwrap_or(Pos2::new(980.0, 110.0 + (idx as f32 * 140.0)));
                let end_pin = to_screen(out_pos) + Vec2::new(0.0, card_height * 0.5);
                draw_wire(
                    last_out_pin,
                    end_pin,
                    Color32::from_rgb(90, 210, 145),
                    last_shape_str.clone(),
                );
            }
        } else {
            // Blank Canvas: connect Input Port(s) -> Output Port(s) with muted pending wire
            for (i_idx, input) in state.spec.inputs.iter().enumerate() {
                let in_key = format!("__input__:{}", input.id);
                let in_pos = state
                    .node_positions
                    .get(&in_key)
                    .copied()
                    .unwrap_or(Pos2::new(40.0, 110.0 + (i_idx as f32 * 140.0)));
                let start_pin = to_screen(in_pos) + Vec2::new(card_width, card_height * 0.5);

                for (o_idx, output) in state.spec.outputs.iter().enumerate() {
                    let out_key = format!("__output__:{}", output.id);
                    let out_pos = state
                        .node_positions
                        .get(&out_key)
                        .copied()
                        .unwrap_or(Pos2::new(660.0, 110.0 + (o_idx as f32 * 140.0)));
                    let end_pin = to_screen(out_pos) + Vec2::new(0.0, card_height * 0.5);
                    draw_wire(
                        start_pin,
                        end_pin,
                        Color32::from_rgb(80, 95, 125),
                        Some("Add Layers Between I/O".to_string()),
                    );
                }
            }
        }

        let mut clicked_node: Option<Option<String>> = None;
        let mut dragged_key: Option<(String, Vec2)> = None;

        // 3A. Render Input Port Cards
        for (idx, input) in state.spec.inputs.iter().enumerate() {
            let in_key = format!("__input__:{}", input.id);
            let canvas_pos = state
                .node_positions
                .get(&in_key)
                .copied()
                .unwrap_or(Pos2::new(40.0, 110.0 + (idx as f32 * 140.0)));
            let screen_pos = to_screen(canvas_pos);
            let card_rect = Rect::from_min_size(screen_pos, Vec2::new(card_width, card_height));

            painter.rect_filled(card_rect, CornerRadius::same(6), Color32::from_rgb(20, 34, 46));
            painter.rect_stroke(
                card_rect,
                CornerRadius::same(6),
                Stroke::new(1.5, Color32::from_rgb(65, 175, 225)),
                egui::epaint::StrokeKind::Inside,
            );

            let header_height = 28.0 * zoom;
            let header_rect = Rect::from_min_size(screen_pos, Vec2::new(card_width, header_height));
            painter.rect_filled(
                header_rect,
                CornerRadius {
                    nw: 6,
                    ne: 6,
                    sw: 0,
                    se: 0,
                },
                Color32::from_rgb(28, 56, 76),
            );

            painter.text(
                header_rect.min + Vec2::new(10.0, 6.0) * zoom,
                egui::Align2::LEFT_TOP,
                format!("{} INPUT: {}", bi::BOX_ARROW_IN_RIGHT, input.name),
                egui::FontId::proportional(11.5 * zoom),
                Color32::from_rgb(160, 230, 255),
            );

            let shape_badge = state
                .shape_report
                .input_shapes
                .get(&input.id)
                .map(|s| s.display())
                .unwrap_or_else(|| "[B, D]".to_string());
            painter.text(
                header_rect.max - Vec2::new(8.0, 14.0) * zoom,
                egui::Align2::RIGHT_CENTER,
                shape_badge,
                egui::FontId::monospace(9.5 * zoom),
                Color32::from_rgb(110, 220, 255),
            );

            if zoom >= 0.5 {
                let space_name = tensor_space_short_label(&input.space);
                painter.text(
                    card_rect.min + Vec2::new(10.0, header_height + 8.0 * zoom),
                    egui::Align2::LEFT_TOP,
                    format!("Space: {space_name}"),
                    egui::FontId::monospace(10.5 * zoom),
                    Color32::from_rgb(200, 220, 235),
                );
                painter.text(
                    card_rect.min + Vec2::new(10.0, header_height + 26.0 * zoom),
                    egui::Align2::LEFT_TOP,
                    format!(
                        "Norm: {:?} | Enc: {:?}",
                        input.normalizer, input.coordinate_encoding
                    ),
                    egui::FontId::proportional(10.0 * zoom),
                    Color32::from_rgb(145, 175, 200),
                );
            }

            // Output port pin on right edge
            let out_pin = screen_pos + Vec2::new(card_width, card_height * 0.5);
            let pin_radius = 5.5 * zoom;
            painter.circle_filled(out_pin, pin_radius, Color32::from_rgb(80, 200, 255));
            painter.circle_stroke(out_pin, pin_radius, Stroke::new(1.0, Color32::BLACK));

            let resp = ui.interact(card_rect, Id::new(&in_key), Sense::click_and_drag());
            if resp.clicked() {
                // Deselect layer so Right Inspector shows Blueprint & I/O Configuration
                clicked_node = Some(None);
            }
            if resp.dragged() {
                dragged_key = Some((in_key, resp.drag_delta() / zoom));
            }
        }

        // 3B. Render Computation Layer Cards
        for (idx, node) in state.spec.graph.nodes.iter().enumerate() {
            let canvas_pos = state.get_node_pos(&node.id);
            let screen_pos = to_screen(canvas_pos);
            let node_rect = Rect::from_min_size(screen_pos, Vec2::new(card_width, card_height));

            let is_selected = state.selected_node_id.as_deref() == Some(&node.id);

            let bg_color = if is_selected {
                Color32::from_rgb(38, 44, 62)
            } else {
                Color32::from_rgb(26, 30, 42)
            };

            let border_stroke = if is_selected {
                Stroke::new(2.0, Color32::from_rgb(100, 180, 255))
            } else {
                Stroke::new(1.0, Color32::from_rgb(55, 65, 88))
            };

            painter.rect_filled(node_rect, CornerRadius::same(6), bg_color);
            painter.rect_stroke(
                node_rect,
                CornerRadius::same(6),
                border_stroke,
                egui::epaint::StrokeKind::Inside,
            );

            let header_height = 28.0 * zoom;
            let header_rect = Rect::from_min_size(screen_pos, Vec2::new(card_width, header_height));
            let header_bg = if is_selected {
                Color32::from_rgb(48, 60, 86)
            } else {
                Color32::from_rgb(32, 38, 54)
            };
            painter.rect_filled(
                header_rect,
                CornerRadius {
                    nw: 6,
                    ne: 6,
                    sw: 0,
                    se: 0,
                },
                header_bg,
            );

            painter.text(
                header_rect.min + Vec2::new(10.0, 6.0) * zoom,
                egui::Align2::LEFT_TOP,
                format!("{} {}. {}", bi::LAYERS, idx + 1, node.name),
                egui::FontId::proportional(11.5 * zoom),
                Color32::WHITE,
            );

            let prec = node.precision.unwrap_or(PrecisionFormat::BF16);
            let badge_color = match prec {
                PrecisionFormat::FP8E4M3FN | PrecisionFormat::FP8E5M2 => {
                    Color32::from_rgb(240, 140, 60)
                }
                PrecisionFormat::FP32 => Color32::from_rgb(100, 160, 255),
                PrecisionFormat::INT8Sym => Color32::from_rgb(220, 100, 220),
                _ => Color32::from_rgb(100, 220, 140),
            };
            painter.text(
                header_rect.max - Vec2::new(8.0, 14.0) * zoom,
                egui::Align2::RIGHT_CENTER,
                format!("{:?}", prec),
                egui::FontId::monospace(9.0 * zoom),
                badge_color,
            );

            if zoom >= 0.5 {
                let op_label = format!("{:?}", node.operator);
                let op_short = op_label.split('{').next().unwrap_or(&op_label).trim();
                painter.text(
                    node_rect.min + Vec2::new(10.0, header_height + 8.0 * zoom),
                    egui::Align2::LEFT_TOP,
                    format!("Op: {op_short}"),
                    egui::FontId::monospace(10.5 * zoom),
                    Color32::from_rgb(185, 198, 220),
                );
                painter.text(
                    node_rect.min + Vec2::new(10.0, header_height + 26.0 * zoom),
                    egui::Align2::LEFT_TOP,
                    node.operator.category_tag(),
                    egui::FontId::proportional(10.0 * zoom),
                    Color32::from_rgb(140, 165, 195),
                );
            }

            // Input & Output pins
            let in_pin = screen_pos + Vec2::new(0.0, card_height * 0.5);
            let out_pin = screen_pos + Vec2::new(card_width, card_height * 0.5);
            let pin_radius = 5.0 * zoom;

            painter.circle_filled(in_pin, pin_radius, Color32::from_rgb(100, 180, 255));
            painter.circle_stroke(in_pin, pin_radius, Stroke::new(1.0, Color32::BLACK));

            painter.circle_filled(out_pin, pin_radius, Color32::from_rgb(120, 220, 140));
            painter.circle_stroke(out_pin, pin_radius, Stroke::new(1.0, Color32::BLACK));

            let node_response = ui.interact(node_rect, Id::new(&node.id), Sense::click_and_drag());
            if node_response.clicked() {
                clicked_node = Some(Some(node.id.clone()));
            }
            if node_response.dragged() {
                dragged_key = Some((node.id.clone(), node_response.drag_delta() / zoom));
            }
        }

        // 3C. Render Output Port & Loss Cards
        for (idx, output) in state.spec.outputs.iter().enumerate() {
            let out_key = format!("__output__:{}", output.id);
            let canvas_pos = state
                .node_positions
                .get(&out_key)
                .copied()
                .unwrap_or(Pos2::new(980.0, 110.0 + (idx as f32 * 140.0)));
            let screen_pos = to_screen(canvas_pos);
            let card_rect = Rect::from_min_size(screen_pos, Vec2::new(card_width, card_height));

            painter.rect_filled(card_rect, CornerRadius::same(6), Color32::from_rgb(20, 38, 32));
            painter.rect_stroke(
                card_rect,
                CornerRadius::same(6),
                Stroke::new(1.5, Color32::from_rgb(80, 205, 140)),
                egui::epaint::StrokeKind::Inside,
            );

            let header_height = 28.0 * zoom;
            let header_rect = Rect::from_min_size(screen_pos, Vec2::new(card_width, header_height));
            painter.rect_filled(
                header_rect,
                CornerRadius {
                    nw: 6,
                    ne: 6,
                    sw: 0,
                    se: 0,
                },
                Color32::from_rgb(28, 62, 46),
            );

            painter.text(
                header_rect.min + Vec2::new(10.0, 6.0) * zoom,
                egui::Align2::LEFT_TOP,
                format!("{} OUTPUT: {}", bi::BOX_ARROW_RIGHT, output.name),
                egui::FontId::proportional(11.5 * zoom),
                Color32::from_rgb(165, 245, 195),
            );

            let out_shape_badge = state
                .shape_report
                .output_shapes
                .get(&output.id)
                .map(|s| s.display())
                .unwrap_or_else(|| "[B, Out]".to_string());
            painter.text(
                header_rect.max - Vec2::new(8.0, 14.0) * zoom,
                egui::Align2::RIGHT_CENTER,
                out_shape_badge,
                egui::FontId::monospace(9.5 * zoom),
                Color32::from_rgb(130, 235, 170),
            );

            if zoom >= 0.5 {
                painter.text(
                    card_rect.min + Vec2::new(10.0, header_height + 8.0 * zoom),
                    egui::Align2::LEFT_TOP,
                    format!("Loss: {:?}", output.loss),
                    egui::FontId::monospace(10.5 * zoom),
                    Color32::from_rgb(205, 235, 215),
                );
                painter.text(
                    card_rect.min + Vec2::new(10.0, header_height + 26.0 * zoom),
                    egui::Align2::LEFT_TOP,
                    format!("Constraint: {:?}", output.constraint),
                    egui::FontId::proportional(10.0 * zoom),
                    Color32::from_rgb(150, 195, 170),
                );
            }

            // Input pin on left edge
            let in_pin = screen_pos + Vec2::new(0.0, card_height * 0.5);
            let pin_radius = 5.5 * zoom;
            painter.circle_filled(in_pin, pin_radius, Color32::from_rgb(90, 220, 150));
            painter.circle_stroke(in_pin, pin_radius, Stroke::new(1.0, Color32::BLACK));

            let resp = ui.interact(card_rect, Id::new(&out_key), Sense::click_and_drag());
            if resp.clicked() {
                clicked_node = Some(None);
            }
            if resp.dragged() {
                dragged_key = Some((out_key, resp.drag_delta() / zoom));
            }
        }

        if let Some(ref sel) = clicked_node {
            state.select_node(sel.clone());
        }

        if let Some((key, delta)) = dragged_key {
            let entry = state.node_positions.entry(key).or_insert(Pos2::new(150.0, 150.0));
            *entry += delta;
        }

        // Clicking empty canvas clears selection (only if pointer is not over top HUD bar)
        let hud_rect = Rect::from_min_size(
            rect.min + Vec2::new(10.0, 10.0),
            Vec2::new((rect.width() - 20.0).max(200.0), 34.0),
        );
        let pointer_in_hud = response
            .interact_pointer_pos()
            .is_some_and(|p| hud_rect.contains(p));
        if response.clicked() && clicked_node.is_none() && !pointer_in_hud {
            state.selected_node_id = None;
        }

        // 4. Top Canvas HUD Bar (Zoom, Auto-Layout, I/O + Layer Summary, and Quick Actions)
        painter.rect_filled(
            hud_rect,
            6.0,
            Color32::from_rgba_premultiplied(24, 28, 40, 235),
        );
        painter.rect_stroke(
            hud_rect,
            6.0,
            Stroke::new(1.0, Color32::from_rgb(55, 68, 95)),
            egui::epaint::StrokeKind::Inside,
        );

        let mut hud_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(hud_rect.shrink2(Vec2::new(10.0, 4.0)))
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        hud_ui.horizontal(|ui| {
            if ui
                .small_button(bi::ZOOM_OUT)
                .on_hover_text("Zoom Out")
                .clicked()
            {
                state.zoom_level = (state.zoom_level * 0.85).clamp(0.25, 2.5);
            }
            let zoom_pct = format!("{:.0}%", state.zoom_level * 100.0);
            if ui
                .small_button(zoom_pct)
                .on_hover_text("Reset Zoom to 100%")
                .clicked()
            {
                state.zoom_level = 1.0;
            }
            if ui
                .small_button(bi::ZOOM_IN)
                .on_hover_text("Zoom In")
                .clicked()
            {
                state.zoom_level = (state.zoom_level * 1.15).clamp(0.25, 2.5);
            }

            ui.separator();

            if ui
                .small_button(format!("{} Auto-Layout", bi::CROSSHAIR))
                .on_hover_text("Center camera and arrange Input, Layer, and Output nodes (Home)")
                .clicked()
            {
                state.auto_layout_nodes();
            }

            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "{} {} In -> {} Layers -> {} Out",
                    bi::DIAGRAM_3,
                    state.spec.inputs.len(),
                    state.spec.graph.nodes.len(),
                    state.spec.outputs.len()
                ))
                .size(11.0)
                .color(Color32::from_rgb(165, 195, 230)),
            );

            if let Some(ref sel_id) = state.selected_node_id.clone() {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("Layer: {sel_id}"))
                        .size(11.0)
                        .color(Color32::from_rgb(120, 200, 255))
                        .strong(),
                );
                if ui
                    .small_button(bi::ARROW_UP)
                    .on_hover_text("Move Layer Earlier in Graph")
                    .clicked()
                {
                    state.move_selected_node(-1);
                }
                if ui
                    .small_button(bi::ARROW_DOWN)
                    .on_hover_text("Move Layer Later in Graph")
                    .clicked()
                {
                    state.move_selected_node(1);
                }
                if ui
                    .small_button(bi::COPY)
                    .on_hover_text("Duplicate Selected Layer")
                    .clicked()
                {
                    state.duplicate_selected_node();
                }
                if ui
                    .small_button(
                        egui::RichText::new(bi::TRASH3).color(Color32::from_rgb(240, 100, 100)),
                    )
                    .on_hover_text("Delete Selected Layer (Del)")
                    .clicked()
                {
                    state.delete_selected_node();
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(
                        "Pan: Middle-Drag / Alt+Drag • Zoom: Scroll • Click I/O or Layer: Inspect",
                    )
                    .size(10.5)
                    .color(Color32::from_rgb(150, 165, 190)),
                );
            });
        });

        // 5. If canvas has no computation layers, render a compact Quick-Start Card at bottom-center
        if state.spec.graph.nodes.is_empty() {
            let center = Pos2::new(rect.center().x, rect.max.y - 95.0);
            let overlay_rect = Rect::from_center_size(center, Vec2::new(560.0, 125.0));
            painter.rect_filled(overlay_rect, 10.0, Color32::from_rgb(24, 28, 40));
            painter.rect_stroke(
                overlay_rect,
                10.0,
                Stroke::new(1.5, Color32::from_rgb(70, 100, 160)),
                egui::epaint::StrokeKind::Inside,
            );

            let mut child_ui =
                ui.new_child(egui::UiBuilder::new().max_rect(overlay_rect.shrink(14.0)));
            child_ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "{} Blank Computation Graph — Connect Input Stream to Output Stream",
                        bi::DIAGRAM_3
                    ))
                    .size(14.0)
                    .color(Color32::from_rgb(100, 190, 255))
                    .strong(),
                );
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    if ui
                        .button(
                            egui::RichText::new(format!("{} Add Linear Layer", bi::PLUS_CIRCLE))
                                .strong(),
                        )
                        .clicked()
                    {
                        let id = "linear_1".to_string();
                        state.spec.graph.nodes.push(shared::NodeSpec {
                            id: id.clone(),
                            name: "Dense Linear Layer".to_string(),
                            operator: shared::OperatorKind::Linear {
                                in_features: 256,
                                out_features: 256,
                                bias: true,
                            },
                            precision: Some(shared::PrecisionFormat::BF16),
                            metadata: Default::default(),
                        });
                        state.auto_layout_nodes();
                        state.select_node(Some(id));
                        state.refresh_analyses();
                    }
                    if ui
                        .button(
                            egui::RichText::new(format!(
                                "{} Add Attention Backbone",
                                bi::PLUS_CIRCLE
                            ))
                            .strong(),
                        )
                        .clicked()
                    {
                        let id = "mha_1".to_string();
                        state.spec.graph.nodes.push(shared::NodeSpec {
                            id: id.clone(),
                            name: "Multi-Head Attention".to_string(),
                            operator: shared::OperatorKind::MHA {
                                num_heads: 8,
                                d_k: 64,
                                d_v: 64,
                                dropout: 0.0,
                            },
                            precision: Some(shared::PrecisionFormat::BF16),
                            metadata: Default::default(),
                        });
                        state.auto_layout_nodes();
                        state.select_node(Some(id));
                        state.refresh_analyses();
                    }
                    if ui
                        .button(
                            egui::RichText::new(format!(
                                "{} Auto-Architect from I/O",
                                bi::MAGIC
                            ))
                            .color(Color32::from_rgb(120, 220, 140))
                            .strong(),
                        )
                        .clicked()
                    {
                        let synthesized = shared::advisor::auto_architect::synthesize_architecture(
                            &format!("{}_synthesized", state.spec.metadata.model_id),
                            &format!("Synthesized {}", state.spec.metadata.display_name),
                            state.spec.inputs.clone(),
                            state.spec.outputs.clone(),
                        );
                        *state = StudioState::new_with_starter(synthesized);
                    }
                });

                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Tip: Click '+' on any operator in the Left Palette, or click Input/Output cards above to configure tensor spaces.",
                    )
                    .size(11.0)
                    .color(Color32::from_rgb(160, 170, 190)),
                );
            });
        }
    }

    fn draw_grid(&self, painter: &egui::Painter, rect: Rect, pan: Vec2, zoom: f32) {
        painter.rect_filled(rect, 0.0, Color32::from_rgb(18, 20, 28));

        let grid_size = 32.0 * zoom;
        if grid_size < 8.0 {
            return;
        }

        let start_x = rect.min.x + (pan.x % grid_size);
        let start_y = rect.min.y + (pan.y % grid_size);

        let grid_stroke = Stroke::new(1.0, Color32::from_rgb(28, 32, 44));

        let mut x = start_x;
        while x < rect.max.x {
            painter.line_segment(
                [Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)],
                grid_stroke,
            );
            x += grid_size;
        }

        let mut y = start_y;
        while y < rect.max.y {
            painter.line_segment(
                [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                grid_stroke,
            );
            y += grid_size;
        }
    }
}

fn tensor_space_short_label(space: &TensorSpace) -> String {
    match space {
        TensorSpace::ContinuousVector { dimension, .. } => format!("ContinuousVector({dimension})"),
        TensorSpace::ImageRGB {
            resolution,
            channels,
        } => format!("ImageRGB({}x{}x{})", channels, resolution[0], resolution[1]),
        TensorSpace::Video3D {
            frames,
            resolution,
            channels,
        } => format!(
            "Video3D({}x{}x{}x{})",
            frames, channels, resolution[0], resolution[1]
        ),
        TensorSpace::TimeSeries {
            channels, horizon, ..
        } => format!("TimeSeries(C={channels}, H={horizon})"),
        TensorSpace::GraphRelational {
            node_features,
            edge_features,
            ..
        } => format!("Graph(V={node_features}, E={edge_features})"),
        TensorSpace::DiscreteTokens { vocab_size, .. } => format!("DiscreteTokens(V={vocab_size})"),
        TensorSpace::CoordinateField { domain_dim, .. } => format!("CoordField(dim={domain_dim})"),
        TensorSpace::RigidBodyTrajectory {
            horizon_steps,
            dof_per_step,
        } => format!("Trajectory(H={horizon_steps}, DoF={dof_per_step})"),
        TensorSpace::TabularFeatures {
            num_continuous,
            num_categorical,
            ..
        } => format!("Tabular(cont={num_continuous}, cat={num_categorical})"),
    }
}
