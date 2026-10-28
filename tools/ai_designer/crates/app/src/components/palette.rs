//! Operator Catalog Palette, Starter Blueprint Switcher, and Auto-Architect Trigger.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{self, Color32, RichText, ScrollArea, Ui};
use shared::{NodeSpec, OperatorKind, PrecisionFormat, REPO_CATEGORIES};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaletteTab {
    #[default]
    Layers,
    RepositoryCatalog,
}

pub struct PaletteView {
    pub current_tab: PaletteTab,
    pub search_query: String,
    pub selected_category: Option<&'static str>,
    pub selected_complexity: Option<&'static str>,
}

impl Default for PaletteView {
    fn default() -> Self {
        Self {
            current_tab: PaletteTab::Layers,
            search_query: String::new(),
            selected_category: None,
            selected_complexity: None,
        }
    }
}

pub struct StarterTemplate {
    pub file_name: &'static str,
    pub display_name: &'static str,
    pub yaml_content: &'static str,
}

pub const TEMPLATES: &[StarterTemplate] = &[
    StarterTemplate {
        file_name: "01_spatiotemporal_dit_flow.aid.yaml",
        display_name: "01. Spatiotemporal DiT Flow",
        yaml_content: include_str!("../../../../templates/01_spatiotemporal_dit_flow.aid.yaml"),
    },
    StarterTemplate {
        file_name: "02_bimanual_diffusion_act_policy.aid.yaml",
        display_name: "02. Bimanual Diffusion ACT Policy",
        yaml_content: include_str!(
            "../../../../templates/02_bimanual_diffusion_act_policy.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "03_streaming_codec_mamba.aid.yaml",
        display_name: "03. Streaming Codec Mamba",
        yaml_content: include_str!("../../../../templates/03_streaming_codec_mamba.aid.yaml"),
    },
    StarterTemplate {
        file_name: "04_symplectic_fourier_neural_operator.aid.yaml",
        display_name: "04. Symplectic Fourier Neural Operator",
        yaml_content: include_str!(
            "../../../../templates/04_symplectic_fourier_neural_operator.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "05_equivariant_biomolecular_flow.aid.yaml",
        display_name: "05. Equivariant Biomolecular Flow",
        yaml_content: include_str!(
            "../../../../templates/05_equivariant_biomolecular_flow.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "06_patch_xlstm_probabilistic_forecaster.aid.yaml",
        display_name: "06. Patch xLSTM Forecaster",
        yaml_content: include_str!(
            "../../../../templates/06_patch_xlstm_probabilistic_forecaster.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "07_monotonic_ple_tabpfn.aid.yaml",
        display_name: "07. Monotonic PLE TabPFN",
        yaml_content: include_str!("../../../../templates/07_monotonic_ple_tabpfn.aid.yaml"),
    },
    StarterTemplate {
        file_name: "08_hybrid_jamba_mla_verifier.aid.yaml",
        display_name: "08. Hybrid Jamba-2 MLA Verifier",
        yaml_content: include_str!("../../../../templates/08_hybrid_jamba_mla_verifier.aid.yaml"),
    },
    StarterTemplate {
        file_name: "09_autograd_custom_kernel_backward.aid.yaml",
        display_name: "09. Autograd Custom Kernel Backward",
        yaml_content: include_str!(
            "../../../../templates/09_autograd_custom_kernel_backward.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "10_optimal_transport_sinkhorn_matcher.aid.yaml",
        display_name: "10. Optimal Transport Sinkhorn Matcher",
        yaml_content: include_str!(
            "../../../../templates/10_optimal_transport_sinkhorn_matcher.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "11_muon_wsd_accelerated_transformer.aid.yaml",
        display_name: "11. Muon WSD Accelerated Transformer",
        yaml_content: include_str!(
            "../../../../templates/11_muon_wsd_accelerated_transformer.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "12_safe_cbf_qp_diffusion_policy.aid.yaml",
        display_name: "12. Safe CBF-QP Diffusion Policy",
        yaml_content: include_str!(
            "../../../../templates/12_safe_cbf_qp_diffusion_policy.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "13_structural_causal_neural_scm.aid.yaml",
        display_name: "13. Structural Causal Neural SCM",
        yaml_content: include_str!(
            "../../../../templates/13_structural_causal_neural_scm.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "14_monosemantic_sparse_autoencoder_sae.aid.yaml",
        display_name: "14. Monosemantic Sparse Autoencoder (SAE)",
        yaml_content: include_str!(
            "../../../../templates/14_monosemantic_sparse_autoencoder_sae.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "15_differentially_private_dp_sgd.aid.yaml",
        display_name: "15. Differentially Private DP-SGD",
        yaml_content: include_str!(
            "../../../../templates/15_differentially_private_dp_sgd.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "16_torch_compile_inductor_fastpath.aid.yaml",
        display_name: "16. Torch Compile Inductor Fastpath",
        yaml_content: include_str!(
            "../../../../templates/16_torch_compile_inductor_fastpath.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "17_bitnet_ternary_1_58_quant.aid.yaml",
        display_name: "17. BitNet Ternary 1.58b Quant",
        yaml_content: include_str!("../../../../templates/17_bitnet_ternary_1_58_quant.aid.yaml"),
    },
    StarterTemplate {
        file_name: "18_awq_activation_aware_quantizer.aid.yaml",
        display_name: "18. AWQ Activation-Aware Quantizer",
        yaml_content: include_str!(
            "../../../../templates/18_awq_activation_aware_quantizer.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "19_speculative_medusa_draft_decoder.aid.yaml",
        display_name: "19. Speculative Medusa Draft Decoder",
        yaml_content: include_str!(
            "../../../../templates/19_speculative_medusa_draft_decoder.aid.yaml"
        ),
    },
    StarterTemplate {
        file_name: "20_fsdp2_ring_attention_transformer.aid.yaml",
        display_name: "20. FSDP2 Ring Attention Transformer",
        yaml_content: include_str!(
            "../../../../templates/20_fsdp2_ring_attention_transformer.aid.yaml"
        ),
    },
];

pub fn get_template_by_filename(filename: &str) -> Option<&'static str> {
    TEMPLATES
        .iter()
        .find(|t| t.file_name == filename)
        .map(|t| t.yaml_content)
}

impl PaletteView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ui.vertical(|ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new(format!("{} Operator & Topic Catalog", bi::LAYERS))
                        .size(15.0)
                        .strong(),
                );
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.current_tab,
                    PaletteTab::Layers,
                    format!("{} Layers & Starters", bi::LAYERS),
                );
                ui.selectable_value(
                    &mut self.current_tab,
                    PaletteTab::RepositoryCatalog,
                    format!("{} Repo Catalog (18)", bi::BOOK),
                );
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(bi::SEARCH).color(Color32::from_rgb(160, 175, 200)));
                ui.add(
                    egui::TextEdit::singleline(&mut self.search_query)
                        .hint_text("Search layers, ops, or topics...")
                        .desired_width((ui.available_width() - 32.0).max(100.0)),
                );
                if !self.search_query.is_empty()
                    && ui
                        .small_button(bi::X_LG)
                        .on_hover_text("Clear search query")
                        .clicked()
                {
                    self.search_query.clear();
                }
            });

            ui.add_space(6.0);
            ui.separator();

            match self.current_tab {
                PaletteTab::Layers => self.render_layers_tab(ui, state),
                PaletteTab::RepositoryCatalog => self.render_catalog_tab(ui, state),
            }
        });
    }

    fn render_layers_tab(&mut self, ui: &mut Ui, state: &mut StudioState) {
        // Starter Blueprint Selector (All 20 Blueprints)
        ui.label(
            RichText::new("Starter Blueprints (20)")
                .size(11.5)
                .color(Color32::from_rgb(180, 180, 180)),
        );
        egui::ComboBox::from_id_salt("template_selector")
            .selected_text(&state.spec.metadata.display_name)
            .width(ui.available_width() - 10.0)
            .show_ui(ui, |ui| {
                for tmpl in TEMPLATES {
                    let is_active = state.spec.metadata.display_name.contains(tmpl.display_name)
                        || state.spec.metadata.model_id
                            == tmpl
                                .file_name
                                .trim_end_matches(".aid.yaml")
                                .trim_end_matches(".yaml")
                                .trim_start_matches(|c: char| c.is_ascii_digit() || c == '_');
                    if ui.selectable_label(is_active, tmpl.display_name).clicked() {
                        load_template(state, tmpl.yaml_content);
                    }
                }
            });

        ui.add_space(6.0);
        if ui
            .button(
                RichText::new(format!("{} Auto-Architect Synthesis", bi::MAGIC))
                    .color(Color32::from_rgb(120, 220, 140))
                    .strong(),
            )
            .on_hover_text("Automatically synthesize a backbone architecture from declared I/O ports")
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

        ui.add_space(8.0);
        ui.separator();

        // Layer Catalog Filters (Category + Complexity/Paradigm)
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} Layer Filters", bi::FUNNEL))
                    .size(12.0)
                    .color(Color32::from_rgb(180, 195, 220))
                    .strong(),
            );
            if self.selected_category.is_some()
                || self.selected_complexity.is_some()
                || !self.search_query.is_empty()
            {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button(format!("{} Reset", bi::ARROW_COUNTERCLOCKWISE))
                        .on_hover_text("Clear all active layer filters")
                        .clicked()
                    {
                        self.selected_category = None;
                        self.selected_complexity = None;
                        self.search_query.clear();
                    }
                });
            }
        });

        ui.add_space(4.0);
        let cat_label = self.selected_category.unwrap_or("All Categories (6)");
        egui::ComboBox::from_id_salt("palette_category_filter")
            .selected_text(cat_label)
            .width(ui.available_width() - 10.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(self.selected_category.is_none(), "All Categories (6)")
                    .clicked()
                {
                    self.selected_category = None;
                }
                for &cat in &[
                    "Attention & Memory",
                    "State Space & Recurrent",
                    "Feedforward & Splines",
                    "Spectral & Physics",
                    "Norms & Activations",
                    "Routing & MoE",
                ] {
                    if ui
                        .selectable_label(self.selected_category == Some(cat), cat)
                        .clicked()
                    {
                        self.selected_category = Some(cat);
                    }
                }
            });

        ui.add_space(4.0);
        let comp_label = self.selected_complexity.unwrap_or("All Complexity / Paradigms");
        egui::ComboBox::from_id_salt("palette_complexity_filter")
            .selected_text(comp_label)
            .width(ui.available_width() - 10.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(
                        self.selected_complexity.is_none(),
                        "All Complexity / Paradigms",
                    )
                    .clicked()
                {
                    self.selected_complexity = None;
                }
                for &comp in &[
                    "Linear O(T) Sequence",
                    "KV-Compressed Attention",
                    "Quadratic / Global Mixing",
                    "Physics & Geometric",
                    "Sparse / Conditional",
                    "Dense Projection",
                    "Shape-Preserving",
                    "Optimization-as-Architecture",
                ] {
                    if ui
                        .selectable_label(self.selected_complexity == Some(comp), comp)
                        .clicked()
                    {
                        self.selected_complexity = Some(comp);
                    }
                }
            });

        ui.add_space(6.0);
        ui.separator();

        ScrollArea::vertical().show(ui, |ui| {
            self.render_category(
                ui,
                state,
                "Attention & Memory",
                &[
                    (
                        "Multi-Head Attention",
                        OperatorKind::MHA {
                            num_heads: 8,
                            d_k: 64,
                            d_v: 64,
                            dropout: 0.0,
                        },
                    ),
                    (
                        "Multi-Head Latent Attn",
                        OperatorKind::MLA {
                            num_heads: 16,
                            d_latent_kv: 512,
                            d_rope: 64,
                            d_v: 128,
                        },
                    ),
                    (
                        "Grouped Query Attn",
                        OperatorKind::GQA {
                            num_q_heads: 8,
                            num_kv_heads: 2,
                            d_k: 64,
                        },
                    ),
                    (
                        "Differential Attention",
                        OperatorKind::DiffAttn {
                            num_heads: 8,
                            d_k: 64,
                        },
                    ),
                    (
                        "Linear Kernel Attention",
                        OperatorKind::LinearAttention {
                            num_heads: 8,
                            d_k: 64,
                            feature_map: "elu+1".to_string(),
                        },
                    ),
                    (
                        "Sliding Window Sink",
                        OperatorKind::SlidingWindowSink {
                            window_size: 512,
                            num_sink_tokens: 4,
                        },
                    ),
                ],
            );

            self.render_category(
                ui,
                state,
                "State Space & Recurrent",
                &[
                    (
                        "Mamba-2 SSD (Duality)",
                        OperatorKind::Mamba2SSD {
                            d_model: 512,
                            d_state: 64,
                            d_conv: 4,
                            expand: 2,
                        },
                    ),
                    (
                        "Mamba S6 Selective",
                        OperatorKind::MambaS6 {
                            d_model: 512,
                            d_state: 16,
                            d_conv: 4,
                            expand: 2,
                        },
                    ),
                    (
                        "xLSTM (Matrix Memory)",
                        OperatorKind::xLSTM {
                            d_model: 512,
                            num_heads: 4,
                            variant: "mLSTM".to_string(),
                        },
                    ),
                    (
                        "Linear Recurrent Unit",
                        OperatorKind::LRU {
                            d_model: 512,
                            d_state: 64,
                        },
                    ),
                    (
                        "Titans Neural Memory",
                        OperatorKind::TitansMemory {
                            d_model: 512,
                            memory_depth: 2,
                        },
                    ),
                    (
                        "Next-Gen Reservoir",
                        OperatorKind::NextGenReservoir {
                            delay_taps: 4,
                            poly_degree: 2,
                        },
                    ),
                ],
            );

            self.render_category(
                ui,
                state,
                "Feedforward & Splines",
                &[
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: 512,
                            hidden_features: 1376,
                        },
                    ),
                    (
                        "Dense Linear",
                        OperatorKind::Linear {
                            in_features: 512,
                            out_features: 512,
                            bias: true,
                        },
                    ),
                    (
                        "KAN Spline Layer",
                        OperatorKind::KANLayer {
                            in_features: 64,
                            out_features: 64,
                            grid_size: 5,
                            spline_order: 3,
                        },
                    ),
                    (
                        "Token Embedding",
                        OperatorKind::Embedding {
                            num_embeddings: 32000,
                            embedding_dim: 512,
                        },
                    ),
                    (
                        "Unrolled LISTA Solver",
                        OperatorKind::LISTA {
                            num_unrolled_steps: 10,
                            lambda: 0.1,
                        },
                    ),
                    (
                        "Deep Equilibrium (DEQ)",
                        OperatorKind::DEQ {
                            d_model: 512,
                            max_iters: 15,
                            tolerance: 1e-4,
                        },
                    ),
                ],
            );

            self.render_category(
                ui,
                state,
                "Spectral & Physics",
                &[
                    (
                        "Fourier Neural Operator",
                        OperatorKind::FourierNeuralOperator {
                            modes: vec![16, 16],
                            width: 64,
                        },
                    ),
                    (
                        "Wavelet Conv (WTConv)",
                        OperatorKind::WTConv {
                            in_channels: 64,
                            out_channels: 64,
                            wavelet_levels: 3,
                        },
                    ),
                    (
                        "Equivariant Graph GNN",
                        OperatorKind::EGNN {
                            node_dim: 128,
                            edge_dim: 64,
                        },
                    ),
                    (
                        "Evoformer Triangle",
                        OperatorKind::EvoformerTriangle { d_pair: 128 },
                    ),
                    (
                        "Invariant Point Attn",
                        OperatorKind::InvariantPointAttention {
                            num_heads: 8,
                            num_points: 4,
                        },
                    ),
                    (
                        "ConvNeXt V2 Block",
                        OperatorKind::ConvNeXtV2Block {
                            dim: 128,
                            kernel_size: 7,
                            mlp_ratio: 4.0,
                        },
                    ),
                ],
            );

            self.render_category(
                ui,
                state,
                "Norms & Activations",
                &[
                    (
                        "RMSNorm Layer",
                        OperatorKind::RMSNorm {
                            dim: 512,
                            eps: 1e-5,
                        },
                    ),
                    (
                        "LayerNorm Layer",
                        OperatorKind::LayerNorm {
                            dim: 512,
                            eps: 1e-5,
                        },
                    ),
                    ("QK-Norm Guard", OperatorKind::QKNorm { d_head: 64 }),
                    (
                        "Logit Soft-Cap",
                        OperatorKind::LogitSoftCap { cap_val: 30.0 },
                    ),
                    (
                        "RevIN Instance Norm",
                        OperatorKind::RevINLayer { channels: 64 },
                    ),
                    (
                        "Snake Periodic Activation",
                        OperatorKind::Snake { frequency: 1.0 },
                    ),
                ],
            );

            self.render_category(
                ui,
                state,
                "Routing & MoE",
                &[
                    (
                        "DeepSeek Sparse MoE",
                        OperatorKind::DeepSeekMoE {
                            total_experts: 16,
                            routed_k: 4,
                            shared_experts: 2,
                            d_model: 512,
                            d_ffn: 768,
                        },
                    ),
                    (
                        "Slot Attention",
                        OperatorKind::SlotAttention {
                            num_slots: 8,
                            slot_dim: 64,
                            iters: 3,
                        },
                    ),
                    (
                        "Mixture-of-Depths",
                        OperatorKind::MixtureOfDepths {
                            capacity_factor: 0.5,
                        },
                    ),
                    ("Entmax 1.5 Sparse Gate", OperatorKind::Entmax15Gate),
                    ("Sparsemax Gate", OperatorKind::SparsemaxGate),
                    (
                        "Hard Concrete L0 Gate",
                        OperatorKind::HardConcreteL0 { dim: 128 },
                    ),
                ],
            );
        });
    }

    fn render_catalog_tab(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ui.label(
            RichText::new("Click any category or topic to open its reference in AI-Examples:")
                .size(11.0)
                .color(Color32::from_rgb(175, 185, 200)),
        );
        ui.add_space(4.0);

        let query = self.search_query.to_lowercase();

        ScrollArea::vertical().show(ui, |ui| {
            for cat in REPO_CATEGORIES {
                let matches_name = cat.name.to_lowercase().contains(&query);
                let matches_desc = cat.description.to_lowercase().contains(&query);
                let matches_id = cat.id.to_lowercase().contains(&query);
                let matches_topic = cat
                    .representative_topics
                    .iter()
                    .any(|t| t.to_lowercase().contains(&query));
                let matches_starter = cat
                    .recommended_starter
                    .is_some_and(|s| s.to_lowercase().contains(&query));

                if !query.is_empty()
                    && !matches_name
                    && !matches_desc
                    && !matches_id
                    && !matches_topic
                    && !matches_starter
                {
                    continue;
                }

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let cat_url = cat.category_url();
                        ui.hyperlink_to(
                            RichText::new(format!(
                                "{} [{:02}] {} {}",
                                bi::FOLDER2_OPEN,
                                cat.index,
                                cat.name,
                                bi::BOX_ARROW_UP_RIGHT
                            ))
                            .strong()
                            .color(Color32::from_rgb(100, 190, 255)),
                            &cat_url,
                        )
                        .on_hover_text(format!("Open category directory: {}", cat.id));
                    });

                    ui.label(
                        RichText::new(cat.description)
                            .size(11.0)
                            .color(Color32::from_rgb(170, 175, 185)),
                    );

                    ui.horizontal_wrapped(|ui| {
                        for topic in cat.representative_topics {
                            let topic_url = cat.topic_url(topic);
                            ui.hyperlink_to(
                                RichText::new(format!("{} {topic}", bi::LINK_45DEG))
                                    .font(egui::FontId::monospace(10.0))
                                    .color(Color32::from_rgb(140, 215, 185)),
                                &topic_url,
                            )
                            .on_hover_text(format!("Open '{topic}' in AI-Examples ({})", cat.id));
                        }
                    });

                    if let Some(starter) = cat.recommended_starter {
                        let clean_starter = starter.trim_end_matches(".aid.yaml").trim_end_matches(".yaml");
                        ui.horizontal(|ui| {
                            if ui
                                .button(
                                    RichText::new(format!(
                                        "{} Load Blueprint ({clean_starter})",
                                        bi::ROCKET_TAKEOFF
                                    ))
                                    .size(11.0)
                                    .color(Color32::from_rgb(120, 220, 140)),
                                )
                                .clicked()
                                && let Some(content) = get_template_by_filename(starter)
                            {
                                load_template(state, content);
                            }
                        });
                    }
                });
                ui.add_space(4.0);
            }
        });
    }

    fn render_category(
        &mut self,
        ui: &mut Ui,
        state: &mut StudioState,
        category: &'static str,
        items: &[(&'static str, OperatorKind)],
    ) {
        if let Some(sel_cat) = self.selected_category
            && sel_cat != category
        {
            return;
        }

        let query = self.search_query.to_lowercase();
        let matching_items: Vec<&(&'static str, OperatorKind)> = items
            .iter()
            .filter(|(name, op)| {
                let ov = op.overview();
                if let Some(sel_comp) = self.selected_complexity
                    && ov.complexity_tag != sel_comp
                {
                    return false;
                }
                if !query.is_empty() {
                    let match_name = name.to_lowercase().contains(&query);
                    let match_cat = category.to_lowercase().contains(&query);
                    let match_desc = ov.what_it_does.to_lowercase().contains(&query);
                    let match_comp = ov.complexity_tag.to_lowercase().contains(&query);
                    if !match_name && !match_cat && !match_desc && !match_comp {
                        return false;
                    }
                }
                true
            })
            .collect();

        if matching_items.is_empty() {
            return;
        }

        egui::CollapsingHeader::new(
            RichText::new(format!("{category} ({})", matching_items.len()))
                .size(12.5)
                .strong(),
        )
        .default_open(true)
        .show(ui, |ui| {
            for (name, op) in matching_items {
                let ov = op.overview();
                let url = op.curriculum_url();
                ui.horizontal(|ui| {
                    if ui
                        .small_button(
                            RichText::new(bi::PLUS_LG)
                                .color(Color32::from_rgb(120, 225, 155))
                                .strong(),
                        )
                        .on_hover_text(format!("Append '{name}' to computation graph"))
                        .clicked()
                    {
                        let id = format!(
                            "{}_{}",
                            name.to_lowercase()
                                .replace([' ', '-'], "_")
                                .replace(['(', ')'], ""),
                            state.spec.graph.nodes.len() + 1
                        );
                        state.spec.graph.nodes.push(NodeSpec {
                            id: id.clone(),
                            name: name.to_string(),
                            operator: op.clone(),
                            precision: Some(PrecisionFormat::BF16),
                            metadata: Default::default(),
                        });
                        state.auto_layout_nodes();
                        state.select_node(Some(id));
                        state.refresh_analyses();
                    }

                    ui.label(*name).on_hover_text(format!(
                        "{}\n\nTypical usage: {}\nUsually followed by: {}",
                        ov.what_it_does, ov.typical_usage, ov.typically_followed_by
                    ));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.hyperlink_to(
                            RichText::new(bi::BOOK)
                                .size(11.0)
                                .color(Color32::from_rgb(110, 190, 250)),
                            &url,
                        )
                        .on_hover_text(format!(
                            "Open curriculum reference: {}",
                            op.curriculum_slug()
                        ));
                    });
                });
            }
        });
    }
}

fn load_template(state: &mut StudioState, yaml_str: &str) {
    if let Ok(spec) = shared::format::aid_yaml::from_yaml_str(yaml_str) {
        *state = StudioState::new_with_starter(spec);
    }
}
