use app::TemplateApp;
use eframe::{App, Storage};
use shared::{AppState, Priority};
use std::collections::HashMap;

#[derive(Default)]
struct MockStorage {
    data: HashMap<String, String>,
}

impl Storage for MockStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.data.insert(key.to_owned(), value);
    }
    fn remove_string(&mut self, key: &str) {
        self.data.remove(key);
    }
    fn flush(&mut self) {}
}

#[test]
fn test_template_app_initialization() {
    let app = TemplateApp::default();
    assert_eq!(app.state.collection.total_count(), 3);
    assert_eq!(app.state.collection.completed_count(), 0);
}

#[test]
fn test_template_app_save_and_load() {
    let mut storage = MockStorage::default();
    let mut app = TemplateApp::default();

    // Add item and toggle
    let new_id = app.state.collection.add(
        "Persistent Task",
        "Should be saved in storage",
        Priority::High,
    );
    app.state.collection.toggle(new_id);

    // Save
    app.save(&mut storage);

    // Verify storage contents
    let serialized = storage
        .get_string(eframe::APP_KEY)
        .expect("App key must exist in storage");
    assert!(serialized.contains("Persistent Task"));

    // Simulate reload from storage
    let loaded_state: AppState = eframe::get_value(&storage, eframe::APP_KEY).unwrap();
    assert_eq!(loaded_state.collection.total_count(), 4);
    assert_eq!(loaded_state.collection.completed_count(), 1);
}

#[test]
fn test_template_app_export_dialog() {
    let mut app = TemplateApp::default();
    app.open_export_dialog(app::ExportFormat::Json);

    assert!(app.show_export_dialog.is_some());
    assert!(
        app.export_text_buffer
            .contains("Explore Cross-Platform State Persistence")
    );
}

#[test]
fn test_template_app_load_multi_tier_json_and_ron() {
    use app::storage_manager::{
        DEDICATED_STORAGE_KEY, deserialize_app_state, load_state_multi_tier,
    };

    let mut original_state = AppState::default();
    original_state
        .collection
        .add("Task A", "Description A", Priority::High);

    // 1. JSON roundtrip
    let json_str = serde_json::to_string(&original_state).unwrap();
    let from_json = deserialize_app_state(&json_str).expect("JSON should deserialize");
    assert_eq!(from_json.collection.total_count(), 4);

    // 2. RON roundtrip
    let ron_str = ron::to_string(&original_state).unwrap();
    let from_ron = deserialize_app_state(&ron_str).expect("RON should deserialize");
    assert_eq!(from_ron.collection.total_count(), 4);

    // 3. Load from dedicated key in storage (JSON)
    let mut storage_dedicated = MockStorage::default();
    storage_dedicated.set_string(DEDICATED_STORAGE_KEY, json_str.clone());
    let loaded_dedicated =
        load_state_multi_tier(Some(&storage_dedicated)).expect("Should load from dedicated key");
    assert_eq!(loaded_dedicated.collection.total_count(), 4);

    // 4. Load from legacy app key in storage (JSON)
    let mut storage_json_app = MockStorage::default();
    storage_json_app.set_string(eframe::APP_KEY, json_str);
    let loaded_json =
        load_state_multi_tier(Some(&storage_json_app)).expect("Should load from app key JSON");
    assert_eq!(loaded_json.collection.total_count(), 4);

    // 5. Load from legacy app key in storage (RON)
    let mut storage_ron_app = MockStorage::default();
    storage_ron_app.set_string(eframe::APP_KEY, ron_str);
    let loaded_ron =
        load_state_multi_tier(Some(&storage_ron_app)).expect("Should load from app key RON");
    assert_eq!(loaded_ron.collection.total_count(), 4);
}

#[test]
fn test_template_app_save_populates_both_keys() {
    use app::storage_manager::{DEDICATED_STORAGE_KEY, load_state_multi_tier};

    let mut storage = MockStorage::default();
    let mut app = TemplateApp::default();
    app.state.collection.add(
        "New Saved Item",
        "Active Persistence Test",
        Priority::Medium,
    );

    app.save(&mut storage);

    // Both eframe::APP_KEY (RON) and DEDICATED_STORAGE_KEY (JSON) should be populated
    assert!(storage.get_string(eframe::APP_KEY).is_some());
    assert!(storage.get_string(DEDICATED_STORAGE_KEY).is_some());

    let loaded = load_state_multi_tier(Some(&storage)).expect("Should restore state successfully");
    assert_eq!(loaded.collection.total_count(), 4);
}

#[test]
fn test_template_app_telemetry_modal_and_data() {
    let mut app = TemplateApp::default();
    assert!(!app.show_telemetry_modal);
    assert!(app.telemetry_data.success_rate > 99.0);
    assert_eq!(app.telemetry_data.channels.len(), 3);

    app.show_telemetry_modal = true;
    assert!(app.show_telemetry_modal);
}

#[test]
fn test_template_app_memory_and_fallback_diagnostics() {
    use app::storage_manager::MEMORY_WORKING_SET_LIMIT;

    let mut app = TemplateApp::default();
    assert_eq!(
        app.storage_diag.memory_capacity_limit,
        MEMORY_WORKING_SET_LIMIT
    );
    assert_eq!(app.storage_diag.memory_items_count, 0);

    // Persisting state updates memory item count and evaluates fallback
    app.persist_state();
    assert_eq!(app.storage_diag.memory_items_count, 3);
    assert!(!app.storage_diag.fallback_tier_active);

    // Simulate item load beyond memory working set capacity
    for i in 0..110 {
        app.state.collection.add(
            format!("Bulk Task {i}"),
            "Memory capacity test",
            Priority::Low,
        );
    }

    app.persist_state();
    assert_eq!(app.storage_diag.memory_items_count, 113);
    // Over memory limit: fallback tier is active to protect memory from overfilling!
    assert!(app.storage_diag.fallback_tier_active);
}

#[test]
fn test_studio_workflow_layer_manipulation_and_pareto() {
    let mut app = TemplateApp::default();

    // Pareto frontier candidates should be pre-computed on initialization
    assert_eq!(app.studio.pareto_candidates.len(), 5);
    let initial_len = app.studio.spec.graph.nodes.len();
    assert!(initial_len >= 2);

    // Select first node and move it down
    let first_id = app.studio.spec.graph.nodes[0].id.clone();
    let second_id = app.studio.spec.graph.nodes[1].id.clone();
    app.studio.select_node(Some(first_id.clone()));
    assert!(app.studio.move_selected_node(1));
    assert_eq!(app.studio.spec.graph.nodes[0].id, second_id);
    assert_eq!(app.studio.spec.graph.nodes[1].id, first_id);

    // Duplicate selected node
    let dup_id = app
        .studio
        .duplicate_selected_node()
        .expect("Should duplicate selected node");
    assert_eq!(app.studio.spec.graph.nodes.len(), initial_len + 1);
    assert_eq!(app.studio.selected_node_id.as_deref(), Some(dup_id.as_str()));

    // Delete duplicated node
    assert!(app.studio.delete_selected_node());
    assert_eq!(app.studio.spec.graph.nodes.len(), initial_len);
    assert!(app.studio.selected_node_id.is_none());

    // Auto-layout resets zoom and pan and positions Input, Layer, and Output cards
    app.studio.zoom_level = 1.75;
    app.studio.auto_layout_nodes();
    assert_eq!(app.studio.zoom_level, 1.0);
    assert!(
        app.studio
            .node_positions
            .keys()
            .any(|k| k.starts_with("__input__:"))
    );
    assert!(
        app.studio
            .node_positions
            .keys()
            .any(|k| k.starts_with("__output__:"))
    );
}

#[test]
fn test_bootstrap_icons_and_operator_overviews() {
    let ctx = eframe::egui::Context::default();
    app::install_bootstrap_icons(&ctx);
    assert!(!app::icons::BOOTSTRAP_ICONS_TTF.is_empty());
    assert_eq!(app::bi::LIGHTNING_CHARGE_FILL, "\u{f46c}");

    // Verify operator overview, recommended next layers, and clickable curriculum URLs
    let app = TemplateApp::default();
    for node in &app.studio.spec.graph.nodes {
        let ov = node.operator.overview();
        assert!(!ov.what_it_does.is_empty());
        assert!(!ov.typical_usage.is_empty());
        assert!(!ov.typically_followed_by.is_empty());
        assert!(node.operator.curriculum_url().starts_with("https://github.com/"));
        assert!(!node.operator.recommended_next_layers().is_empty());
    }

    // Verify file extensions (.aid for native byte format, .yaml for YAML export)
    assert_eq!(
        shared::BlueprintFileFormat::NativeAid.default_extension(),
        "aid"
    );
    assert_eq!(
        shared::BlueprintFileFormat::Yaml.default_extension(),
        "yaml"
    );
}
