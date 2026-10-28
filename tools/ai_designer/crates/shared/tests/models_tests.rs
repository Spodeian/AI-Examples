use shared::{
    ItemCollection, Priority, export_to_csv, export_to_json, import_from_csv, import_from_json,
};

#[test]
fn test_item_collection_defaults_and_operations() {
    let mut collection = ItemCollection::default();
    assert_eq!(collection.total_count(), 3);
    assert_eq!(collection.completed_count(), 0);
    assert_eq!(collection.completion_ratio(), 0.0);

    let id = collection.add("New Task", "Task Description", Priority::High);
    assert_eq!(collection.total_count(), 4);

    collection.toggle(id);
    assert_eq!(collection.completed_count(), 1);
    assert_eq!(collection.completion_ratio(), 0.25);

    let removed = collection.remove(id);
    assert!(removed);
    assert_eq!(collection.total_count(), 3);
    assert_eq!(collection.completed_count(), 0);
}

#[test]
fn test_json_roundtrip() {
    let collection = ItemCollection::default();
    let json = export_to_json(&collection).expect("Failed to export JSON");
    assert!(json.contains("Explore Cross-Platform State Persistence"));

    let imported = import_from_json(&json).expect("Failed to import JSON");
    assert_eq!(imported.items.len(), collection.items.len());
    assert_eq!(imported.items[0].title, collection.items[0].title);
}

#[test]
fn test_csv_roundtrip() {
    let collection = ItemCollection::default();
    let csv = export_to_csv(&collection);
    assert!(csv.starts_with("id,title,description,priority,completed"));

    let imported = import_from_csv(&csv).expect("Failed to import CSV");
    assert_eq!(imported.items.len(), collection.items.len());
    assert_eq!(imported.items[0].title, collection.items[0].title);
    assert_eq!(imported.items[0].priority, collection.items[0].priority);
}

#[test]
fn test_clear_completed() {
    let mut collection = ItemCollection::default();
    collection.toggle(1);
    assert_eq!(collection.completed_count(), 1);

    collection.clear_completed();
    assert_eq!(collection.total_count(), 2);
    assert_eq!(collection.completed_count(), 0);
}

#[test]
fn test_compressed_bson_roundtrip() {
    let mut collection = ItemCollection::default();
    collection.add(
        "BSON Persistence Item",
        "State compressed with zlib",
        Priority::High,
    );

    let bytes = shared::export_to_compressed_bson(&collection).expect("BSON export failed");
    assert!(!bytes.is_empty());

    let restored = shared::import_from_compressed_bson(&bytes).expect("BSON import failed");
    assert_eq!(restored.total_count(), 4);
    assert_eq!(restored.items[3].title, "BSON Persistence Item");
}

#[test]
fn test_state_backward_compatibility() {
    let legacy_json = r#"{"config":{},"collection":{"items":[]}}"#;
    let app_state: Result<shared::AppState, _> = serde_json::from_str(legacy_json);
    assert!(
        app_state.is_ok(),
        "Should parse legacy state without failure"
    );
    let state = app_state.unwrap();
    assert_eq!(state.config.active_page, shared::AppPage::Designer);
}

#[test]
fn test_app_page_routing() {
    let mut state = shared::AppState::default();
    assert_eq!(state.config.active_page, shared::AppPage::Designer);
    assert_eq!(state.config.active_page.label(), "AI Designer Studio");

    state.config.active_page = shared::AppPage::About;
    assert_eq!(state.config.active_page.label(), "About");

    let serialized = serde_json::to_string(&state).expect("Serialization failed");
    let deserialized: shared::AppState =
        serde_json::from_str(&serialized).expect("Deserialization failed");
    assert_eq!(deserialized.config.active_page, shared::AppPage::About);
}
