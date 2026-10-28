#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_browser_environment() {
    let window = web_sys::window().expect("Window should exist in browser environment");
    let document = window
        .document()
        .expect("Document should exist in browser environment");
    assert!(document.body().is_some());
}

#[wasm_bindgen_test]
fn test_local_storage_availability() {
    let window = web_sys::window().expect("Window should exist");
    let storage = window
        .local_storage()
        .ok()
        .flatten()
        .expect("LocalStorage should be accessible");
    storage
        .set_item("test_key", "test_val")
        .expect("Write should succeed");
    assert_eq!(
        storage.get_item("test_key").unwrap().as_deref(),
        Some("test_val")
    );
    storage
        .remove_item("test_key")
        .expect("Removal should succeed");
}
