//! Unified multi-tiered storage engine, persistence manager, PWA install bridge, and diagnostics for template app.

use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use spodeian_cache::ContentAddressedStorage;
use spodeian_cache::PreferentialRouter;
#[allow(unused_imports)]
use spodeian_cache::StorageTier;
use thiserror::Error;
#[allow(unused_imports)]
use tracing::{error, info, warn};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Errors arising during state persistence or multi-tiered storage operations.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Storage content is empty")]
    EmptyContent,
    #[error("Failed to deserialize state: JSON error: {0}; RON error: {1}")]
    DeserializationFailed(serde_json::Error, ron::error::SpannedError),
    #[error("All browser storage tiers failed")]
    AllTiersFailed,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Cache error: {0}")]
    Cache(#[from] spodeian_cache::CacheError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StorageBackend {
    #[default]
    LocalStorage,
    IndexedDb,
    CacheApi,
    NativeCas,
    MemoryOnly,
}

impl StorageBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalStorage => "Local Storage (Fast Tier)",
            Self::IndexedDb => "IndexedDB (Structured Relational Tier)",
            Self::CacheApi => "Cache API (Large Binary Weights Tier)",
            Self::NativeCas => "Native CAS (Content-Addressed Disk Tier)",
            Self::MemoryOnly => "In-Memory Only (Ephemeral)",
        }
    }
}

/// Target working capacity of items held directly in RAM for sub-microsecond latency.
/// Once items exceed this threshold, the persistent fallback tier (IndexedDB on web, CAS on desktop)
/// retains the complete durable dataset, preventing memory from overfilling.
pub const MEMORY_WORKING_SET_LIMIT: usize = 100;

#[derive(Debug, Clone)]
pub struct StorageDiagnostics {
    pub is_persisted: Option<bool>,
    pub pwa_install_available: bool,
    pub is_pwa_installed: bool,
    pub backend: StorageBackend,
    pub quota_exceeded: bool,
    pub idb_active: bool,
    pub usage_bytes: u64,
    pub quota_bytes: u64,
    pub memory_items_count: usize,
    pub memory_capacity_limit: usize,
    pub fallback_tier_active: bool,
}

impl Default for StorageDiagnostics {
    fn default() -> Self {
        Self {
            is_persisted: None,
            pwa_install_available: false,
            is_pwa_installed: false,
            backend: StorageBackend::default(),
            quota_exceeded: false,
            idb_active: false,
            usage_bytes: 0,
            quota_bytes: 0,
            memory_items_count: 0,
            memory_capacity_limit: MEMORY_WORKING_SET_LIMIT,
            fallback_tier_active: false,
        }
    }
}

/// Query current storage persistence and PWA status from browser environment
#[allow(unused_mut)]
pub fn query_storage_diagnostics() -> StorageDiagnostics {
    let mut diag = StorageDiagnostics::default();

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            // Check if PWA is installed or installable
            if let Ok(val) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__pwaInstallAvailable"),
            ) {
                diag.pwa_install_available = val.as_bool().unwrap_or(false);
            }
            if let Ok(val) =
                js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__pwaInstalled"))
            {
                diag.is_pwa_installed = val.as_bool().unwrap_or(false);
            }

            // Check persistence state
            if let Ok(val) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__storagePersisted"),
            ) {
                if let Some(b) = val.as_bool() {
                    diag.is_persisted = Some(b);
                }
            }
        }
    }

    diag
}

/// Request persistent storage from the browser (immune to automatic eviction)
pub fn request_persistent_storage() {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(func) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__requestPersistentStorage"),
            ) {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let _ = func.call0(&window);
                    info!("Triggered __requestPersistentStorage from template UI");
                }
            }
        }
    }
}

/// Trigger the native PWA installation prompt
pub fn trigger_pwa_install() {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(func) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__triggerPWAInstall"),
            ) {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let _ = func.call0(&window);
                    info!("Triggered __triggerPWAInstall from template UI");
                }
            }
        }
    }
}

pub const DEDICATED_STORAGE_KEY: &str = "serverless_template_app_state";

/// Robust dual-format deserializer for AppState, attempting JSON first and falling back to RON.
pub fn deserialize_app_state(content: &str) -> Result<shared::AppState, StorageError> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err(StorageError::EmptyContent);
    }

    // 1. Attempt JSON deserialization
    match serde_json::from_str::<shared::AppState>(trimmed) {
        Ok(state) => Ok(state),
        Err(json_err) => {
            // 2. Attempt RON deserialization
            match ron::from_str::<shared::AppState>(trimmed) {
                Ok(state) => Ok(state),
                Err(ron_err) => Err(StorageError::DeserializationFailed(json_err, ron_err)),
            }
        }
    }
}

/// Multi-tiered loader for AppState.
/// Checks window.localStorage (on wasm32) and eframe::Storage across both dedicated and legacy keys,
/// supporting both JSON and RON formats seamlessly.
pub fn load_state_multi_tier(storage: Option<&dyn eframe::Storage>) -> Option<shared::AppState> {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(local_storage)) = window.local_storage() {
                // Tier 1: Check dedicated key in browser localStorage
                if let Ok(Some(content)) = local_storage.get_item(DEDICATED_STORAGE_KEY) {
                    match deserialize_app_state(&content) {
                        Ok(state) => {
                            info!(
                                "Successfully restored AppState from localStorage [{}]",
                                DEDICATED_STORAGE_KEY
                            );
                            return Some(state);
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse AppState from localStorage [{}]: {}",
                                DEDICATED_STORAGE_KEY, e
                            );
                        }
                    }
                }

                // Tier 2: Check standard 'app' key in browser localStorage (fallback/legacy)
                if let Ok(Some(content)) = local_storage.get_item(eframe::APP_KEY) {
                    match deserialize_app_state(&content) {
                        Ok(state) => {
                            info!(
                                "Successfully restored AppState from localStorage [{}]",
                                eframe::APP_KEY
                            );
                            return Some(state);
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse AppState from localStorage [{}]: {}",
                                eframe::APP_KEY,
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    // Tier 3: Check eframe::Storage
    if let Some(storage) = storage {
        // Check dedicated key in eframe storage
        if let Some(raw) = storage.get_string(DEDICATED_STORAGE_KEY) {
            match deserialize_app_state(&raw) {
                Ok(state) => {
                    info!(
                        "Successfully restored AppState from eframe::Storage [{}]",
                        DEDICATED_STORAGE_KEY
                    );
                    return Some(state);
                }
                Err(e) => {
                    warn!(
                        "Failed to parse AppState from eframe::Storage [{}]: {}",
                        DEDICATED_STORAGE_KEY, e
                    );
                }
            }
        }

        // Check 'app' key string in eframe storage
        if let Some(raw) = storage.get_string(eframe::APP_KEY) {
            match deserialize_app_state(&raw) {
                Ok(state) => {
                    info!(
                        "Successfully restored AppState from eframe::Storage [{}]",
                        eframe::APP_KEY
                    );
                    return Some(state);
                }
                Err(e) => {
                    warn!(
                        "Failed to parse AppState from eframe::Storage [{}]: {}",
                        eframe::APP_KEY,
                        e
                    );
                }
            }
        }

        // Check native eframe::get_value (RON deserializer)
        if let Some(state) = eframe::get_value::<shared::AppState>(storage, eframe::APP_KEY) {
            info!("Successfully restored AppState from eframe::get_value (RON).");
            return Some(state);
        }
    }

    None
}

/// Save state using preferential multi-tiered routing:
/// - Cache API for large binaries / models (> 512 KB)
/// - IndexedDB for structured relational states
/// - LocalStorage for lightweight config (< 16 KB)
/// - ContentAddressedStorage on native desktop & mobile
pub fn save_state_multi_tier(key: &str, json_str: &str) -> Result<StorageBackend, StorageError> {
    let size = json_str.len();
    let is_large_or_binary = size > 512 * 1024;
    let recommended_tier =
        PreferentialRouter::determine_tier(size, "application/json", is_large_or_binary);
    let _ = (key, &recommended_tier);

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            // Tier 1 (Large/Binary): If Cache API is recommended or payload is large
            if recommended_tier == StorageTier::CacheApi {
                if let Ok(func) = js_sys::Reflect::get(
                    &window,
                    &wasm_bindgen::JsValue::from_str("__saveToCacheApi"),
                ) {
                    if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                        let k = wasm_bindgen::JsValue::from_str(key);
                        let v = wasm_bindgen::JsValue::from_str(json_str);
                        let _ = func.call2(&window, &k, &v);
                        info!("Preferentially stored large asset to Cache API [{}]", key);
                        return Ok(StorageBackend::CacheApi);
                    }
                }
            }

            // Tier 2: Try localStorage for fast synchronous session data if small (< 16 KB)
            let mut local_storage_saved = false;
            if size < 16 * 1024 {
                if let Ok(Some(storage)) = window.local_storage() {
                    if storage.set_item(key, json_str).is_ok() {
                        local_storage_saved = true;
                    }
                }
            }

            // Tier 3: Always synchronize to IndexedDB as the durable structured fallback tier
            // to protect memory from overfilling and ensure multi-megabyte safety.
            if let Ok(func) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__saveToIndexedDB"),
            ) {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let k = wasm_bindgen::JsValue::from_str(key);
                    let v = wasm_bindgen::JsValue::from_str(json_str);
                    let _ = func.call2(&window, &k, &v);
                    info!("Synchronized state to IndexedDB fallback tier [{}]", key);
                    if !local_storage_saved {
                        return Ok(StorageBackend::IndexedDb);
                    }
                }
            }

            if local_storage_saved {
                return Ok(StorageBackend::LocalStorage);
            }

            return Err(StorageError::AllTiersFailed);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let cas_dir = std::env::temp_dir().join("egui_template_cas_store");
        if let Ok(cas) = ContentAddressedStorage::new(&cas_dir) {
            let _ = cas.put(json_str.as_bytes());
            return Ok(StorageBackend::NativeCas);
        }
    }

    Ok(StorageBackend::MemoryOnly)
}

pub use spodeian_web_utils::{trigger_binary_download, trigger_text_download};
