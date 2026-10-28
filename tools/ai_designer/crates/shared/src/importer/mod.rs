//! Importer module: Ingesting architectures from Safetensors, HuggingFace, and AI-Examples topics.

pub mod ai_examples_catalog;
pub mod huggingface_import;
pub mod python_ast_import;
pub mod safetensors_import;

pub use ai_examples_catalog::{
    REPO_CATEGORIES, TopicCategory, get_category_by_id, list_category_names,
};
pub use huggingface_import::import_from_huggingface_config;
pub use python_ast_import::{extract_module_classes, import_from_python_source};
pub use safetensors_import::{
    SafetensorsTensorInfo, import_spec_from_safetensors, parse_safetensors_header,
};
