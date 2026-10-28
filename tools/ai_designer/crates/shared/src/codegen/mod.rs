//! Multi-Backend Code & Project Generator: PyTorch 2, JAX/Equinox, Rust Candle, and Fused Hardware Kernels.

pub mod candle_gen;
pub mod harness_gen;
pub mod jax_gen;
pub mod kernel_gen;
pub mod pytorch_gen;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::spec::AiDesignSpec;

pub use candle_gen::*;
pub use harness_gen::*;
pub use jax_gen::*;
pub use kernel_gen::*;
pub use pytorch_gen::*;

/// Complete bundle of generated multi-backend source code and assets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectBundle {
    pub model_id: String,
    pub files: HashMap<String, String>,
}

impl ProjectBundle {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            files: HashMap::new(),
        }
    }

    pub fn insert(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.files.insert(path.into(), content.into());
    }

    /// Writes all bundled source files to a target directory on disk.
    pub fn save_to_disk(&self, base_dir: &Path) -> std::io::Result<()> {
        for (rel_path, content) in &self.files {
            let full_path = base_dir.join(rel_path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&full_path, content)?;
        }
        Ok(())
    }
}

/// Generates the complete multi-backend code package for a given AiDesignSpec.
pub fn generate_all_backends(spec: &AiDesignSpec) -> ProjectBundle {
    let mut bundle = ProjectBundle::new(&spec.metadata.model_id);

    // 1. PyTorch 2 Target
    let pt_files = generate_pytorch_project(spec);
    for (name, content) in pt_files {
        bundle.insert(format!("pytorch/{}", name), content);
    }

    // 2. JAX / Equinox Target
    let jax_files = generate_jax_project(spec);
    for (name, content) in jax_files {
        bundle.insert(format!("jax/{}", name), content);
    }

    // 3. Rust Candle Target
    let candle_files = generate_candle_project(spec);
    for (name, content) in candle_files {
        bundle.insert(format!("candle/{}", name), content);
    }

    // 4. Custom Fused Hardware Kernels (WebGPU WGSL + CPU SIMD + Triton)
    let kernel_files = generate_kernel_bundle(spec);
    for (name, content) in kernel_files {
        bundle.insert(name, content);
    }

    // 5. Cross-Backend Parity & Benchmark Harness
    let harness_py = generate_verification_harness(spec);
    bundle.insert("verify_and_bench.py".to_string(), harness_py);

    bundle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_project_bundle_generation() {
        let spec = AiDesignSpec::new("test_bundle", "Test Bundle Model");
        let bundle = generate_all_backends(&spec);

        assert!(bundle.files.contains_key("pytorch/model.py"));
        assert!(bundle.files.contains_key("pytorch/train.py"));
        assert!(bundle.files.contains_key("jax/model.py"));
        assert!(bundle.files.contains_key("candle/Cargo.toml"));
        assert!(bundle.files.contains_key("candle/src/model.rs"));
        assert!(bundle.files.contains_key("kernels/shaders.wgsl"));
        assert!(bundle.files.contains_key("kernels/cpu_simd.rs"));
        assert!(bundle.files.contains_key("verify_and_bench.py"));
    }
}
