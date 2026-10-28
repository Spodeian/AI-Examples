//! Comprehensive Codecs & Containers: .aid (YAML, JSON, TOML, BSON, RON), .pkg weights containers,
//! .aidpkg unified single-file packages, compressed archives (TAR, TAR.GZ), and universal import/export.

pub mod aid_bson;
pub mod aid_envelope;
pub mod aid_json;
pub mod aid_ron;
pub mod aid_toml;
pub mod aid_yaml;
pub mod aidpkg;
pub mod archive;
pub mod pkg_envelope;
pub mod quantizer;
pub mod universal;

pub use aid_bson::*;
pub use aid_envelope::*;
pub use aid_json::*;
pub use aid_ron::*;
pub use aid_toml::*;
pub use aid_yaml::*;
pub use aidpkg::*;
pub use archive::*;
pub use pkg_envelope::*;
pub use quantizer::*;
pub use universal::*;
