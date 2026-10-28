//! Analytical engines: shape inference, optimizer partitioning, analytical roofline modeling, hardware introspection, and transitive parity verification.

pub mod hardware_probe;
pub mod optimizer_partitioner;
pub mod parity_verifier;
pub mod roofline;
pub mod shape_inference;

pub use hardware_probe::*;
pub use optimizer_partitioner::*;
pub use parity_verifier::*;
pub use roofline::*;
pub use shape_inference::*;
