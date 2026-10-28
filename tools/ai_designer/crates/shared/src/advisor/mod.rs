//! Intelligent I/O-Driven Advisory Engine, Failure-Mode Diagnostics, and Auto-Architect.

pub mod auto_architect;
pub mod diagnostics;
pub mod pareto_search;

pub use auto_architect::*;
pub use diagnostics::*;
pub use pareto_search::*;
