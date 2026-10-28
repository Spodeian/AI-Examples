//! Modular UI components, canvas viewports, inspector sidebars, and execution docks.

pub mod about;
pub mod advisor_dock;
pub mod canvas_node;
pub mod canvas_tree;
pub mod canvas_yaml;
pub mod code_viewer;
pub mod design_wizard;
pub mod inspector;
pub mod item_list;
pub mod modals;
pub mod navbar;
pub mod palette;
pub mod roofline_dock;
pub mod studio_runner;
pub mod telemetry_viewer;

pub use about::*;
pub use advisor_dock::*;
pub use canvas_node::*;
pub use canvas_tree::*;
pub use canvas_yaml::*;
pub use code_viewer::*;
pub use design_wizard::*;
pub use inspector::*;
pub use item_list::*;
pub use modals::*;
pub use navbar::*;
pub use palette::*;
pub use roofline_dock::*;
pub use studio_runner::*;
pub use telemetry_viewer::*;
