//! Category 03 Building Blocks & Losses Integration Test Suite for ai-candle.

#[path = "03_building_blocks_and_losses/test_01_activation_functions_zoo.rs"]
mod test_01_activation_functions_zoo;

#[path = "03_building_blocks_and_losses/test_02_normalization_layers_complete.rs"]
mod test_02_normalization_layers_complete;

#[path = "03_building_blocks_and_losses/test_03_weight_initialization_and_mup_scaling.rs"]
mod test_03_weight_initialization_and_mup_scaling;

#[path = "03_building_blocks_and_losses/test_04_positional_and_structural_encodings.rs"]
mod test_04_positional_and_structural_encodings;

#[path = "03_building_blocks_and_losses/test_05_pooling_and_downsampling_operators.rs"]
mod test_05_pooling_and_downsampling_operators;

#[path = "03_building_blocks_and_losses/test_06_regularization_and_training_stabilizers.rs"]
mod test_06_regularization_and_training_stabilizers;
