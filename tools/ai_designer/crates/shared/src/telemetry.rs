//! Standard telemetry dashboard payload and metrics for template applications.

use serde::{Deserialize, Serialize};

/// System telemetry dashboard data model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryDashboardData {
    pub title: String,
    pub timestamp_utc: i64,
    pub uptime_secs: u64,
    pub total_operations: usize,
    pub success_rate: f64,
    pub throughput_ops_sec: f64,
    pub memory_mb: f64,
    pub active_backend: String,
    pub channels: Vec<TelemetryChannelMetric>,
}

/// Channel-level metric entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryChannelMetric {
    pub name: String,
    pub count: usize,
    pub latency_ms: f64,
    pub status: String,
}

impl Default for TelemetryDashboardData {
    fn default() -> Self {
        Self {
            title: "AI Designer Studio Subsystem Telemetry".to_string(),
            timestamp_utc: 1774900000,
            uptime_secs: 1420,
            total_operations: 8450,
            success_rate: 99.98,
            throughput_ops_sec: 1250.0,
            memory_mb: 42.5,
            active_backend: "IndexedDB / Persistent Storage".to_string(),
            channels: vec![
                TelemetryChannelMetric {
                    name: "Symbolic Shape & Advisor Engine".to_string(),
                    count: 320,
                    latency_ms: 0.42,
                    status: "Healthy".to_string(),
                },
                TelemetryChannelMetric {
                    name: "Roofline, Pareto & Codegen Pipeline".to_string(),
                    count: 45,
                    latency_ms: 1.15,
                    status: "Optimal".to_string(),
                },
                TelemetryChannelMetric {
                    name: "egui Canvas & Multi-Tier Storage".to_string(),
                    count: 8085,
                    latency_ms: 0.12,
                    status: "Healthy".to_string(),
                },
            ],
        }
    }
}
