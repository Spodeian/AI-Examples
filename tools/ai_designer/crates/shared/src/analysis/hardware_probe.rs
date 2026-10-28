//! Host Hardware Detection & Introspection Engine.
//! Probes CPU parallelism, host OS, architecture, and NVIDIA GPU properties via nvidia-smi.

use crate::spec::HardwareTargetProfile;
use serde::{Deserialize, Serialize};
use std::process::Command;

/// Comprehensive host hardware inspection report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostHardwareReport {
    pub os: String,
    pub arch: String,
    pub cpu_logical_cores: usize,
    pub gpu_detected: bool,
    pub gpu_name: Option<String>,
    pub gpu_vram_mib: Option<u64>,
    pub driver_version: Option<String>,
    pub recommended_profile: HardwareTargetProfile,
}

/// Detects live host hardware configuration and computes a recommended roofline profile.
pub fn detect_host_hardware() -> HostHardwareReport {
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    // Query nvidia-smi if available
    let (gpu_name, gpu_vram_mib, driver_version) = query_nvidia_smi();
    let gpu_detected = gpu_name.is_some();

    let recommended_profile = if let Some(ref name) = gpu_name {
        let vram_bytes = gpu_vram_mib.unwrap_or(8192) * 1024 * 1024;
        let (peak_fp16, peak_fp8, mem_bw) = estimate_gpu_performance(name);

        HardwareTargetProfile {
            profile_name: format!("Host_{}", name.replace(' ', "_")),
            max_vram_bytes: vram_bytes,
            peak_fp16_tflops: peak_fp16,
            peak_fp8_tflops: peak_fp8,
            memory_bandwidth_gbs: mem_bw,
        }
    } else {
        // Fallback to Host CPU SIMD
        HardwareTargetProfile {
            profile_name: format!("Host_CPU_{}threads", cpu_cores),
            max_vram_bytes: 16 * 1024 * 1024 * 1024, // 16GB default host RAM
            peak_fp16_tflops: (cpu_cores as f64) * 0.1,
            peak_fp8_tflops: (cpu_cores as f64) * 0.2,
            memory_bandwidth_gbs: 64.0, // typical DDR4/DDR5 dual-channel
        }
    };

    HostHardwareReport {
        os,
        arch,
        cpu_logical_cores: cpu_cores,
        gpu_detected,
        gpu_name,
        gpu_vram_mib,
        driver_version,
        recommended_profile,
    }
}

fn query_nvidia_smi() -> (Option<String>, Option<u64>, Option<String>) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
        .output();

    if let Ok(out) = output
        && out.status.success()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        if let Some(first_line) = text.lines().next() {
            let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let name = parts[0].to_string();
                let vram_str = parts[1].trim_end_matches("MiB").trim();
                let vram_mib = vram_str.parse::<u64>().ok();
                let driver = parts[2].to_string();
                return (Some(name), vram_mib, Some(driver));
            }
        }
    }
    (None, None, None)
}

fn estimate_gpu_performance(name: &str) -> (f64, f64, f64) {
    let upper = name.to_uppercase();
    if upper.contains("H100") {
        (989.0, 1978.0, 3350.0)
    } else if upper.contains("A100") {
        (312.0, 624.0, 2039.0)
    } else if upper.contains("4090") {
        (330.0, 660.0, 1008.0)
    } else if upper.contains("4080") {
        (242.0, 484.0, 716.8)
    } else if upper.contains("3090") {
        (142.0, 284.0, 936.2)
    } else if upper.contains("3080") {
        (119.0, 238.0, 760.0)
    } else if upper.contains("3060") {
        (65.0, 130.0, 448.0)
    } else if upper.contains("4060") {
        (75.0, 150.0, 272.0)
    } else if upper.contains("4070") {
        (184.0, 368.0, 504.0)
    } else {
        (50.0, 100.0, 300.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_hardware_detection() {
        let report = detect_host_hardware();
        assert!(report.cpu_logical_cores > 0);
        assert!(!report.os.is_empty());
        assert!(!report.arch.is_empty());
        assert!(report.recommended_profile.max_vram_bytes > 0);
        assert!(report.recommended_profile.peak_fp16_tflops > 0.0);
    }
}
