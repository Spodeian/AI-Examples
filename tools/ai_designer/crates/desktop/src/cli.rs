//! Headless CLI Interface for AI Designer: batch compilation, packaging, linting, importing,
//! and multi-format conversion matrix (.json, .toml, .yaml, .bson, .ron, .aid, .pkg, .aidpkg, .tar.gz).

use shared::{
    AdvisorySeverity, ArchitectureArchetype, BlueprintFileFormat, PkgWeightsFormat,
    PrecisionFormat, REPO_CATEGORIES, analyze_roofline, create_aidpkg, export_project_directory,
    generate_pareto_frontier, import_from_huggingface_config, import_from_python_source,
    import_project_directory, import_spec_from_safetensors, infer_shapes, load_blueprint_auto,
    pack_pkg, parse_safetensors_header, read_aidpkg, read_tar_gz, run_advisor_checks,
    save_blueprint_auto, scaffold_archetype, to_bson_bytes, to_json_string, to_yaml_string,
    unpack_pkg,
};
use std::fs;
use std::path::Path;

pub fn run_cli(args: &[String]) -> Result<bool, String> {
    if args.len() <= 1 {
        return Ok(false); // No subcommand passed -> launch GUI
    }

    let cmd = args[1].as_str();
    if cmd == "--gui" {
        return Ok(false);
    }

    match cmd {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(true)
        }
        "new" | "init" => {
            if args.len() < 3 {
                return Err("Usage: desktop new <model_id> [--preset <blank|transformer|vision|physics|robotics|timeseries|tabular|residual|moe>] [--name <display_name>] [--dim <d_model>] [--depth <depth>] [--precision <bf16|fp16|fp8|bitnet|fp32>] [-o <out_file>]".to_string());
            }
            let model_id = &args[2];
            let preset = get_arg_val(args, "--preset").unwrap_or_else(|| "transformer".to_string());
            let display_name = get_arg_val(args, "--name");
            let dim: usize = get_arg_val(args, "--dim")
                .and_then(|s| s.parse().ok())
                .unwrap_or(512);
            let depth: usize = get_arg_val(args, "--depth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);
            let prec_str = get_arg_val(args, "--precision").unwrap_or_else(|| "bf16".to_string());
            let out_target =
                get_arg_val(args, "-o").unwrap_or_else(|| format!("{model_id}.yaml"));
            cmd_new(
                model_id,
                &preset,
                display_name.as_deref(),
                dim,
                depth,
                &prec_str,
                &out_target,
            )?;
            Ok(true)
        }
        "template" | "templates" => {
            let sub = args.get(2).map(|s| s.as_str());
            let out_target = get_arg_val(args, "-o");
            cmd_template(sub, out_target.as_deref())?;
            Ok(true)
        }
        "catalog" => {
            print_catalog();
            Ok(true)
        }
        "hardware" | "info" => {
            print_hardware();
            Ok(true)
        }
        "verify" => {
            if args.len() < 3 {
                return Err("Usage: desktop verify <blueprint.yaml | blueprint.aid>".to_string());
            }
            let blueprint_path = &args[2];
            cmd_verify(blueprint_path)?;
            Ok(true)
        }
        "pareto" => {
            if args.len() < 3 {
                return Err("Usage: desktop pareto <blueprint.yaml | blueprint.aid>".to_string());
            }
            let blueprint_path = &args[2];
            cmd_pareto(blueprint_path)?;
            Ok(true)
        }
        "export" => {
            if args.len() < 3 {
                return Err("Usage: desktop export <blueprint.yaml | .aid | .json | .toml | .bson> [-o <out_dir_or_file>]".to_string());
            }
            let blueprint_path = &args[2];
            let out_target = get_arg_val(args, "-o").unwrap_or_else(|| "build_output".to_string());
            cmd_export(blueprint_path, &out_target)?;
            Ok(true)
        }
        "lint" => {
            if args.len() < 3 {
                return Err(
                    "Usage: desktop lint <blueprint.yaml | blueprint.aid> [--fix]".to_string(),
                );
            }
            let blueprint_path = &args[2];
            let fix = args.iter().any(|a| a == "--fix");
            cmd_lint(blueprint_path, fix)?;
            Ok(true)
        }
        "package" => {
            if args.len() < 3 {
                return Err("Usage: desktop package <blueprint.yaml | blueprint.aid> [--weights <weights.safetensors>] -o <out.aidpkg | out.pkg | out.tar.gz>".to_string());
            }
            let blueprint_path = &args[2];
            let weights_path = get_arg_val(args, "--weights");
            let out_pkg = get_arg_val(args, "-o").unwrap_or_else(|| "model.aidpkg".to_string());
            cmd_package(blueprint_path, weights_path.as_deref(), &out_pkg)?;
            Ok(true)
        }
        "unpack" => {
            if args.len() < 3 {
                return Err("Usage: desktop unpack <model.aidpkg | model.pkg | archive.tar.gz> [-o <out_dir>]".to_string());
            }
            let pkg_path = &args[2];
            let out_dir =
                get_arg_val(args, "-o").unwrap_or_else(|| "extracted_package".to_string());
            cmd_unpack(pkg_path, &out_dir)?;
            Ok(true)
        }
        "import" => {
            if args.len() < 3 {
                return Err(
                    "Usage: desktop import <src_file_or_dir> -o <out_file_or_dir>".to_string(),
                );
            }
            let src_path = &args[2];
            let out_target =
                get_arg_val(args, "-o").unwrap_or_else(|| "imported.yaml".to_string());
            cmd_import(src_path, &out_target)?;
            Ok(true)
        }
        "audit" => {
            let root_dir = if args.len() >= 3 {
                &args[2]
            } else {
                "../../pytorch2"
            };
            cmd_audit(root_dir)?;
            Ok(true)
        }
        other => Err(format!(
            "Unknown subcommand: '{}'. Run 'desktop --help' for usage.",
            other
        )),
    }
}

fn get_arg_val(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn print_help() {
    println!(
        r#"
=============================================================================
  AI DESIGNER STUDIO & TRAINING SUITE — HEADLESS CLI & UNIVERSAL CODEC
=============================================================================
Usage:
  desktop [SUBCOMMAND] [OPTIONS]
  desktop --gui                  Launch interactive egui Studio GUI (default)

Subcommands:
  new <model_id> [OPTIONS]
      Scaffolds a new AI architecture from scratch or guided archetype:
      Options:
        --preset <blank|transformer|vision|physics|robotics|timeseries|tabular|residual|moe>
        --name <display_name>
        --dim <d_model> (e.g. 512)
        --depth <depth> (e.g. 4)
        --precision <bf16|fp16|fp8|bitnet|fp32>
        -o <out_file> (e.g. model.aid, model.yaml, model.json, model.aidpkg)

  template [list | <name_or_index>] [-o <out_file>]
      Inspects or scaffolds any of the 20 pre-engineered starter blueprints.

  export <blueprint> [-o <out_dir_or_file>]
      Exports to a complete multi-backend project directory (with PyTorch, JAX,
      Candle, WGSL, Triton code) OR directly to a single-file format based on
      extension (.yaml, .json, .toml, .bson, .ron, .aid, .aidpkg, .tar.gz).

  import <src_path> -o <out_target>
      Ingests any supported format (.safetensors, config.json, .yaml, .json,
      .toml, .bson, .ron, .aid, .pkg, .aidpkg, .tar.gz, or project directory)
      and converts it seamlessly to the requested output format.

  package <blueprint> [--weights <weights>] -o <out.aidpkg | out.pkg | out.tar.gz>
      Bundles blueprint and weights into a page-aligned .aidpkg, weights-only
      .pkg container, or compressed .tar.gz archive.

  unpack <pkg_or_archive> [-o <out_dir>]
      Extracts all artifacts (blueprints in YAML/JSON/BSON, manifest, weights)
      from .aidpkg, .pkg, or .tar.gz into a target folder.

  verify <blueprint>
      Executes deterministic Seed-42 forward pass, checking activation stability,
      vanishing/exploding magnitudes, and verifying absence of NaN/Inf traps.

  pareto <blueprint>
      Synthesizes and evaluates 5 hardware-aware Pareto architecture candidates
      (Edge-Minimal, Latency-Optimal, Workstation-Balanced, High-Capacity, Frontier).

  lint <blueprint> [--fix]
      Inspects blueprint against 55+ domain rules with optional AST auto-fix.

  audit [pytorch2_dir]
      Audits repository PyTorch 2 modules and extracts architecture AST statistics.

  hardware, info
      Probes host CPU parallelism, OS, architecture, and live GPU specs via nvidia-smi.

  catalog
      Lists the 18 architectural categories and representative topics in AI-Examples.

  help, --help, -h
      Displays this manual.
"#
    );
}

fn cmd_new(
    model_id: &str,
    preset: &str,
    display_name: Option<&str>,
    dim: usize,
    depth: usize,
    prec_str: &str,
    out_target: &str,
) -> Result<(), String> {
    let archetype = match preset.to_lowercase().as_str() {
        "blank" | "scratch" | "custom" => ArchitectureArchetype::Blank,
        "transformer" | "llm" | "lm" | "gpt" => ArchitectureArchetype::GenerativeLLM,
        "vision" | "dit" | "diffusion" => ArchitectureArchetype::VisionDiffusion,
        "physics" | "pinn" | "fno" => ArchitectureArchetype::PhysicsOperator,
        "robotics" | "act" | "policy" => ArchitectureArchetype::RoboticsPolicy,
        "timeseries" | "xlstm" | "ts" => ArchitectureArchetype::TimeSeriesForecaster,
        "tabular" | "tabpfn" => ArchitectureArchetype::TabularPriorNetwork,
        "residual" | "deep" | "mlp" => ArchitectureArchetype::DeepResidual,
        "moe" | "sparse" => ArchitectureArchetype::DenseMoE,
        other => {
            return Err(format!(
                "Unknown preset '{other}'. Available: blank, transformer, vision, physics, robotics, timeseries, tabular, residual, moe"
            ));
        }
    };

    let precision = match prec_str.to_lowercase().as_str() {
        "bf16" => PrecisionFormat::BF16,
        "fp16" => PrecisionFormat::FP16,
        "fp8" | "fp8_e4m3" => PrecisionFormat::FP8E4M3FN,
        "bitnet" | "bitnet158" => PrecisionFormat::BitNet158b,
        "fp32" => PrecisionFormat::FP32,
        other => {
            return Err(format!(
                "Unknown precision '{other}'. Choose: bf16, fp16, fp8, bitnet, fp32"
            ));
        }
    };

    let name = display_name.unwrap_or_else(|| archetype.display_name());
    let spec = scaffold_archetype(archetype, model_id, name, dim, depth, precision);

    let out_path = Path::new(out_target);
    save_blueprint_auto(&spec, out_path, None)?;

    let _shapes = infer_shapes(&spec);
    let roofline = analyze_roofline(&spec, 8, 256);
    let findings = run_advisor_checks(&spec);

    println!("\n=============================================================================");
    println!("  AI ARCHITECTURE CREATED SUCCESSFULLY");
    println!("=============================================================================");
    println!("  Model ID           : {}", spec.metadata.model_id);
    println!("  Display Title      : {}", spec.metadata.display_name);
    println!(
        "  Archetype Preset   : {:?} ({})",
        archetype,
        archetype.domain()
    );
    println!("  Hidden Dimension   : {}", dim);
    println!("  Layers / Depth     : {}", spec.graph.nodes.len());
    let partition = shared::partition_parameters(&spec);
    let vram_mb = roofline.vram.total_allocated_bytes as f64 / (1024.0 * 1024.0);
    println!("  Total Parameters   : {}", partition.total_param_count);
    println!(
        "  FLOPs / Token      : {:.2} MFLOPs",
        roofline.total_flops_per_token as f64 / 1e6
    );
    println!("  Est. VRAM Footprint: {:.2} MiB", vram_mb);
    println!("  Roofline Regime    : {:?}", roofline.regime);
    println!("  Advisor Checks     : {} findings", findings.len());
    println!("  Output File Saved  : {}", out_path.display());
    println!("=============================================================================\n");

    Ok(())
}

fn cmd_template(sub: Option<&str>, out_target: Option<&str>) -> Result<(), String> {
    let mut templates_dir = Path::new("templates");
    let fallback_dir = Path::new("tools/ai_designer/templates");
    if !templates_dir.exists() && fallback_dir.exists() {
        templates_dir = fallback_dir;
    }
    if !templates_dir.exists() {
        return Err(
            "Templates directory not found. Please run from tools/ai_designer directory."
                .to_string(),
        );
    }

    let mut entries = Vec::new();
    if let Ok(rd) = fs::read_dir(templates_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("yaml") {
                entries.push(p);
            }
        }
    }
    entries.sort();

    match sub {
        None | Some("list") => {
            println!(
                "\n=== Available Starter Blueprints ({} Templates) ===",
                entries.len()
            );
            for (idx, p) in entries.iter().enumerate() {
                let filename = p.file_name().unwrap_or_default().to_string_lossy();
                println!("  [{:02}] {}", idx + 1, filename);
            }
            println!(
                "\nTo scaffold a template, run: desktop template <name_or_index> -o <target_file>\n"
            );
            Ok(())
        }
        Some(target) => {
            let found = if let Ok(idx) = target.parse::<usize>() {
                if idx > 0 && idx <= entries.len() {
                    Some(&entries[idx - 1])
                } else {
                    None
                }
            } else {
                entries.iter().find(|p| {
                    let name = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase();
                    name.contains(&target.to_lowercase())
                })
            };

            match found {
                Some(path) => {
                    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
                    if let Some(out) = out_target {
                        fs::write(out, &content).map_err(|e| e.to_string())?;
                        println!(
                            ">>> Scaffolding template '{}' to '{}' completed.",
                            path.display(),
                            out
                        );
                    } else {
                        print!("{content}");
                    }
                    Ok(())
                }
                None => Err(format!(
                    "Template '{target}' not found. Run 'desktop template list' to view available templates."
                )),
            }
        }
    }
}

fn print_hardware() {
    let report = shared::analysis::detect_host_hardware();
    println!("\n=== Host Hardware & System Introspection ===");
    println!("  Operating System   : {} ({})", report.os, report.arch);
    println!("  CPU Logical Cores  : {}", report.cpu_logical_cores);
    if report.gpu_detected {
        println!("  GPU Detected       : YES");
        println!(
            "  GPU Model          : {}",
            report.gpu_name.as_deref().unwrap_or("Unknown")
        );
        println!(
            "  GPU VRAM           : {} MiB ({:.2} GiB)",
            report.gpu_vram_mib.unwrap_or(0),
            report.gpu_vram_mib.unwrap_or(0) as f64 / 1024.0
        );
        println!(
            "  Driver Version     : {}",
            report.driver_version.as_deref().unwrap_or("Unknown")
        );
    } else {
        println!("  GPU Detected       : NO (Falling back to Host CPU SIMD)");
    }
    println!("\n--- Recommended Target Roofline Profile ---");
    println!(
        "  Profile Name       : {}",
        report.recommended_profile.profile_name
    );
    println!(
        "  Max VRAM Allocation: {:.2} GiB",
        report.recommended_profile.max_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  Peak FP16 Compute  : {:.1} TFLOPs",
        report.recommended_profile.peak_fp16_tflops
    );
    println!(
        "  Peak FP8 Compute   : {:.1} TFLOPs",
        report.recommended_profile.peak_fp8_tflops
    );
    println!(
        "  Memory Bandwidth   : {:.1} GB/s",
        report.recommended_profile.memory_bandwidth_gbs
    );
    println!(
        "  Ridge Point (Ops/B): {:.1} FLOPs/Byte\n",
        report.recommended_profile.ridge_point()
    );
}

fn print_catalog() {
    println!("\n=== AI-Examples Architecture & Category Catalog (18 Categories) ===\n");
    for cat in REPO_CATEGORIES {
        println!("[{}] {}", cat.id, cat.name);
        println!("    Description : {}", cat.description);
        println!("    Topics      : {}", cat.representative_topics.join(", "));
        if let Some(starter) = cat.recommended_starter {
            println!("    Starter     : templates/{}", starter);
        }
        println!();
    }
}

fn cmd_export(blueprint_path: &str, out_target: &str) -> Result<(), String> {
    println!(">>> Loading blueprint: {}", blueprint_path);
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let out_path = Path::new(out_target);
    let filename = out_path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let is_single_file = filename.contains('.');

    if is_single_file {
        println!(
            ">>> Exporting blueprint to format based on extension: {}",
            out_path.display()
        );
        save_blueprint_auto(&spec, out_path, None)?;
        println!(">>> Single-file export successful: {}", out_path.display());
    } else {
        println!(
            ">>> Generating complete Multi-Backend Project Directory for '{}'...",
            spec.metadata.model_id
        );
        export_project_directory(&spec, out_path, BlueprintFileFormat::Yaml, None)?;
        println!(
            "\n>>> Export successful! Turnkey multi-backend project written to: {}",
            out_path.display()
        );
    }
    Ok(())
}

fn cmd_lint(blueprint_path: &str, fix: bool) -> Result<(), String> {
    println!(">>> Linting blueprint: {}", blueprint_path);
    let mut spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let findings = run_advisor_checks(&spec);
    if findings.is_empty() {
        println!("  [OK] No mathematical or architectural issues detected. Blueprint is clean!");
        return Ok(());
    }

    println!(">>> Found {} diagnostic advisories:", findings.len());
    for (i, finding) in findings.iter().enumerate() {
        let tag = match finding.severity {
            AdvisorySeverity::CriticalMathematicalDefect => "[CRITICAL DEFECT]",
            AdvisorySeverity::Warning => "[WARNING]",
            AdvisorySeverity::BestPractice => "[BEST PRACTICE]",
            AdvisorySeverity::Info => "[INFO]",
        };
        println!("  {} {}. {}", tag, i + 1, finding.title);
        println!("       Rationale: {}", finding.mathematical_rationale);
        if let Some(ref fix_op) = finding.quick_fix {
            println!("       Quick-Fix Available: {:?}", fix_op);
        }
    }

    if fix {
        println!("\n>>> Applying automated AST quick-fixes...");
        let mut fix_count = 0;
        for finding in &findings {
            if let Some(ref mutation) = finding.quick_fix {
                mutation.apply(&mut spec);
                fix_count += 1;
            }
        }
        save_blueprint_auto(&spec, Path::new(blueprint_path), None)?;
        println!(
            "  [OK] Applied {} quick-fixes and updated {}",
            fix_count, blueprint_path
        );
    } else {
        println!("\n>>> Tip: Pass '--fix' to automatically apply recommended mutations.");
    }

    Ok(())
}

fn cmd_package(
    blueprint_path: &str,
    weights_path: Option<&str>,
    out_pkg: &str,
) -> Result<(), String> {
    println!(">>> Packaging: {}", blueprint_path);
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let weights_bytes = if let Some(wp) = weights_path {
        println!(">>> Reading weights from: {}", wp);
        Some(fs::read(wp).map_err(|e| format!("Failed to read weights file: {}", e))?)
    } else {
        None
    };

    let out = Path::new(out_pkg);
    if out_pkg.ends_with(".pkg") {
        if let Some(ref w) = weights_bytes {
            let pkg = pack_pkg(w, PkgWeightsFormat::Safetensors, spec.graph.nodes.len());
            fs::write(out, pkg).map_err(|e| format!("Failed to write .pkg: {}", e))?;
            println!(
                ">>> Successfully created page-aligned weights package: {}",
                out_pkg
            );
        } else {
            return Err("Cannot create .pkg weights container without --weights file".to_string());
        }
    } else if out_pkg.ends_with(".tar.gz") || out_pkg.ends_with(".tgz") {
        save_blueprint_auto(&spec, out, weights_bytes.as_deref())?;
        println!(
            ">>> Successfully created compressed TAR.GZ archive: {}",
            out_pkg
        );
    } else {
        let pkg_bytes = create_aidpkg(&spec, weights_bytes.as_deref(), None)
            .map_err(|e| format!("Package creation failed: {}", e))?;
        fs::write(out, &pkg_bytes)
            .map_err(|e| format!("Failed to write package '{}': {}", out_pkg, e))?;
        println!(
            ">>> Successfully created page-aligned binary package: {} ({} bytes)",
            out_pkg,
            pkg_bytes.len()
        );
    }

    Ok(())
}

fn cmd_unpack(pkg_path: &str, out_dir: &str) -> Result<(), String> {
    println!(">>> Unpacking: {}", pkg_path);
    let bytes = fs::read(pkg_path).map_err(|e| format!("Failed to read package: {}", e))?;
    let out = Path::new(out_dir);
    fs::create_dir_all(out).map_err(|e| format!("Failed to create out dir: {}", e))?;

    if pkg_path.ends_with(".pkg") {
        let (manifest, weights) = unpack_pkg(&bytes)?;
        let ext = manifest.format.extension();
        let weights_name = format!("weights.{}", ext);
        fs::write(out.join(&weights_name), &weights)
            .map_err(|e| format!("Failed to write weights: {}", e))?;
        println!(
            "  [+] Extracted weights container: {} ({} bytes)",
            weights_name,
            weights.len()
        );
    } else if pkg_path.ends_with(".tar.gz") || pkg_path.ends_with(".tgz") {
        let entries = read_tar_gz(&bytes)?;
        for entry in entries {
            let dest = out.join(&entry.path);
            if let Some(p) = dest.parent() {
                fs::create_dir_all(p).ok();
            }
            fs::write(&dest, &entry.data).map_err(|e| format!("Failed to write: {}", e))?;
            println!("  [+] Extracted from archive: {}", entry.path);
        }
    } else {
        let contents =
            read_aidpkg(&bytes).map_err(|e| format!("Failed to parse package: {}", e))?;

        fs::write(out.join("manifest.json"), &contents.manifest_json)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;
        println!("  [+] Extracted: manifest.json");

        let yaml = to_yaml_string(&contents.spec)
            .map_err(|e| format!("Failed to serialize YAML: {}", e))?;
        fs::write(out.join("blueprint.yaml"), &yaml)
            .map_err(|e| format!("Failed to write YAML blueprint: {}", e))?;
        println!("  [+] Extracted: blueprint.yaml");

        let json = to_json_string(&contents.spec)?;
        fs::write(out.join("blueprint.json"), &json)
            .map_err(|e| format!("Failed to write JSON blueprint: {}", e))?;
        println!("  [+] Extracted: blueprint.json");

        let bson = to_bson_bytes(&contents.spec)?;
        fs::write(out.join("blueprint.bson"), &bson)
            .map_err(|e| format!("Failed to write BSON blueprint: {}", e))?;
        println!("  [+] Extracted: blueprint.bson");

        if let Some(w) = contents.weights_bytes {
            fs::write(out.join("weights.safetensors"), &w)
                .map_err(|e| format!("Failed to write weights: {}", e))?;
            println!(
                "  [+] Extracted: weights.safetensors ({} bytes, 4096-byte page aligned)",
                w.len()
            );
        }
    }

    println!(">>> Unpack complete to directory: {}", out.display());
    Ok(())
}

fn cmd_import(src_path: &str, out_target: &str) -> Result<(), String> {
    println!(">>> Ingesting model from: {}", src_path);
    let src = Path::new(src_path);

    let (spec, weights) = if src.is_dir() {
        import_project_directory(src)?
    } else if src_path.ends_with(".safetensors") {
        let bytes = fs::read(src).map_err(|e| format!("Failed to read source file: {}", e))?;
        let parsed_headers = parse_safetensors_header(&bytes)?;
        let model_id = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_model");
        (
            import_spec_from_safetensors(model_id, &parsed_headers)?,
            Some(bytes),
        )
    } else if src_path.ends_with("config.json") {
        let text =
            fs::read_to_string(src).map_err(|e| format!("File is not valid UTF-8 text: {}", e))?;
        (import_from_huggingface_config(&text)?, None)
    } else if src_path.ends_with(".py") {
        println!(">>> Parsing Python AST and extracting PyTorch nn.Module architecture...");
        let text =
            fs::read_to_string(src).map_err(|e| format!("File is not valid UTF-8 text: {}", e))?;
        (import_from_python_source(&text, None)?, None)
    } else {
        (load_blueprint_auto(src)?, None)
    };

    let out = Path::new(out_target);
    let filename = out.file_name().and_then(|f| f.to_str()).unwrap_or("");
    if filename.contains('.') {
        save_blueprint_auto(&spec, out, weights.as_deref())?;
    } else {
        export_project_directory(&spec, out, BlueprintFileFormat::Yaml, weights.as_deref())?;
    }

    println!(">>> Ingestion complete! Output written to: {}", out_target);
    Ok(())
}

fn cmd_verify(blueprint_path: &str) -> Result<(), String> {
    println!(
        ">>> Verifying Transitive Parity & Numerical Stability: {}",
        blueprint_path
    );
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    let report = shared::analysis::verify_transitive_parity(&spec);

    println!("\n=== Parity & Numerical Condition Report ===");
    println!("  Model ID           : {}", report.model_id);
    println!("  Nodes Evaluated    : {}", report.nodes_evaluated);
    println!("  Max Activation     : {:.4e}", report.max_activation);
    println!("  Min Activation     : {:.4e}", report.min_activation);
    println!(
        "  NaN / Inf Traps    : {}",
        if report.has_nan_or_inf {
            "DETECTED (FAIL)"
        } else {
            "None (Clean)"
        }
    );
    println!(
        "  Condition Health   : {}",
        if report.condition_stable {
            "Stable"
        } else {
            "Ill-conditioned"
        }
    );
    println!("  Verification Status: {:?}", report.status);

    println!("\n--- Activation Health Diagnostics ---");
    for diag in &report.diagnostics {
        println!("  [*] {}", diag);
    }
    println!();

    if report.status == shared::analysis::ParityStatus::Failed {
        return Err("Numerical verification failed: model graph contains NaN/Inf or catastrophic instability.".to_string());
    }

    Ok(())
}

fn cmd_pareto(blueprint_path: &str) -> Result<(), String> {
    println!(
        ">>> Generating Multi-Objective Pareto Frontier: {}",
        blueprint_path
    );
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    let candidates = generate_pareto_frontier(&spec, None);

    println!("\n=== Multi-Objective Pareto Architecture Frontier (5 Candidates) ===");
    println!(
        "{:<24} | {:<12} | {:<12} | {:<12} | {:<12} | {:<8}",
        "Candidate", "Parameters", "FLOPs/Token", "Latency (ms)", "VRAM (MiB)", "OOM Risk"
    );
    println!(
        "{:-<24}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+---------",
        "", "", "", "", ""
    );

    for c in &candidates {
        let vram_mib = c.vram_bytes as f64 / (1024.0 * 1024.0);
        let oom_str = if c.is_oom_risk { "YES (FAIL)" } else { "Safe" };
        println!(
            "{:<24} | {:<12} | {:<12} | {:<12.3} | {:<12.1} | {:<8}",
            c.label, c.total_param_count, c.flops_per_token, c.latency_ms, vram_mib, oom_str
        );
    }
    println!("\nCandidate Profiles:");
    for c in &candidates {
        println!("  [*] {}: {}", c.label, c.description);
    }
    println!();
    Ok(())
}

fn cmd_audit(search_dir: &str) -> Result<(), String> {
    println!(">>> Sweeping and auditing models in: {}", search_dir);
    let root = Path::new(search_dir);
    if !root.exists() {
        return Err(format!("Directory '{}' does not exist.", search_dir));
    }

    let mut py_files = Vec::new();
    find_files_recursive(root, &mut py_files);

    if py_files.is_empty() {
        println!(">>> No models found in '{}'.", search_dir);
        return Ok(());
    }

    println!(
        ">>> Discovered {} model candidate files. Running 55+ advisory rules & numerical parity audits...",
        py_files.len()
    );

    let mut audited_count = 0;
    let mut clean_count = 0;
    let mut total_advisories = 0;
    let mut critical_defects = 0;
    let mut parity_passed = 0;

    for path in &py_files {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let spec = if path.extension().and_then(|e| e.to_str()) == Some("py") {
            match import_from_python_source(&text, None) {
                Ok(s) => s,
                Err(_) => continue,
            }
        } else {
            match load_blueprint_auto(path) {
                Ok(s) => s,
                Err(_) => continue,
            }
        };

        audited_count += 1;
        let findings = run_advisor_checks(&spec);
        let parity = shared::analysis::verify_transitive_parity(&spec);

        if parity.status == shared::analysis::ParityStatus::Passed {
            parity_passed += 1;
        }

        if findings.is_empty() {
            clean_count += 1;
        } else {
            total_advisories += findings.len();
            for f in &findings {
                if matches!(f.severity, AdvisorySeverity::CriticalMathematicalDefect) {
                    critical_defects += 1;
                }
            }
        }
    }

    println!("\n=============================================================================");
    println!("  AI-EXAMPLES ARCHITECTURE COMPLIANCE & REPOSITORY AUDIT REPORT");
    println!("=============================================================================");
    println!("  Search Directory       : {}", search_dir);
    println!("  Models Audited         : {}", audited_count);
    println!(
        "  Parity Pass Rate       : {} / {} ({:.1}%)",
        parity_passed,
        audited_count,
        (parity_passed as f64 / audited_count.max(1) as f64) * 100.0
    );
    println!(
        "  Clean (0 Advisories)   : {} / {} ({:.1}%)",
        clean_count,
        audited_count,
        (clean_count as f64 / audited_count.max(1) as f64) * 100.0
    );
    println!("  Total Advisories Raised: {}", total_advisories);
    println!("  Critical Defect Alerts : {}", critical_defects);
    println!("=============================================================================\n");

    Ok(())
}

fn find_files_recursive(dir: &Path, list: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_files_recursive(&path, list);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && (ext == "py" || ext == "yaml")
            {
                list.push(path);
            }
        }
    }
}
