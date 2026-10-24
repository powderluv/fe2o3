#![forbid(unsafe_code)]

mod host_cpu;
mod protocol;
mod reference_attention;
mod reference_basic;
mod reference_gpt;
mod reference_low_precision;
mod reference_systems;
mod references;
mod systems_cpu;

use protocol::{Request, Response};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

fn execute(input: Vec<u8>) -> Result<Vec<u8>, String> {
    if input.is_empty() || input.len() > protocol::MAX_REQUEST {
        return Err("request byte bound".to_owned());
    }
    let request: Request = serde_json::from_slice(&input).map_err(|e| e.to_string())?;
    request.validate()?;
    request.corpus_shape(&references::generate(&request.kernel)?)?;
    let outputs = references::evaluate(&request)?;
    let response = Response::new(&request, Sha256::digest(&input).into(), outputs)?;
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let bytes = match args.as_slice() {
        [mode] if mode == "list" => serde_json::to_vec(&serde_json::json!({
            "schema": "fe2o3-tutorial-reference-corpus-v92", "kernels": references::kernels(),
            "authority": false,
        }))
        .map_err(|e| e.to_string())?,
        [mode, kernel] if mode == "generate" => {
            let request = references::generate(kernel)?;
            request.validate()?;
            serde_json::to_vec(&request).map_err(|e| e.to_string())?
        }
        [mode] if mode == "execute" => {
            let mut input = Vec::new();
            std::io::stdin()
                .take(protocol::MAX_REQUEST as u64 + 1)
                .read_to_end(&mut input)
                .map_err(|e| e.to_string())?;
            execute(input)?
        }
        [mode, path] if mode == "execute" => {
            use rustix::fs::{Mode, OFlags};
            use std::os::unix::fs::MetadataExt;
            let file = std::fs::File::from(
                rustix::fs::openat(
                    rustix::fs::CWD,
                    path,
                    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(|e| e.to_string())?,
            );
            let before = file.metadata().map_err(|e| e.to_string())?;
            if !before.is_file() || before.len() > protocol::MAX_REQUEST as u64 {
                return Err("bounded regular request required".to_owned());
            }
            let mut input = Vec::new();
            (&file)
                .take(before.len() + 1)
                .read_to_end(&mut input)
                .map_err(|e| e.to_string())?;
            let identity = |m: std::fs::Metadata| {
                (
                    m.dev(),
                    m.ino(),
                    m.mode(),
                    m.nlink(),
                    m.uid(),
                    m.gid(),
                    m.len(),
                    m.mtime(),
                    m.mtime_nsec(),
                    m.ctime(),
                    m.ctime_nsec(),
                )
            };
            if input.len() as u64 != before.len()
                || identity(before) != identity(file.metadata().map_err(|e| e.to_string())?)
                || identity(file.metadata().map_err(|e| e.to_string())?)
                    != identity(std::fs::symlink_metadata(path).map_err(|e| e.to_string())?)
            {
                return Err("request file changed".to_owned());
            }
            execute(input)?
        }
        _ => return Err("expected list, generate KERNEL or execute [REQUEST]".to_owned()),
    };
    if bytes.len() > protocol::MAX_RESPONSE {
        return Err("response byte bound".to_owned());
    }
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&bytes)
        .and_then(|()| stdout.write_all(b"\n"))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_current_corpus_request_executes_independent_reference_and_covers_all_outputs() {
        let names = references::kernels();
        assert_eq!(names.len(), 46);
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
        for name in names {
            let request = references::generate(name).unwrap();
            request.validate().unwrap();
            let bytes = serde_json::to_vec(&request).unwrap();
            let response = execute(bytes.clone()).unwrap_or_else(|error| panic!("{name}: {error}"));
            let json: serde_json::Value = serde_json::from_slice(&response).unwrap();
            assert_eq!(json["kernel"], name);
            assert_eq!(json["authority"], false);
            assert_eq!(
                json["request_sha256"],
                serde_json::json!(<[u8; 32]>::from(Sha256::digest(&bytes)))
            );
            assert!(!json["outputs"].as_array().unwrap().is_empty());
        }
        for name in [
            "aggregate_enum",
            "aggregate_pointer",
            "aggregate_drop",
            "unknown",
        ] {
            assert!(references::generate(name).is_err());
        }
    }

    fn bf16_case(kernel: &str, reduction: usize, full: bool) -> (Request, Vec<u32>) {
        let dimension = if full { 16 } else { 1 };
        let mut a = vec![0_u16; dimension * reduction];
        let mut b = vec![0_u16; reduction * dimension];
        let values = if full {
            for row in 0..16 {
                for depth in 0..16 {
                    a[row * 16 + depth] = 0x3f80 + (row + depth) as u16;
                }
            }
            for depth in 0..16 {
                for col in 0..16 {
                    let num = 128 + 2 * col - depth;
                    b[depth * 16 + col] = if num >= 128 {
                        0x3f80 + (num - 128) as u16
                    } else {
                        0x3f00 + 2 * (num - 64) as u16
                    };
                }
            }
            (0..16)
                .flat_map(|r| {
                    (0..16).map(move |c| {
                        ((16 * (128 + r) * (128 + 2 * c) + 120 * (2 * c - r) - 1240) as f32
                            / 16384.0)
                            .to_bits()
                    })
                })
                .collect::<Vec<_>>()
        } else if reduction == 1 {
            a[0] = 0x3f81;
            b[0] = 0x3f7f;
            vec![0x3f80_7f00]
        } else {
            a.copy_from_slice(&[0x4281, 0x4280]);
            b.copy_from_slice(&[0x4281, 0xc282]);
            vec![0x3e80_0000]
        };
        let guards = if full { 8 } else { 1 };
        let mut initial = vec![0_u32; dimension * dimension];
        initial.extend(vec![0x7f80_0001; guards]);
        let mut expected = values;
        expected.extend(vec![0x7f80_0001; guards]);
        let mut args = vec![
            protocol::buffer(
                "u16",
                "read_only",
                &a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            ),
            protocol::buffer(
                "u16",
                "read_only",
                &b.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            ),
            protocol::buffer(
                "f32",
                "read_write",
                &initial
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ];
        args.extend(
            [
                dimension, dimension, reduction, reduction, dimension, dimension,
            ]
            .map(|v| protocol::scalar(v as u32)),
        );
        args.extend([protocol::float_scalar(1.0), protocol::float_scalar(0.0)]);
        let groups = if kernel == "simt_gemm_general_v1" && full {
            4
        } else {
            1
        };
        (Request::new(kernel, 64, groups, args), expected)
    }

    #[test]
    fn closed_bf16_cases_consume_exact_operands_outputs_and_canaries() {
        for kernel in ["tiled_gemm_general_v1", "simt_gemm_general_v1"] {
            for (k, full) in [(1, false), (2, false), (16, true)] {
                let (request, expected) = bf16_case(kernel, k, full);
                let bytes = serde_json::to_vec(&request).unwrap();
                let result: serde_json::Value =
                    serde_json::from_slice(&execute(bytes.clone()).unwrap()).unwrap();
                assert_eq!(result["kernel"], kernel);
                assert_eq!(
                    result["request_sha256"],
                    serde_json::json!(<[u8; 32]>::from(Sha256::digest(&bytes)))
                );
                assert_eq!(
                    result["outputs"][0]["bytes"],
                    protocol::encode(
                        &expected
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>()
                    )
                );
            }
        }
    }

    #[test]
    fn closed_bf16_enrollment_refuses_name_shape_stride_abi_and_canary_mutations() {
        let (request, _) = bf16_case("simt_gemm_general_v1", 16, true);
        let original = serde_json::to_value(&request).unwrap();
        for (index, value) in [
            (
                0,
                serde_json::json!({"kind":"scalar","type":"u16","bits":"0x0000"}),
            ),
            (
                3,
                serde_json::json!({"kind":"scalar","type":"u32","bits":"0x00000011"}),
            ),
            (
                6,
                serde_json::json!({"kind":"scalar","type":"u32","bits":"0x0000000f"}),
            ),
        ] {
            let mut changed = original.clone();
            changed["arguments"][index] = value;
            assert!(execute(serde_json::to_vec(&changed).unwrap()).is_err());
        }
        for (key, value) in [
            ("alignment", serde_json::json!(2)),
            ("access", serde_json::json!("read_only")),
            ("element", serde_json::json!("u32")),
            ("bytes", serde_json::json!("0x00000000")),
        ] {
            let mut changed = original.clone();
            changed["arguments"][2][key] = value;
            assert!(execute(serde_json::to_vec(&changed).unwrap()).is_err());
        }
        for (key, value) in [
            ("kernel", serde_json::json!("simt_gemm_general_unknown")),
            ("grid", serde_json::json!([64, 1, 1])),
            ("workgroup", serde_json::json!([32, 1, 1])),
        ] {
            let mut changed = original.clone();
            changed[key] = value;
            assert!(execute(serde_json::to_vec(&changed).unwrap()).is_err());
        }
        let mut changed = original;
        let bytes = changed["arguments"][2]["bytes"]
            .as_str()
            .unwrap()
            .to_owned();
        changed["arguments"][2]["bytes"] = format!("{}00000000", &bytes[..bytes.len() - 8]).into();
        assert!(execute(serde_json::to_vec(&changed).unwrap()).is_err());
    }

    #[test]
    fn actual_request_values_are_consumed_and_malformed_shapes_refused() {
        let request = references::generate("vecadd").unwrap();
        let original = serde_json::to_vec(&request).unwrap();
        let baseline = execute(original.clone()).unwrap();
        let mut changed: serde_json::Value = serde_json::from_slice(&original).unwrap();
        let input = changed["arguments"][0]["bytes"]
            .as_str()
            .unwrap()
            .to_owned();
        changed["arguments"][0]["bytes"] = format!("0x0000803f{}", &input[10..]).into();
        assert_ne!(
            execute(serde_json::to_vec(&changed).unwrap()).unwrap(),
            baseline
        );
        for field in ["kernel", "grid", "workgroup", "arguments"] {
            let mut invalid: serde_json::Value = serde_json::from_slice(&original).unwrap();
            invalid[field] = serde_json::Value::Null;
            assert!(execute(serde_json::to_vec(&invalid).unwrap()).is_err());
        }
        changed["unexpected"] = true.into();
        assert!(execute(serde_json::to_vec(&changed).unwrap()).is_err());
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "reference-v92 refused: {}",
                error.chars().take(1024).collect::<String>()
            );
            std::process::ExitCode::FAILURE
        }
    }
}
