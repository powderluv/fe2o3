#![forbid(unsafe_code)]

use std::{fs, path::Path, process::Command};

fn checked(command: &mut Command) -> Vec<u8> {
    let output = command.output().expect("start existing tutorial tool");
    assert!(
        output.status.success(),
        "{command:?}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    output.stdout
}

#[test]
#[ignore = "exports both real fill selections and executes the existing pinned offline simulator"]
fn source_simulation_covers_empty_tails_and_canaries() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let scratch =
        std::env::temp_dir().join(format!("fe2o3-fill-simulation-{}", std::process::id()));
    fs::create_dir(&scratch).expect("fresh owned scratch");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| env!("CARGO").into());
    for tiled in [false, true] {
        let selection = if tiled { "tile" } else { "simt" };
        let bundle = scratch.join(format!("{selection}.fe2sim"));
        let mut export = Command::new("bash");
        export
            .current_dir(root)
            .args([
                "scripts/quickstart.sh",
                "simulate-source",
                "--crate",
                "fe2o3_fill",
                "--request",
                "scripts/quickstart/fill-canary-request.json",
                "--expectation",
                "scripts/quickstart/fill-canary-expectation.json",
                "--target",
                "gfx942",
                "--output",
            ])
            .arg(&bundle)
            .args(["--", "--package", "fe2o3-fill", "--lib"]);
        if tiled {
            export.args(["--features", "structured-tile"]);
        }
        checked(&mut export);
        for length in [0_usize, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129] {
            let canary = "000200c6";
            let initial = (0..length)
                .map(|i| ["00000080", "2301c07f", "0000807f"][i % 3])
                .collect::<String>();
            let input = format!("{canary}{initial}{canary}");
            let expected = format!("{canary}{}{canary}", "00002a42".repeat(length));
            let bytes = (length + 2) * 4;
            let initialized = format!(
                "{}{}",
                "ff".repeat(bytes / 8),
                if bytes % 8 == 0 { "" } else { "0f" }
            );
            let view = format!(
                r#"{{"kind":"buffer_view","backing":7,"element":"f32","access":"read_write","alignment":4,"byte_offset":4,"elements":{length}}}"#
            );
            let grid = length.max(1).div_ceil(64) * 64;
            let request = scratch.join(format!("{selection}-{length}.request.json"));
            let expectation = scratch.join(format!("{selection}-{length}.expectation.json"));
            let result = scratch.join(format!("{selection}-{length}.result.json"));
            fs::write(&request, format!(
                r#"{{"schema":"fe2o3-simulation-request-v1","kernel":"fill","grid":[{grid},1,1],"workgroup":[64,1,1],"arguments":[{view}],"shared_buffers":[{{"id":7,"element":"f32","access":"read_write","alignment":4,"bytes":"0x{input}"}}]}}"#
            )).unwrap();
            fs::write(&expectation, format!(
                r#"{{"schema":"fe2o3-simulation-expectation-v1","arguments":[{view}],"shared_buffers":[{{"id":7,"buffer":{{"element":"f32","access":"read_write","alignment":4,"bytes":"0x{expected}","initialized":"0x{initialized}"}}}}]}}"#
            )).unwrap();
            let actual = checked(
                Command::new(&cargo)
                    .current_dir(root)
                    .env("FE2O3_HIP_SYS_DISABLE", "1")
                    .env("FE2O3_HSA_RUNTIME_DISABLE", "1")
                    .args([
                        "run",
                        "--locked",
                        "--offline",
                        "--quiet",
                        "-p",
                        "fe2o3-kir-sim-cli",
                        "--bin",
                        "fe2o3-kir-sim",
                        "--",
                        "--bundle",
                    ])
                    .arg(&bundle)
                    .arg("--request")
                    .arg(&request),
            );
            fs::write(&result, actual).unwrap();
            eprintln!("{selection}: {length} logical elements, two canaries, grid={grid}");
            checked(
                Command::new("python3")
                    .current_dir(root)
                    .args([
                        "-I",
                        "-B",
                        "scripts/check-simulation-expectation.py",
                        "--expectation",
                    ])
                    .arg(&expectation)
                    .arg("--result")
                    .arg(&result),
            );
        }
    }
    fs::remove_dir_all(scratch).unwrap();
}
