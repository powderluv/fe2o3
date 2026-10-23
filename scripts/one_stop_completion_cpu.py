#!/usr/bin/env python3
"""CPU-only exact-source ObserveCompletion regression; not native qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
KFD = "crates/fe2o3-kfd/src/engineering_gfx950_debug_one_stop_"
SOURCE_PATHS = {
    "resources": KFD + "resources_v1.rs",
    "native": KFD + "native_v1.rs",
    "owner": KFD + "owner_v1.rs",
    "public_tests": KFD + "owner_v1_tests.rs",
    "aql": "crates/fe2o3-aql/src/lib.rs",
    "toolchain": "rust-toolchain.toml",
    "fixture": "scripts/fixtures/one-stop-completion-cpu.rs",
    "runner": "scripts/one_stop_completion_cpu.py",
}
PUBLIC_TESTS = (
    "completion_poll_retained_signal_frontier_lag_is_pending_not_retired",
    "completion_poll_both_observation_orders_need_joint_terminal_tuple",
    "completion_poll_changes_only_one_tuple_and_never_adds_terminal_acceptance",
)
LOOP_TESTS = (
    "extracted_loop_lag_then_frontier_one_completes_once",
    "extracted_loop_forever_lag_stops_at_original_deadline",
    "extracted_loop_positive_at_original_deadline_still_refuses",
    "extracted_loop_expired_before_read_has_no_backend_calls",
    "extracted_loop_invalid_observations_keep_strict_error",
    "extracted_loop_does_not_reenter_completed_phase",
    "extracted_loop_owner_frontier_refuses_before_observation",
    "extracted_loop_currentness_refusal_is_not_polled",
)
EXPECTED_TESTS = frozenset(PUBLIC_TESTS + LOOP_TESTS)
SOURCE_CAP = 1024 * 1024
PIPE_CAP = 1024 * 1024
SCRATCH_CAP = 128 * 1024 * 1024
FILE_CAP = 32 * 1024 * 1024
RAW_STRING = re.compile(r'(?:br|r)(#*)"')
CHAR_LITERAL = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|[^\n])|[^'\\\n])'")


def require(value, message):
    if not value:
        raise ValueError(message)


def rust_mask(text):
    """Mask literals/comments, preserving character offsets; fail unclosed tokens."""
    result = list(text)
    i = 0
    while i < len(text):
        start = i
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            depth = 1
            i += 2
            while depth and i < len(text):
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            require(depth == 0, "unclosed Rust block comment")
        else:
            raw = RAW_STRING.match(text, i) if i == 0 or not (text[i-1].isalnum() or text[i-1] == "_") else None
            char = CHAR_LITERAL.match(text, i)
            if raw:
                terminal = '"' + raw.group(1)
                end = text.find(terminal, i + len(raw.group(0)))
                require(end >= 0, "unclosed Rust raw string")
                i = end + len(terminal)
            elif char:
                i += len(char.group(0))
            elif text[i] == '"':
                i += 1
                while i < len(text) and text[i] != '"':
                    i += 2 if text[i] == "\\" else 1
                require(i < len(text), "unclosed Rust string")
                i += 1
            else:
                i += 1
                continue
        for j in range(start, i):
            if result[j] not in "\r\n":
                result[j] = " "
    return "".join(result)


def unique_match(pattern, masked, label):
    matches = list(re.finditer(pattern, masked, re.MULTILINE))
    require(len(matches) == 1, "missing or duplicated " + label)
    return matches[0]


def brace_end(masked, opening):
    require(opening >= 0 and masked[opening] == "{", "missing opening brace")
    depth = 0
    for i in range(opening, len(masked)):
        if masked[i] == "{":
            depth += 1
        elif masked[i] == "}":
            depth -= 1
            if not depth:
                return i + 1
    raise ValueError("unclosed Rust body")


def function(text, name, test=False):
    masked = rust_mask(text)
    prefix = r"^[ \t]*(?:pub(?:\([^\n)]*\))?\s+)?fn\s+"
    if test:
        prefix = r"^#\[test\][ \t]*\r?\n[ \t]*fn\s+"
    m = unique_match(prefix + re.escape(name) + r"\s*\(", masked, name)
    opening = masked.find("{", m.end())
    require(opening >= 0 and ";" not in masked[m.end():opening], "function body not found: " + name)
    return text[m.start():brace_end(masked, opening)]


def arm(text, name):
    masked = rust_mask(text)
    m = unique_match(r"\bS\s*::\s*" + re.escape(name) + r"\s*=>\s*\{", masked, name)
    opening = masked.index("{", m.start())
    return text[opening:brace_end(masked, opening)]


def constant(text, name):
    masked = rust_mask(text)
    m = unique_match(r"^pub\s+const\s+" + re.escape(name) + r"\s*:[^;]+;", masked, name)
    return text[m.start():m.end()]


def extract(sources):
    loop = arm(sources["native"], "ObserveCompletion")
    destroy = arm(sources["native"], "DestroyQueue")
    require(re.search(r"resources\s*::\s*completion_poll\s*\(", rust_mask(loop)), "poll helper is not used by ObserveCompletion")
    require(re.search(r"resources\s*::\s*completion\s*\(", rust_mask(destroy)), "strict retirement call missing")
    require(not re.search(r"resources\s*::\s*completion_poll\s*\(", rust_mask(destroy)), "retirement must remain strict")
    deadline = re.findall(r"self\s*\.\s*inner\s*\.\s*deadline\s*=\s*Instant\s*::\s*now\s*\(\s*\)\s*\.\s*checked_add\s*\(\s*Duration\s*::\s*from_secs\s*\(\s*60\s*\)\s*\)", rust_mask(sources["owner"]))
    require(len(deadline) == 1, "original owner 60-second deadline construction missing or duplicated")
    return {
        "completion.rs": function(sources["resources"], "completion") + "\n",
        "completion_poll.rs": function(sources["resources"], "completion_poll") + "\n",
        "deadline.rs": function(sources["native"], "check_deadline") + "\n" + function(sources["native"], "before_deadline") + "\n",
        "observe-expression.rs": loop + "\n",
        "constants.rs": "\n".join(constant(sources["aql"], n) for n in ("AMD_SIGNAL_KIND_USER_V1", "AMD_SIGNAL_VALUE_PENDING_V1")) + "\n",
        "public-tests.rs": "\n\n".join(function(sources["public_tests"], n, test=True) for n in PUBLIC_TESTS) + "\n",
    }


def read_snapshot(path, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "input member cap/type")
        chunks = []
        size = 0
        while True:
            chunk = os.read(fd, min(65536, cap + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            require(size <= cap, "input read cap")
        after, named = os.fstat(fd), os.lstat(path)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(all(getattr(before, k) == getattr(after, k) == getattr(named, k) for k in fields), "input changed while reading")
        require(size == before.st_size, "input length changed")
        data = b"".join(chunks)
        return data, {"bytes": size, "sha256": hashlib.sha256(data).hexdigest()}
    finally:
        os.close(fd)


def snapshot(root=ROOT):
    text, pins = {}, {}
    total = 0
    for key, relative in SOURCE_PATHS.items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), "source path leaves repository")
        data, pins[relative] = read_snapshot(path, SOURCE_CAP)
        total += len(data)
        require(total <= SOURCE_CAP, "total source cap")
        text[key] = data.decode("utf-8", errors="strict")
    return text, pins


def channel_from(text):
    channels = re.findall(r'^channel\s*=\s*"(nightly-[0-9]{4}-[0-9]{2}-[0-9]{2})"\s*$', text, re.MULTILINE)
    require(len(channels) == 1, "one repository-pinned nightly channel required")
    return channels[0]


def verify_test_list(text):
    names = re.findall(r"^([a-zA-Z0-9_]+): test$", text, re.MULTILINE)
    require(len(names) == 11 and frozenset(names) == EXPECTED_TESTS, "exact eleven named tests required; no silent fixture omission")


def scratch_size(directory, *, live=False):
    entries, size = 0, 0
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            entries += 1
            require(entries <= 256, "scratch entry cap")
            try:
                info = os.lstat(Path(parent) / name)
            except FileNotFoundError:
                if not live:
                    raise
                # Compiler temporaries may disappear during the live census.
                # The enumerated entry still counts against the entry cap.
                continue
            require(not stat.S_ISLNK(info.st_mode), "scratch symlink refused")
            require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "scratch special file refused")
            size += info.st_size if stat.S_ISREG(info.st_mode) else 0
    require(size <= SCRATCH_CAP, "scratch logical byte cap")
    return {"entries": entries, "logical_bytes": size}


def child_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_CAP, FILE_CAP))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def run_child(command, label, output, env, timeout):
    """Own the group from spawn through setup, bounded I/O and final cleanup."""
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    proc, selector, outcome, primary, primary_traceback = None, None, None, None, None
    prior_handlers, cancelled, cleanup_errors = {}, [], []
    reaped = False
    end = time.monotonic() + timeout

    def interrupted(signum, _frame):
        # Do not throw asynchronously through Popen or cleanup. The bounded
        # loop checks this flag and raises synchronously into its finally.
        if not cancelled:
            cancelled.append(signum)

    def check_cancelled():
        if cancelled:
            raise RuntimeError(label + " interrupted by signal " + str(cancelled[0]))

    def cleanup_step(operation, action):
        try:
            action()
        except BaseException as error:
            cleanup_errors.append({"operation": operation, "error_type": type(error).__name__[:64]})

    try:
        # Installed before spawn: a cancellation cannot kill only the Python
        # parent while its separately sessioned child survives unnoticed.
        for signum in (signal.SIGTERM, signal.SIGINT):
            prior_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        check_cancelled()
        proc = subprocess.Popen(command, cwd=output, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True, preexec_fn=child_limits)
        check_cancelled()
        selector = selectors.DefaultSelector()
        for name in chunks:
            pipe = getattr(proc, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        exited = None
        while selector.get_map() or exited is None:
            check_cancelled()
            require(time.monotonic() < end, label + " deadline")
            scratch_size(output, live=True)
            for key, _ in selector.select(min(0.05, max(0, end - time.monotonic()))):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                require(len(chunks[key.data]) + len(data) <= PIPE_CAP, label + " output cap")
                chunks[key.data].extend(data)
            # Do not reap here: keeping the child waitable retains ownership of
            # its PID until the group cleanup below, avoiding a PID-reuse kill.
            exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        check_cancelled()
        require(exited.si_code == os.CLD_EXITED and exited.si_status == 0, label + " failed")
        outcome = {k: bytes(v).decode("utf-8", errors="strict") for k, v in chunks.items()}
    except BaseException as error:
        primary, primary_traceback = error, error.__traceback__
    finally:
        try:
            if proc is not None:
                def kill_group():
                    # Leader remains waitable until cleanup; no PID-reuse kill.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                cleanup_step("kill-owned-group", kill_group)

                def reap():
                    nonlocal reaped
                    proc.wait(timeout=5)
                    reaped = True
                cleanup_step("reap-direct-child", reap)
            # Each step is independent: even a failed wait still closes pipes
            # and attempts both logs. Cleanup errors cannot turn into success.
            if selector is not None:
                cleanup_step("close-selector", selector.close)
            if proc is not None:
                for name in chunks:
                    pipe = getattr(proc, name)
                    if pipe is not None:
                        cleanup_step("close-" + name, pipe.close)
            for name, data in chunks.items():
                def write_log(name=name, data=data):
                    with (output / (label + "." + name)).open("xb") as f:
                        f.write(data)
                cleanup_step("write-" + name, write_log)
            def write_cleanup():
                with (output / (label + ".cleanup.json")).open("x") as f:
                    json.dump({"spawned": proc is not None, "direct_child_reaped": reaped,
                               "cancelled_signal": cancelled[0] if cancelled else None,
                               "errors": cleanup_errors}, f)
                    f.write("\n")
            cleanup_step("write-cleanup", write_cleanup)
        finally:
            # Handlers stay flag-only through kill/reap/logging. Restore the
            # caller's handlers on all paths, including partial setup failure.
            for signum, handler in prior_handlers.items():
                cleanup_step("restore-signal", lambda s=signum, h=handler: signal.signal(s, h))
    if primary is not None:
        if cleanup_errors and hasattr(primary, "add_note"):
            primary.add_note("CPU child cleanup had errors; inspect retained cleanup log")
        raise primary.with_traceback(primary_traceback)
    check_cancelled()  # A signal during cleanup must not produce false success.
    require(not cleanup_errors, label + " cleanup failed; inspect retained logs")
    return outcome


def compile_command(compiler, output, version):
    # The pinned GNU/Linux x86-64 toolchain uses cc with bundled LLD.
    # Bound its worker fan-out without relaxing the child's address-space cap.
    hosts = re.findall(r"^host: (\S+)$", version, re.MULTILINE)
    require(hosts == ["x86_64-unknown-linux-gnu"], "CPU regression requires one x86_64-unknown-linux-gnu host")
    return [str(compiler), "--edition=2024", "--test", "-C", "debuginfo=0",
            "-C", "codegen-units=1", "-C", "link-arg=-Wl,--threads=1",
            str(output / "cpu.rs"), "-o", str(output / "tests")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="existing empty private directory; otherwise create a retained temporary directory")
    parser.add_argument("--rustup", default="rustup", help="installed rustup executable; no toolchain installation is performed")
    args = parser.parse_args(argv)
    require(sys.platform.startswith("linux"), "Linux process-group/resource limits required")
    output = args.output or Path(tempfile.mkdtemp(prefix="fe2o3-completion-cpu-"))
    require(output.is_absolute(), "output must be absolute")
    info = output.lstat()
    require(stat.S_ISDIR(info.st_mode) and not output.is_symlink() and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700 and not list(output.iterdir()), "output must be fresh empty private owned directory")
    output = output.resolve()
    require(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output), "scratch must be outside the checkout")
    print("Retained CPU diagnostics: " + str(output), flush=True)
    sources, before = snapshot()
    channel = channel_from(sources["toolchain"])
    pieces = extract(sources)
    (output / "extracted").mkdir(mode=0o700)
    for name, text in pieces.items():
        path = output / name if name == "public-tests.rs" else output / "extracted" / name
        with path.open("x") as f:
            f.write(text)
    with (output / "cpu.rs").open("x") as f:
        f.write(sources["fixture"])
    generated = {p.relative_to(output).as_posix(): read_snapshot(p, SOURCE_CAP)[1]
                 for p in [output / "cpu.rs", output / "public-tests.rs", *sorted((output / "extracted").iterdir())]}
    rustup = Path(shutil.which(args.rustup) or args.rustup).resolve(strict=True)
    _, rustup_before = read_snapshot(rustup, 64 * 1024 * 1024)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "RUSTUP_HOME", "CARGO_HOME") if k in os.environ}
    env.update({"LANG": "C", "LC_ALL": "C", "RUSTUP_AUTO_INSTALL": "0", "TMPDIR": str(output)})
    installed = run_child([str(rustup), "toolchain", "list"], "installed", output, env, 10)["stdout"]
    matches = [line.split()[0] for line in installed.splitlines() if line.split() and line.split()[0].startswith(channel + "-")]
    require(len(matches) == 1 and re.fullmatch(r"[a-zA-Z0-9_.-]+", matches[0]), "exactly one installed pinned nightly required; runner never installs it")
    resolve_args = [str(rustup), "which", "--toolchain", matches[0], "rustc"]
    chosen = run_child(resolve_args, "resolve-before", output, env, 10)["stdout"].strip()
    compiler = Path(chosen)
    require(compiler.is_absolute() and compiler.name == "rustc" and compiler.parent.name == "bin" and compiler.parent.parent.name.startswith(channel + "-"), "resolved compiler does not name pinned installed channel")
    _, compiler_before = read_snapshot(compiler, 64 * 1024 * 1024)
    version = run_child([str(compiler), "-Vv"], "version", output, env, 10)["stdout"]
    env["PATH"] = str(compiler.parent) + ":/usr/bin:/bin"
    executable = output / "tests"
    run_child(compile_command(compiler, output, version), "compile", output, env, 60)
    _, executable_before = read_snapshot(executable, FILE_CAP)
    listing = run_child([str(executable), "--list"], "list", output, env, 10)["stdout"]
    verify_test_list(listing)
    result = run_child([str(executable), "--test-threads=1"], "tests", output, env, 10)["stdout"]
    require(re.search(r"test result: ok\. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;", result), "all eleven CPU tests must pass")
    require(run_child(resolve_args, "resolve-after", output, env, 10)["stdout"].strip() == chosen, "compiler channel selection changed")
    require(read_snapshot(rustup, 64 * 1024 * 1024)[1] == rustup_before and read_snapshot(compiler, 64 * 1024 * 1024)[1] == compiler_before, "tool bytes changed")
    require(read_snapshot(executable, FILE_CAP)[1] == executable_before, "CPU test executable changed")
    require(all(read_snapshot(output / p, SOURCE_CAP)[1] == value for p, value in generated.items()), "generated fixture input changed")
    require(snapshot()[1] == before, "source inputs changed during CPU regression")
    report = {"schema": "one-stop-completion-extracted-cpu-v1", "tests_passed": 11,
              "source_inputs": before, "channel": channel, "rustup": rustup_before,
              "rustc": compiler_before, "rustc_version": version, "test_executable": executable_before,
              "extracts": {k: {"bytes": len(v.encode()), "sha256": hashlib.sha256(v.encode()).hexdigest()} for k, v in pieces.items()},
              "scratch": scratch_size(output), "native_executed": False,
              "gpu_executed": False, "real_owner_constructed": False,
              "qualification": "mocked backend and clock; exact production slices only"}
    with (output / "report.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("PASS: 11 exact-source mocked-loop CPU tests; no native/GPU qualification.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("completion CPU regression refused: " + str(error), file=sys.stderr)
        sys.exit(1)
