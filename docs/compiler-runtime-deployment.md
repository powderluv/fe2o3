# Compiler Runtime Deployment

This initial installer fills the compiler-code packaging gap left by the V3
service provisioner. It does not enable compiler execution or complete #272 M1.
The protected runtime guard, proof chain, finalizer and safe launch remain separate
requirements. An installed inventory is not evidence that its files executed.

## Inputs

The bundle contains exactly:

```text
bundle/                              mode 0700
  policy-v2                          mode 0444
  compiler-runtime-manifest-v1        mode 0444
  runtime/                           mode 0700
    <canonical manifest paths>       exact role modes
```

All directories are 0700 and every object has the bundle root's UID/GID. No
symlinks, hardlinks, extended attributes, subordinate mounts or extra paths are
accepted. Paths, counts and lengths use the existing manifest limits: 128 code
files, 256 bytes and 16 components per relative path, 1 GiB per file and 4 GiB
aggregate code. The containing directory must remain stable during verification.

Create the records with the existing `CompilerRuntimeManifestV1` and
`CompilerApprovalPolicyV2` codecs, using an independently reviewed release's
compiler closure, code hashes, proof-runtime identity and actual V3 client-profile
identity. The policy binds the complete closure and the manifest's domain-separated
identity. The proof-helper credentials must differ from both profile services.

Distribute the raw SHA-256 hashes of both complete record files independently of
the bundle. These external pins are not the records' domain-separated identities.
Do not obtain trusted pins from an untrusted bundle or use the synthetic unit-test
files as an approved compiler release. This tool does not discover ELF dependencies
or certify that the inventory includes every executable dependency.

## Verify

```sh
cargo run --locked -p fe2o3-compiler-execution-deployment \
  --bin fe2o3-compiler-runtime-deployment -- \
  verify /absolute/path/to/bundle "$POLICY_SHA256" "$MANIFEST_SHA256"
```

Verification streams exact bytes into sealed memfds, checks the canonical records
and complete roster, and rechecks source descriptors, paths and original content.
The final bounded streaming content pass also refuses a same-length overwrite
when its observed metadata snapshot is unchanged. Metadata/path checks bracket
that pass; this is not protection against privileged concurrent writers.
The returned
move-only owner exposes inert policy/inventory data, not descriptors or compiler
authority. Removing or changing the bundle afterward does not change its copies.

## Assemble Unapproved Records

The same command can assemble a new bundle from explicit operator-selected
inputs. This is not release approval. The six closure pins, proof-runtime
identity and per-file expected hashes are **UNAPPROVED input assertions** until
independent release review establishes their provenance. The assembler measures
every listed code file and compares all bytes to its expected hash; it does not
remeasure Cargo, the wrappers, the complete rustc lib tree or the proof runtime.
Do not replace the existing canonical rustc-tree or proof-runtime identities with
a directory hash, a packaged subset, or unit-test values.

The recipe is an ASCII mode0444 single-link file, at most 64 KiB, with exactly
one final newline. It has the following fixed field order. Angle-bracket values
below denote actual reviewed inputs, not literal recipe syntax:

```text
fe2o3-compiler-runtime-package-input-v1
cargo_sha256=<64 lowercase hex digits>
trampoline_sha256=<64 lowercase hex digits>
wrapper_sha256=<64 lowercase hex digits>
rustc_sha256=<64 lowercase hex digits>
rustc_tree_sha256=<canonical full rustc lib-tree pin>
backend_sha256=<64 lowercase hex digits>
proof_runtime_identity=<existing reviewed proof-runtime identity>
helper_uid=<canonical positive decimal>
helper_gid=<canonical positive decimal>
<role> <positive byte length> <64 lowercase hex digits> <relative path>
```

Rows are strictly sorted by relative path and use exactly one space between
fields. No comments, blank lines, extra fields, duplicate paths, file/directory
prefix collisions, links or traversal are accepted. Closed roles are `rustc`,
`backend`, `proc-macro`, `interpreter`, `proof-helper`, and `shared-library`.
The existing manifest codec requires all five singleton roles and at least one
shared library, its existing count/path/byte bounds, and matching rustc/backend
closure pins. This is an explicit roster, not automatic ELF dependency discovery.

`SOURCE_ROOT/<relative path>` must contain each exact-role-mode source file.
The source root and traversed directories are mode0700 and all source objects
belong to the invoking effective UID/GID, with no xattrs or multiple links.
Unlisted source files are not selected or copied. `PROFILE_ROOT` is a private
root-owned mode0700 offline root containing the genuinely provisioned fixed
`etc/fe2o3/compiler-execution/client-profile-v3`, mode0444, beneath root-owned
mode0755 parents. The assembler reads that original profile and uses its actual
identity; helper UID/GID must differ from both profile services.

The ordinary nonroot compiler stage is not directly suitable for a later root
packaging invocation: source ownership must match the actual invoking effective
UID/GID, while the genuine profile root remains root-owned mode0700. Before
packaging, the authorized operator must prepare NEW private source copies with
that operator's ownership, exact role modes and the original expected per-file
lengths/hashes, preserving the original stage and receipts unchanged. The
[privileged packaging preparation procedure](runtime-release-installation.md#privileged-packaging-preparation)
defines this separate step. The CLI has no preparation subcommand; do not
substitute in-place chown/chmod of the build stage, sudo builds, profile
permission changes or invented proof/approval identities. UNAPPROVED package
output still requires independent complete-record approval afterward.

`DESTINATION` must be absent beneath an existing private mode0700 parent owned
by the invoking UID/GID, outside both input trees. Paths are canonical absolute
paths of at most 4096 bytes. Exclude concurrent administrative writers and keep
input roots, directory/file identities and the output parent stable. Original
descriptors remain retained through copying and final existing-bundle verification.
Files and directory entries are synced; no symlinks or hardlinks are created.
Failure may leave an explicitly unapproved partial bundle: there is no automatic
rollback, overwrite or reuse. Discard/recover only the operator's owned scratch.

```sh
fe2o3-compiler-runtime-deployment package "$RECIPE" "$SOURCE_ROOT" \
  "$PROFILE_ROOT" "$DESTINATION" "$WORK_LIMIT" "$STORAGE_LIMIT"
```

The two limits are canonical positive decimal `usize` values and fund one
original budget for recipe parsing and packaging. The library exposes
`CompilerRuntimePackagePlanV1::READ_WORK`, `READ_STORAGE`, returned-plan `STORAGE`,
and `plan.quota().work()/storage()`. Retain the returned plan charge on the caller
account before packaging. Package storage is additional peak above that floor;
the quota covers full source/output/verifier backing overlap, fixed streaming
buffers, bounded metadata and nested canonical profile/record codecs. Work,
peak and failed-request history are never reset or refunded. These are logical
bounds, not wall-clock, allocator-RSS or ELF-runtime enforcement guarantees.

Success emits `release_approval=UNAPPROVED`, `unapproved_policy_sha256`,
`unapproved_manifest_sha256`, and `compiler_execution_authority=false`.
These are reviewable raw record hashes, not approved owner construction. The
assembler does not install anything, provision a profile, start services or run
input executables. Independent approval must precede using these hashes as
trusted installer pins.

The ignored `compiler_runtime_package::tests::native_public_package_requires_root_owned_profile_and_roundtrips`
test requires `FE2O3_RUNTIME_PACKAGE_NATIVE=isolated-disposable-root` and effective
UID/GID 0 before creating any fixtures. It tests the public root-ownership branch
only in TempDir-owned scratch with explicitly synthetic inert profile/code bytes;
it is not qualification of an approved compiler or runtime.

## Install

The existing `scripts/build-static-compiler-execution-deployment-verifier.sh`
builds this command alongside the service deployment tools and applies the same
static ELF, loader-independence and argument-gate checks. It does not add the
command to the service inventory or invoke installation during provisioning.

Use a dedicated, offline root. It must be root-owned mode0700, not the running
host root, and already contain protected `opt/fe2o3`, `etc/fe2o3`, and a genuinely
provisioned `etc/fe2o3/compiler-execution/client-profile-v3`. The installer does
not manufacture a profile, start services or prove that the root is offline.
The operator must exclude services and privileged concurrent writers.

Run the built tool as root with filesystem support and `CAP_LINUX_IMMUTABLE`:

```sh
fe2o3-compiler-runtime-deployment install /absolute/path/to/bundle \
  "$POLICY_SHA256" "$MANIFEST_SHA256" /absolute/path/to/offline-root
```

Both `opt/fe2o3/compiler-runtime-v1` and `etc/fe2o3/build-authority` must be absent.
There is no upgrade, overwrite, existing-install reuse, legacy-policy fallback,
or read-only-mount substitute for `FS_IMMUTABLE`.

The installer retains the original profile and created files, verifies the profile
binding, copies sealed inputs, sets exact ownership/modes and immutable flags,
and syncs content and directories. It validates exact rosters, inodes, hashes,
metadata and immutable flags before publishing the runtime directory, before
publishing the policy/manifest directory, and before reporting success. Checking
immutability never reapplies a lost flag. Publication uses no-replace directory
renames because immutable files themselves cannot be renamed.

These are two publications, not an atomic pair. Errors after staging begins report
the stage, root device/inode, planned paths and whether each staging directory was
created. Runtime-only publication and post-publication validation/durability
failure are not success. The tool neither clears immutable flags nor rolls back
published data. Keep the root offline and recover explicitly or discard the
dedicated root. Path replacement can displace the reported names; use the retained
root identity in the diagnostic, not blind recursive cleanup of a pathname.

## Validation Boundary

The library tests use synthetic code bytes and explicitly synthetic immutability
callbacks for filesystem and interruption controls. A separately ignored native
test exercises the public installer, real immutable flags, directory publication
and no-overwrite behavior. Its runner must confine it to fresh private scratch and
perform terminal cleanup even after failure. Neither test family executes rustc,
provisions an approved release, runs Verus or launches a GPU kernel.

The existing fixed-origin production approval/runtime constructors remain the
only admission path. `RuntimeEnforcementUnavailable` is not removed by packaging.

The separately pinned proof runtime accepts its external system interpreter in
exact mode `0555` or `0755`. It retains the observed mode, so switching between
those modes after admission still refuses. Root ownership, one link, complete
hash, canonical symlink chain and no-cross-device final-file lookup remain
mandatory. This compatibility does not permit a bind-file alias, change the
compiler inventory's `0555` role mode, or solve deployment path/inode joins.

## Observe The Canonical Rustc Library Tree

`cargo fe2o3 engineering rustc-runtime --lib-tree /absolute/toolchain/lib`
prints a single `RustcLibTreeObservationV1` JSON record with the canonical
library-tree SHA-256 used by the production Cargo pinning path. The directory
must be an absolute, lexically canonical UTF-8 path with no symlink components.
The command uses the existing `PinnedRustcLibTree` content transcript, mutation
journal, limits, and final revalidation; it does not run rustc or alter the tree.

The record has `authority: false` and `complete_elf_closure: false`. It measures
the supplied library tree only, not the rustc executable, dynamic loader, system
libraries, compiler closure approval, or an installed execution profile. A later
production invocation still independently pins and validates its actual tree
against the declared digest. This observation does not construct an admitted
profile, provision a service, or authorize compilation, publication, or launch.
