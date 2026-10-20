#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

mod codegen_metadata_v1;
mod decode;
mod decode_v2;
mod decode_v3;
mod digest;
mod digest_v2;
mod digest_v3;
mod encode;
mod encode_v2;
mod encode_v3;
mod error;
mod model;
mod model_v2;
mod model_v3;
mod portable_metadata_v1;
mod reference_enrollment_v1;
mod rustc_args_v2;

pub use codegen_metadata_v1::{
    CARGO_METADATA_BUILD_OBSERVATION_DOMAIN_V2, CARGO_METADATA_BUILD_OBSERVATION_ENV_V2,
    CargoMetadataBuildObservationV2, RustcCodegenMetadataErrorV1,
    derive_cargo_metadata_build_observation_v2, ordered_rustc_codegen_metadata_v1,
};
pub use decode::decode_descriptor_v1;
pub use decode_v2::decode_descriptor_v2;
pub use decode_v3::decode_descriptor_v3;
pub use digest::{INVOCATION_DIGEST_DOMAIN_V1, InvocationDigest};
pub use digest_v2::{INVOCATION_DIGEST_DOMAIN_V2, InvocationDigestV2};
pub use digest_v3::{INVOCATION_DIGEST_DOMAIN_V3, InvocationDigestV3};
pub use encode::{
    INVOCATION_DESCRIPTOR_MAGIC, INVOCATION_DESCRIPTOR_VERSION, encode_descriptor_v1,
};
pub use encode_v2::{
    INVOCATION_DESCRIPTOR_MAGIC_V2, INVOCATION_DESCRIPTOR_VERSION_V2, encode_descriptor_v2,
};
pub use encode_v3::{
    INVOCATION_DESCRIPTOR_MAGIC_V3, INVOCATION_DESCRIPTOR_VERSION_V3, encode_descriptor_v3,
};
pub use error::{DecodeError, DigestError, ValidationError};
pub use model::{
    AmdTargetIdTextV1, BackendToolsV1, CargoIdentityV1, CargoPackageV1, CargoTargetKindV1,
    CargoTargetV1, CompileEnvironmentEntryV1, CrateTypeV1, DeviceConfigurationV1, EditionV1,
    MAX_ARGUMENT_BYTES, MAX_COMPILE_ENVIRONMENT_ENTRIES, MAX_DESCRIPTOR_BYTES,
    MAX_ENVIRONMENT_VALUE_BYTES, MAX_FEATURES, MAX_NAME_BYTES, MAX_PATH_BYTES, MAX_RUSTC_ARGUMENTS,
    MAX_TEXT_BYTES, OutputDomainV1, RustcIdentityV1, RustcInvocationDescriptorV1, RustcUnitV1,
    TestStateV1, ToolIdentityV1, VerificationModeV1,
};
pub use model_v2::{
    CompileEnvironmentEntryV2, CompileEnvironmentV2, MAX_ARGUMENT_BYTES_V2,
    MAX_COMPILE_ENVIRONMENT_ENTRIES_V2, MAX_DESCRIPTOR_BYTES_V2, MAX_ENVIRONMENT_VALUE_BYTES_V2,
    MAX_NAME_BYTES_V2, MAX_PATH_BYTES_V2, MAX_RUSTC_ARGUMENTS_V2, RustcInvocationDescriptorV2,
    RustcUnitV2,
};
pub use model_v3::{MAX_DESCRIPTOR_BYTES_V3, RustcInvocationDescriptorV3};
pub use portable_metadata_v1::{
    PORTABLE_SELECTED_METADATA_DOMAIN_V1, PortableMetadataErrorV1, PortablePackageIdentityV1,
    capture_cargo_package_identity_v1, portable_rustc_metadata_v1,
};
pub use reference_enrollment_v1::{
    MAX_REFERENCE_ENROLLMENT_BINDINGS_V1, MAX_REFERENCE_ENROLLMENT_BYTES_V1,
    MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1, REFERENCE_ENROLLMENT_ENV_V1,
    ReferenceEnrollmentBindingV1, ReferenceEnrollmentDecodeErrorV1, ReferenceEnrollmentRequestV1,
};
pub use rustc_args_v2::{
    RUSTC_SEPARATE_VALUE_OPTIONS_V2, RustcArgsErrorV2, RustcCompileInvocationV2, RustcInvocationV2,
    RustcPassthroughInvocationV2, classify_rustc_invocation_v2,
    is_rustc_codegen_backend_option_value_v2, is_rustc_codegen_backend_selector_v2,
    is_rustc_option_terminator_v2,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_v2;
#[cfg(test)]
mod tests_v3;
