use std::{error::Error, fmt};

use sha2::{Digest, Sha256};

use crate::{
    KERNEL_IR_MAGIC_V1, KERNEL_IR_VERSION_V9, KernelIrDecodeError, KernelIrEncodeError,
    MAX_MODULE_BYTES_V1, Module, VerificationErrors, decode_module_v9, encode_module_v9,
    verify_module,
};

/// Exact domain bytes for verified canonical Kernel IR V9 policy identities.
pub const VERIFIED_CANONICAL_KERNEL_IR_V9_IDENTITY_DOMAIN_V1: &[u8] =
    b"FE2O3/VERIFIED-CANONICAL-KERNEL-IR/V9\0";
/// Frozen policy version for exact, semantically verified Kernel IR V9 ownership.
pub const VERIFIED_CANONICAL_KERNEL_IR_V9_IDENTITY_POLICY_V1: u16 = 1;

const VERSION_OFFSET: usize = 8;
const VERSION_END: usize = VERSION_OFFSET + 2;

/// Typed identity minted only for exact canonical Kernel IR V9 bytes accepted
/// by the semantic verifier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VerifiedCanonicalKernelIrIdentityV9 {
    digest: [u8; 32],
    canonical_length: u64,
}

impl VerifiedCanonicalKernelIrIdentityV9 {
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub const fn canonical_length(&self) -> u64 {
        self.canonical_length
    }
}

/// Move-only owner of one exact V9 encoding whose decoded module passed
/// semantic verification.
#[derive(Debug, Eq, PartialEq)]
pub struct VerifiedCanonicalKernelIrV9 {
    canonical_bytes: Vec<u8>,
    identity: VerifiedCanonicalKernelIrIdentityV9,
}

impl VerifiedCanonicalKernelIrV9 {
    pub fn from_module(module: Module) -> Result<Self, VerifiedCanonicalKernelIrErrorV9> {
        let canonical_bytes =
            encode_module_v9(&module).map_err(VerifiedCanonicalKernelIrErrorV9::Encode)?;
        let decoded = decode_exact_v9(&canonical_bytes)?;
        verify_module(&decoded).map_err(VerifiedCanonicalKernelIrErrorV9::Verification)?;
        if decoded != module {
            return Err(VerifiedCanonicalKernelIrErrorV9::RoundTripMismatch);
        }
        Ok(Self::from_validated_bytes(canonical_bytes))
    }

    pub fn from_canonical_bytes(
        canonical_bytes: Vec<u8>,
    ) -> Result<Self, VerifiedCanonicalKernelIrErrorV9> {
        Self::from_canonical_bytes_with_module(canonical_bytes).map(|(owner, _)| owner)
    }

    /// Takes ownership of exact canonical V9 bytes and returns both their owner and the same
    /// semantically verified decoded module without performing a second full decode.
    pub fn from_canonical_bytes_with_module(
        canonical_bytes: Vec<u8>,
    ) -> Result<(Self, Module), VerifiedCanonicalKernelIrErrorV9> {
        let decoded = decode_exact_v9(&canonical_bytes)?;
        verify_module(&decoded).map_err(VerifiedCanonicalKernelIrErrorV9::Verification)?;
        Ok((Self::from_validated_bytes(canonical_bytes), decoded))
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub const fn identity(&self) -> &VerifiedCanonicalKernelIrIdentityV9 {
        &self.identity
    }

    pub fn revalidate(&self) -> Result<(), VerifiedCanonicalKernelIrErrorV9> {
        let decoded = decode_exact_v9(&self.canonical_bytes)?;
        verify_module(&decoded).map_err(VerifiedCanonicalKernelIrErrorV9::Verification)?;
        if canonical_identity(&self.canonical_bytes) != self.identity {
            return Err(VerifiedCanonicalKernelIrErrorV9::IdentityMismatch);
        }
        Ok(())
    }

    pub fn into_canonical_bytes(self) -> Vec<u8> {
        self.canonical_bytes
    }

    fn from_validated_bytes(canonical_bytes: Vec<u8>) -> Self {
        let identity = canonical_identity(&canonical_bytes);
        Self {
            canonical_bytes,
            identity,
        }
    }
}

fn decode_exact_v9(bytes: &[u8]) -> Result<Module, VerifiedCanonicalKernelIrErrorV9> {
    if bytes.len() > MAX_MODULE_BYTES_V1 {
        return Err(VerifiedCanonicalKernelIrErrorV9::Decode(
            KernelIrDecodeError::TooLarge {
                max: MAX_MODULE_BYTES_V1,
            },
        ));
    }
    let magic =
        bytes
            .get(..KERNEL_IR_MAGIC_V1.len())
            .ok_or(VerifiedCanonicalKernelIrErrorV9::Decode(
                KernelIrDecodeError::Truncated,
            ))?;
    if magic != KERNEL_IR_MAGIC_V1 {
        return Err(VerifiedCanonicalKernelIrErrorV9::Decode(
            KernelIrDecodeError::InvalidMagic,
        ));
    }
    let version_bytes =
        bytes
            .get(VERSION_OFFSET..VERSION_END)
            .ok_or(VerifiedCanonicalKernelIrErrorV9::Decode(
                KernelIrDecodeError::Truncated,
            ))?;
    let version = u16::from_le_bytes([version_bytes[0], version_bytes[1]]);
    if version != KERNEL_IR_VERSION_V9 {
        return Err(VerifiedCanonicalKernelIrErrorV9::NotExactV9 { version });
    }
    decode_module_v9(bytes).map_err(VerifiedCanonicalKernelIrErrorV9::Decode)
}

fn canonical_identity(bytes: &[u8]) -> VerifiedCanonicalKernelIrIdentityV9 {
    let canonical_length =
        u64::try_from(bytes.len()).expect("hard-bounded canonical Kernel IR length fits u64");
    let domain_length = u32::try_from(VERIFIED_CANONICAL_KERNEL_IR_V9_IDENTITY_DOMAIN_V1.len())
        .expect("frozen canonical Kernel IR identity domain length fits u32");
    let mut digest = Sha256::new();
    digest.update(domain_length.to_le_bytes());
    digest.update(VERIFIED_CANONICAL_KERNEL_IR_V9_IDENTITY_DOMAIN_V1);
    digest.update(VERIFIED_CANONICAL_KERNEL_IR_V9_IDENTITY_POLICY_V1.to_le_bytes());
    digest.update(canonical_length.to_le_bytes());
    digest.update(bytes);
    VerifiedCanonicalKernelIrIdentityV9 {
        digest: digest.finalize().into(),
        canonical_length,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VerifiedCanonicalKernelIrErrorV9 {
    Encode(KernelIrEncodeError),
    Decode(KernelIrDecodeError),
    Verification(VerificationErrors),
    NotExactV9 { version: u16 },
    RoundTripMismatch,
    IdentityMismatch,
}

impl fmt::Display for VerifiedCanonicalKernelIrErrorV9 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(error) => {
                write!(formatter, "cannot encode canonical Kernel IR V9: {error}")
            }
            Self::Decode(error) => {
                write!(formatter, "cannot decode canonical Kernel IR V9: {error}")
            }
            Self::Verification(error) => error.fmt(formatter),
            Self::NotExactV9 { version } => {
                write!(
                    formatter,
                    "expected exact Kernel IR V9 bytes, found V{version}"
                )
            }
            Self::RoundTripMismatch => {
                formatter.write_str("Kernel IR V9 round trip changed bytes or semantics")
            }
            Self::IdentityMismatch => {
                formatter.write_str("canonical Kernel IR V9 identity mismatch")
            }
        }
    }
}

impl Error for VerifiedCanonicalKernelIrErrorV9 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Encode(error) => Some(error),
            Self::Decode(error) => Some(error),
            Self::Verification(error) => Some(error),
            Self::NotExactV9 { .. } | Self::RoundTripMismatch | Self::IdentityMismatch => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_owner_rejects_a_v8_envelope() {
        let module = Module::new("canonical-v9-hostile");
        let bytes = crate::encode_module_v8(&module).unwrap();
        assert!(matches!(
            VerifiedCanonicalKernelIrV9::from_canonical_bytes(bytes),
            Err(VerifiedCanonicalKernelIrErrorV9::NotExactV9 { version: 8 })
        ));
    }

    #[test]
    fn exact_owner_returns_the_same_verified_v9_module() {
        let module = Module::new("canonical-v9-module-custody");
        let bytes = crate::encode_module_v9(&module).unwrap();
        let (owner, decoded) =
            VerifiedCanonicalKernelIrV9::from_canonical_bytes_with_module(bytes.clone()).unwrap();
        assert_eq!(owner.canonical_bytes(), bytes);
        assert_eq!(decoded, module);
    }
}
