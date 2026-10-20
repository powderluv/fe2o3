use std::{fmt, str::FromStr};

use sha2::{Digest, Sha256};

use crate::{DigestError, RustcInvocationDescriptorV3, encode_descriptor_v3};

/// Domain separator for V3 rustc build-invocation coordination identities.
///
/// The terminating NUL is part of the normative byte sequence.
pub const INVOCATION_DIGEST_DOMAIN_V3: &[u8] = b"FE2O3/RUSTC-BUILD-INVOCATION/V3\0";

/// A nonzero SHA-256 coordination identity for one canonical V3 descriptor.
///
/// This digest is not artifact evidence or launch authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct InvocationDigestV3([u8; 32]);

impl InvocationDigestV3 {
    /// Calculates the V3 domain-separated digest of a descriptor.
    pub fn calculate(descriptor: &RustcInvocationDescriptorV3) -> Result<Self, DigestError> {
        let encoded = encode_descriptor_v3(descriptor)?;
        Self::calculate_encoded(&encoded)
    }

    /// Hashes already-encoded bytes without allocating or re-encoding them.
    ///
    /// This does not validate canonicality or authenticate the input. Callers
    /// requiring a descriptor must retain its original validated encoding.
    pub fn calculate_encoded(encoded: &[u8]) -> Result<Self, DigestError> {
        let mut hasher = Sha256::new();
        hasher.update(INVOCATION_DIGEST_DOMAIN_V3);
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
        Self::from_bytes(hasher.finalize().into())
    }

    /// Constructs a digest from bytes, rejecting the reserved all-zero value.
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, DigestError> {
        if bytes == [0; 32] {
            return Err(DigestError::ReservedAllZero);
        }
        Ok(Self(bytes))
    }

    /// Parses exactly 64 canonical lowercase hexadecimal characters.
    pub fn from_hex(value: &str) -> Result<Self, DigestError> {
        if value.len() != 64 {
            return Err(DigestError::InvalidHexLength);
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex_nibble(pair[0])
                .ok_or(DigestError::InvalidHexCharacter { index: index * 2 })?;
            let low = decode_hex_nibble(pair[1]).ok_or(DigestError::InvalidHexCharacter {
                index: index * 2 + 1,
            })?;
            bytes[index] = (high << 4) | low;
        }
        Self::from_bytes(bytes)
    }

    /// Borrows the 32 digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns the 32 digest bytes by value.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Returns canonical lowercase hexadecimal text.
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        output
    }
}

impl fmt::Display for InvocationDigestV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl FromStr for InvocationDigestV3 {
    type Err = DigestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_hex(value)
    }
}

fn decode_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}
