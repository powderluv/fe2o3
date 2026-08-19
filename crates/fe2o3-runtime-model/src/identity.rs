//! Stable, typed identities used by the runtime transition model.

pub const RUNTIME_IDENTITY_SCHEMA_VERSION_V1: u16 = 1;
pub const IDENTITY_DIGEST_BYTES_V1: usize = 32;

/// Opaque canonical digest supplied by a future evidence layer.
///
/// Constructing this value does not authenticate it or grant runtime authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct IdentityDigestV1([u8; IDENTITY_DIGEST_BYTES_V1]);

impl IdentityDigestV1 {
    pub const fn from_untrusted_bytes(bytes: [u8; IDENTITY_DIGEST_BYTES_V1]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; IDENTITY_DIGEST_BYTES_V1] {
        &self.0
    }
}

macro_rules! digest_identity {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(IdentityDigestV1);

        impl $name {
            pub const fn from_untrusted_digest(digest: IdentityDigestV1) -> Self {
                Self(digest)
            }

            pub const fn digest(self) -> IdentityDigestV1 {
                self.0
            }
        }
    };
}

digest_identity!(
    /// Identity of the abstract runtime semantics and invariant set.
    RuntimeModelIdV1
);
digest_identity!(
    /// Authenticated compiler artifact identity; authentication is external.
    RuntimeArtifactIdV1
);
digest_identity!(
    /// Identity of a validated code load plan.
    CodeLoadPlanIdV1
);
digest_identity!(
    /// Identity of a validated queue construction plan.
    QueuePlanIdV1
);
digest_identity!(
    /// Identity of one model-only compute-AQL queue configuration.
    QueueConfigurationIdV1
);
digest_identity!(
    /// Identity of one model-only device-observation domain.
    DeviceObservationDomainIdV1
);
digest_identity!(
    /// Identity of the reviewed device-correlation profile.
    DeviceAdmissionProfileIdV1
);

macro_rules! numeric_identity {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub u64);
    };
}

numeric_identity!(
    /// Stable physical GPU identity within an admitted platform inventory.
    PhysicalDeviceIdV1
);
numeric_identity!(
    /// Software-selected monotonic admission incarnation of one physical GPU.
    ///
    /// This value prevents model token reuse. A topology observation does not
    /// establish that a GPU reset or hardware reinitialization occurred.
    DeviceGenerationV1
);
numeric_identity!(
    /// Runtime VM identity.
    VmIdV1
);
numeric_identity!(
    /// GPU allocation identity.
    AllocationIdV1
);
numeric_identity!(
    /// Monotonic incarnation of one allocation identity within an exact VM.
    AllocationGenerationV1
);
numeric_identity!(
    /// Process-lifetime GPU virtual-address reservation identity.
    VaReservationIdV1
);
numeric_identity!(
    /// GPU virtual mapping identity.
    MappingIdV1
);
numeric_identity!(
    /// Identity of one model publication retaining a mapping.
    MemoryPublicationIdV1
);
numeric_identity!(
    /// Loaded code identity.
    LoadedCodeIdV1
);
numeric_identity!(
    /// Queue incarnation identity.
    QueueInstanceIdV1
);
numeric_identity!(
    /// Monotonic incarnation of a queue.
    QueueGenerationV1
);
numeric_identity!(
    /// Dispatch identity.
    DispatchIdV1
);
numeric_identity!(
    /// Completion observation identity.
    CompletionIdV1
);

/// A physical device incarnation. Every lower-level identity is rooted here.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceKeyV1 {
    pub physical: PhysicalDeviceIdV1,
    pub generation: DeviceGenerationV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VmKeyV1 {
    pub device: DeviceKeyV1,
    pub id: VmIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationKeyV1 {
    pub vm: VmKeyV1,
    pub id: AllocationIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MappingKeyV1 {
    pub allocation: AllocationKeyV1,
    pub id: MappingIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoadedCodeKeyV1 {
    pub vm: VmKeyV1,
    pub id: LoadedCodeIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QueueKeyV1 {
    pub vm: VmKeyV1,
    pub id: QueueInstanceIdV1,
    pub generation: QueueGenerationV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DispatchKeyV1 {
    pub queue: QueueKeyV1,
    pub id: DispatchIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompletionKeyV1 {
    pub dispatch: DispatchKeyV1,
    pub id: CompletionIdV1,
}
