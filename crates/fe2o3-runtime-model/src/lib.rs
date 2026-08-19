#![no_std]
#![forbid(unsafe_code)]

//! Pure Rust executable model for issue #137 runtime lifecycles.
//!
//! The model performs no I/O and grants no KFD, DRM, load, dispatch, completion,
//! or proof authority. It is the finite state-machine carrier that future Verus
//! specifications and syscall refinement layers can relate to concrete runtime
//! execution.
//!
//! All identities, observations, and transitions are intentionally constructible
//! by model clients. Therefore no value from this crate is runtime evidence.
//! Production adapters must seal identity and quiescence witnesses and prove a
//! refinement from their concrete operations before consuming modeled states.

extern crate alloc;

mod device_identity;
mod device_projection;
mod identity;
mod memory_lifecycle;
mod model;
mod queue_lifecycle;

pub use device_identity::*;
pub use device_projection::*;
pub use identity::*;
pub use memory_lifecycle::*;
pub use model::*;
pub use queue_lifecycle::*;

#[cfg(test)]
mod device_identity_tests;
#[cfg(test)]
mod device_projection_tests;
#[cfg(test)]
mod memory_lifecycle_tests;
#[cfg(test)]
mod queue_lifecycle_tests;
#[cfg(test)]
mod tests;
