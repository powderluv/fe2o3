//! Host-side target and launch checks; these confer no execution authority.

/// Canonical code-object target produced by the gfx950 runner.
pub const GFX950_TARGET: &str = "gfx950:xnack-";

/// Recognizes a gfx950 agent name with XNACK disabled or unspecified.
/// Unknown, repeated, empty, or contradictory features are rejected.
pub fn gfx950_agent_matches(name: &str) -> bool {
    agent_matches("gfx950", name)
}

fn agent_matches(processor: &str, name: &str) -> bool {
    let mut parts = name.split(':');
    if parts.next() != Some(processor) {
        return false;
    }
    let mut xnack = false;
    let mut sramecc = false;
    for part in parts {
        match part {
            "xnack-" if !xnack => xnack = true,
            "sramecc+" | "sramecc-" if !sramecc => sramecc = true,
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod target_tests {
    use super::*;
    #[test]
    fn accepts_gfx950_agent_names() {
        for name in [
            "gfx950",
            GFX950_TARGET,
            "gfx950:sramecc+:xnack-",
            "gfx950:xnack-:sramecc-",
        ] {
            assert!(gfx950_agent_matches(name), "{name}");
        }
    }
    #[test]
    fn rejects_other_targets_and_conflicting_features() {
        for name in [
            "",
            "gfx942",
            "gfx942:xnack-",
            "gfx9500",
            "gfx950:xnack+",
            "gfx950:",
            "gfx950:xnack-:xnack-",
            "gfx950:sramecc+:sramecc-",
            "gfx950:unknown",
            " gfx950",
        ] {
            assert!(!gfx950_agent_matches(name), "{name}");
        }
    }
}

/// Uses the original checked scalar shape and rounded 256-thread launch.
/// The 1x64 memory-tile mode has the same flattened output ownership.
pub fn gfx950_workgroups(
    m: u32,
    n: u32,
    k: u32,
) -> Result<Option<u32>, crate::harness::ShapeError> {
    crate::harness::Shape::checked(m, n, k)?.expected_groups()
}
#[cfg(test)]
mod launch_tests {
    use super::*;
    #[test]
    fn rounded_launch_and_no_dispatch_are_preserved() {
        assert_eq!(gfx950_workgroups(0, 257, 3).unwrap(), None);
        assert_eq!(gfx950_workgroups(7, 0, 3).unwrap(), None);
        for (n, groups) in [(1, 1), (255, 1), (256, 1), (257, 2)] {
            assert_eq!(gfx950_workgroups(1, n, 0).unwrap(), Some(groups));
        }
        assert!(gfx950_workgroups(u32::MAX, u32::MAX, 1).is_err());
    }
}

/// Matches a qualification device observation to the compiled source target.
/// This does not authenticate the code object or authorize a production launch.
pub fn qualification_target_matches(requested: &str, observed: &str, wave: u32) -> bool {
    matches!(requested, "gfx942" | "gfx950") && wave == 64 && agent_matches(requested, observed)
}

#[cfg(test)]
mod qualification_target_tests {
    use super::*;
    #[test]
    fn observed_target_and_wave_must_match_compiled_target() {
        for target in ["gfx942", "gfx950"] {
            assert!(qualification_target_matches(target, target, 64));
            assert!(qualification_target_matches(
                target,
                &format!("{target}:sramecc+:xnack-"),
                64
            ));
            assert!(!qualification_target_matches(target, target, 32));
            assert!(!qualification_target_matches(
                target,
                &format!("{target}:xnack+"),
                64
            ));
        }
        assert!(!qualification_target_matches("gfx942", "gfx950", 64));
        assert!(!qualification_target_matches("gfx950", "gfx942", 64));
        assert!(!qualification_target_matches("gfx1100", "gfx1100", 64));
    }
}
