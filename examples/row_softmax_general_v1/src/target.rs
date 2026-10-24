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

/// Returns a checked Wave64 workgroup count. Empty rows do not dispatch.
/// Tile mode assigns one wave per 64-column tile and repeats full-row reductions.
pub fn gfx950_workgroups(rows: u32, columns: u32, tiled: bool) -> Option<u32> {
    if columns == 0 || columns > 4096 {
        return None;
    }
    let tiles = if tiled { columns.div_ceil(64) } else { 1 };
    let groups = rows.checked_mul(tiles)?;
    groups.checked_mul(64)?;
    Some(groups)
}

#[cfg(test)]
mod launch_tests {
    use super::*;
    #[test]
    fn dynamic_column_tiles_and_launch_overflow() {
        for columns in [1, 63, 64, 65, 127, 129, 4096] {
            assert_eq!(gfx950_workgroups(3, columns, false), Some(3));
            assert_eq!(
                gfx950_workgroups(3, columns, true),
                Some(3 * columns.div_ceil(64))
            );
        }
        assert_eq!(gfx950_workgroups(0, 65, true), Some(0));
        for tiled in [false, true] {
            assert_eq!(gfx950_workgroups(1, 0, tiled), None);
            assert_eq!(gfx950_workgroups(1, 4097, tiled), None);
            assert_eq!(gfx950_workgroups(u32::MAX, 1, tiled), None);
        }
        assert_eq!(gfx950_workgroups(u32::MAX / 64, 4096, true), None);
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
