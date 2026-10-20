//! Resolve explicit enrollment selectors against this compiler's actual items.
//! Resolution alone is descriptive; the native issuer owns admission and custody.

use super::{KernelRoot, is_fully_monomorphized};
use crate::protected_compiler_execution::native_v3::{
    ReferenceEnrollmentLoanV1, RetainedReferenceEnrollmentStampV1,
};
use crate::reference_effect_v1::{
    ReferenceBindingErrorV1, ReferenceBindingOriginV1,
    authenticate_reference_binding_with_origin_v1,
};
use crate::rustc_semantic_plan_v1::{SourceClosureWorkStampV1, SourceClosureWorkV1};
use rustc_hir::def::DefKind;
use rustc_middle::mir::mono::{CodegenUnit, MonoItem};
use rustc_middle::ty::{Instance, TyCtxt};

fn error(reason: &'static str) -> ReferenceBindingErrorV1 {
    ReferenceBindingErrorV1::new(reason)
}

fn charge(work: &mut SourceClosureWorkV1, amount: usize) -> Result<(), ReferenceBindingErrorV1> {
    work.charge(amount)
        .map_err(|failure| ReferenceBindingErrorV1::new(failure.to_string()))
}

struct ResolvedEnrollmentV1<'tcx> {
    kernel: Instance<'tcx>,
    reference: Instance<'tcx>,
    origin: ReferenceBindingOriginV1,
    logical_name: String,
}

pub(super) struct RetainedEnrollmentV1<'tcx> {
    session: &'tcx rustc_session::Session,
    issuer: RetainedReferenceEnrollmentStampV1,
    source: SourceClosureWorkStampV1,
    bindings: Box<[ResolvedEnrollmentV1<'tcx>]>,
}

impl<'tcx> RetainedEnrollmentV1<'tcx> {
    fn capture(
        tcx: TyCtxt<'tcx>,
        loan: &ReferenceEnrollmentLoanV1<'_>,
        work: &mut SourceClosureWorkV1,
    ) -> Result<Self, ReferenceBindingErrorV1> {
        charge(work, std::mem::size_of::<Self>())?;
        let issuer = loan.capture_stamp(work)?;
        let source = work
            .retain_identity_v1()
            .map_err(|failure| ReferenceBindingErrorV1::new(failure.to_string()))?;
        Ok(Self {
            session: tcx.sess,
            issuer,
            source,
            bindings: Box::default(),
        })
    }

    pub(super) fn revalidate(
        &self,
        tcx: TyCtxt<'tcx>,
        loan: &ReferenceEnrollmentLoanV1<'_>,
        work: &mut SourceClosureWorkV1,
    ) -> Result<(), ReferenceBindingErrorV1> {
        charge(work, 1)?;
        if !std::ptr::eq(self.session, tcx.sess) {
            return Err(error("reference enrollment compiler session changed"));
        }
        if !work
            .matches_identity_v1(&self.source)
            .map_err(|failure| ReferenceBindingErrorV1::new(failure.to_string()))?
        {
            return Err(error("reference enrollment source account changed"));
        }
        loan.revalidate_stamp(&self.issuer, work)
    }

    pub(super) fn validate_inputs(
        &self,
        functions: &[super::CollectedFunction<'tcx>],
        work: &mut SourceClosureWorkV1,
    ) -> Result<(), ReferenceBindingErrorV1> {
        let mut observed = 0_usize;
        for function in functions {
            charge(work, 1)?;
            if function
                .reference_effect_binding
                .as_ref()
                .is_some_and(|binding| {
                    matches!(
                        binding.origin,
                        ReferenceBindingOriginV1::ReferenceEnrollment(_)
                    )
                })
            {
                observed = observed
                    .checked_add(1)
                    .ok_or_else(|| error("reference enrollment census overflow"))?;
            }
        }
        if observed != self.bindings.len() {
            return Err(error("reference enrollment resolved roster changed"));
        }
        for expected in &self.bindings {
            let mut matched = false;
            for function in functions {
                charge(work, 1)?;
                if function.instance != expected.kernel {
                    continue;
                }
                charge(work, expected.logical_name.len())?;
                charge(work, std::mem::size_of::<ReferenceBindingOriginV1>())?;
                if matched
                    || function.role != super::CollectedFunctionRole::KernelEntry
                    || function.reference_instance != Some(expected.reference)
                    || function.logical_name.as_deref() != Some(expected.logical_name.as_str())
                    || !function
                        .reference_effect_binding
                        .as_ref()
                        .is_some_and(|binding| binding.origin == expected.origin)
                {
                    return Err(error(
                        "reference enrollment resolved instance or origin changed",
                    ));
                }
                matched = true;
            }
            if !matched {
                return Err(error("reference enrollment resolved kernel is missing"));
            }
        }
        Ok(())
    }
}

pub(super) fn bind_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    roots: &mut [KernelRoot<Instance<'tcx>>],
    loan: &ReferenceEnrollmentLoanV1<'_>,
    work: &mut SourceClosureWorkV1,
) -> Result<RetainedEnrollmentV1<'tcx>, ReferenceBindingErrorV1> {
    // Capture even when the descriptor has no mappings. An empty roster must
    // not allow import to substitute a different live session or account.
    let mut retained = RetainedEnrollmentV1::capture(tcx, loan, work)?;
    if let Some(request) = loan.request(work)? {
        charge(
            work,
            request
                .bindings()
                .len()
                .checked_mul(std::mem::size_of::<ResolvedEnrollmentV1<'tcx>>())
                .ok_or_else(|| error("reference enrollment storage overflow"))?,
        )?;
        let mut resolved = Vec::new();
        resolved
            .try_reserve_exact(request.bindings().len())
            .map_err(|_| error("reference enrollment allocation failed"))?;
        for (ordinal, binding) in request.bindings().iter().enumerate() {
            charge(work, 1)?;
            let ordinal = u32::try_from(ordinal)
                .map_err(|_| error("reference enrollment ordinal overflow"))?;
            let index = kernel_index_v1(tcx, roots, binding.kernel(), work)?;
            let reference = reference_instance_v1(tcx, cgus, binding.reference(), work)?;
            let root = &mut roots[index];
            charge(
                work,
                root.logical_name
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| error("reference enrollment name work overflow"))?,
            )?;
            let fresh = authenticate_reference_binding_with_origin_v1(
                tcx,
                ReferenceBindingOriginV1::ReferenceEnrollment(loan.origin(ordinal)?),
                root.logical_name.clone(),
                root.target,
                reference,
                work,
            )?;
            charge(
                work,
                root.logical_name
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| error("reference enrollment name work overflow"))?,
            )?;
            charge(work, std::mem::size_of::<ReferenceBindingOriginV1>())?;
            resolved.push(ResolvedEnrollmentV1 {
                kernel: root.target,
                reference,
                origin: fresh.origin.clone(),
                logical_name: root.logical_name.clone(),
            });
            root.reference_effect_binding = Some(fresh);
            root.reference_target = Some(reference);
        }
        retained.bindings = resolved.into_boxed_slice();
    }
    retained.revalidate(tcx, loan, work)?;
    Ok(retained)
}

fn matches_selector(
    tcx: TyCtxt<'_>,
    mut definition: rustc_hir::def_id::DefId,
    selector: &str,
    work: &mut SourceClosureWorkV1,
) -> Result<bool, ReferenceBindingErrorV1> {
    use rustc_hir::definitions::{DefPathData, DefPathDataName};
    // Compare borrowed canonical DefPath components, not diagnostic pretty
    // paths. No candidate String, temporary path Vec or symbol interning.
    // Prepay delimiter scans, component comparison and decimal validation.
    charge(
        work,
        selector
            .len()
            .checked_mul(4)
            .ok_or_else(|| error("reference enrollment selector work overflow"))?,
    )?;
    let mut remaining = selector;
    loop {
        charge(work, 1)?;
        let key = tcx.def_key(definition);
        if key.disambiguated_data.data == DefPathData::CrateRoot {
            let name = tcx.crate_name(definition.krate);
            charge(work, name.as_str().len())?;
            return Ok(key.parent.is_none() && remaining == name.as_str());
        }
        let Some((parent, component)) = remaining.rsplit_once("::") else {
            return Ok(false);
        };
        let data = key.disambiguated_data;
        let (name, suffix, anonymous) = match data.data.name() {
            DefPathDataName::Named(name) => (name, data.disambiguator != 0, false),
            DefPathDataName::Anon { namespace } => (namespace, true, true),
        };
        charge(work, name.as_str().len())?;
        let component = if anonymous {
            let Some(value) = component
                .strip_prefix('{')
                .and_then(|s| s.strip_suffix('}'))
            else {
                return Ok(false);
            };
            value
        } else {
            component
        };
        if suffix {
            let Some((base, ordinal)) = component.rsplit_once('#') else {
                return Ok(false);
            };
            if base != name.as_str()
                || ordinal.is_empty()
                || ordinal.len() > 1 && ordinal.starts_with('0')
                || !ordinal.bytes().all(|byte| byte.is_ascii_digit())
                || ordinal.parse::<u32>().ok() != Some(data.disambiguator)
            {
                return Ok(false);
            }
        } else if component != name.as_str() {
            return Ok(false);
        }
        let Some(index) = key.parent else {
            return Ok(false);
        };
        definition = rustc_hir::def_id::DefId {
            krate: definition.krate,
            index,
        };
        remaining = parent;
    }
}

fn kernel_index_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    roots: &[KernelRoot<Instance<'tcx>>],
    selector: &str,
    work: &mut SourceClosureWorkV1,
) -> Result<usize, ReferenceBindingErrorV1> {
    let mut selected = None;
    for (index, root) in roots.iter().enumerate() {
        if matches_selector(tcx, root.target.def_id(), selector, work)? {
            if selected.replace(index).is_some() {
                return Err(error("enrollment kernel selector is ambiguous"));
            }
        }
    }
    let index =
        selected.ok_or_else(|| error("enrollment kernel is not a registered local root"))?;
    let root = &roots[index];
    charge(work, 1)?;
    if !root.target.def_id().is_local() || !is_fully_monomorphized(tcx, root.target) {
        return Err(error(
            "enrollment kernel is not a fully monomorphized local instance",
        ));
    }
    if root.reference_effect_binding.is_some() || root.reference_target.is_some() {
        return Err(error(
            "enrollment conflicts with an existing source reference registration",
        ));
    }
    Ok(index)
}

fn reference_instance_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    selector: &str,
    work: &mut SourceClosureWorkV1,
) -> Result<Instance<'tcx>, ReferenceBindingErrorV1> {
    let mut definition = None;
    for owner in tcx.hir_body_owners() {
        charge(work, 1)?;
        let id = owner.to_def_id();
        if !matches!(tcx.def_kind(id), DefKind::Fn | DefKind::AssocFn) {
            continue;
        }
        if matches_selector(tcx, id, selector, work)? && definition.replace(id).is_some() {
            return Err(error("enrollment reference selector is ambiguous"));
        }
    }
    let definition =
        definition.ok_or_else(|| error("enrollment reference is not a local function"))?;
    if tcx.generics_of(definition).count() == 0 {
        charge(work, 1)?;
        return Ok(Instance::mono(tcx, definition));
    }

    // A selector does not invent type/const substitutions. For a generic item,
    // require one actual fully instantiated item in the current CGU inventory.
    let mut selected = None;
    for cgu in cgus {
        for (item, _) in cgu.items() {
            charge(work, 1)?;
            let MonoItem::Fn(instance) = item else {
                continue;
            };
            if instance.def_id() != definition || !is_fully_monomorphized(tcx, *instance) {
                continue;
            }
            match selected {
                Some(previous) if previous != *instance => {
                    return Err(error(
                        "enrollment reference has ambiguous monomorphizations",
                    ));
                }
                None => selected = Some(*instance),
                _ => {}
            }
        }
    }
    selected.ok_or_else(|| error("enrollment reference has no actual monomorphized instance"))
}

#[cfg(test)]
#[path = "reference_enrollment_v1_tests.rs"]
mod tests;
