use super::*;
use fe2o3_amdhsa_loader::{KernelGlobalBufferAbiV1, validate};
use rmpv::{Value, decode::read_value, encode::write_value};
use std::io::Cursor;

// Parser-only adversarial data, never execution or instruction-refinement evidence.
pub(in crate::queue) fn synthetic_gfx950_image_v1() -> Vec<u8> {
    let mut image = IMAGE.to_vec();
    let envelope = validate(&image, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    let metadata = envelope.plan().metadata_note();
    let mut cursor = Cursor::new(envelope.metadata_descriptor());
    let mut document = read_value(&mut cursor).unwrap();
    assert_eq!(cursor.position(), metadata.byte_len());
    let Value::Map(fields) = &mut document else {
        panic!("fixture metadata map");
    };
    let (_, target) = fields
        .iter_mut()
        .find(|(key, _)| key.as_str() == Some("amdhsa.target"))
        .unwrap();
    *target = Value::from("amdgcn-amd-amdhsa--gfx950:xnack-");
    let mut encoded = Vec::new();
    write_value(&mut encoded, &document).unwrap();
    assert_eq!(encoded.len() as u64, metadata.byte_len());
    let offset = usize::try_from(metadata.file_offset()).unwrap();
    image[offset..offset + encoded.len()].copy_from_slice(&encoded);
    image[48..52].copy_from_slice(
        &AdmittedProfile::Gfx950XnackOffCov6
            .elf_flags()
            .to_le_bytes(),
    );
    image
}

pub(in crate::queue) fn synthetic_gfx950_program_v1(image: &[u8]) -> ValidatedKernelEnvelope<'_> {
    validate(image, AdmittedProfile::Gfx950XnackOffCov6)
        .unwrap()
        .bind_kernel("inplace_transform")
        .unwrap()
        .reconcile_dispatch_abi(
            [0x95; 32],
            &[KernelGlobalBufferAbiV1::new(
                0,
                "data",
                0,
                1,
                ArgumentAccess::ReadWrite,
            )],
        )
        .unwrap()
}

const TARGET_ERROR: &str = "executable target is not gfx942:xnack-";

#[test]
fn gfx942_profile_guard_rejects_the_inspected_gfx950_envelope() {
    let image = synthetic_gfx950_image_v1();
    let program = synthetic_gfx950_program_v1(&image);
    assert_eq!(
        program.envelope().plan().profile(),
        AdmittedProfile::Gfx950XnackOffCov6
    );
    assert!(matches!(
        validate_gfx942_kernel_profile(&program),
        Err(Gfx942DispatchBindingErrorV1::InvalidCode(TARGET_ERROR))
    ));
    assert!(validate_gfx942_kernel_profiles(&programs()).is_ok());
}

#[test]
fn gfx942_fixed_planner_rejects_every_cross_target_roster_position() {
    let image = synthetic_gfx950_image_v1();
    let layout = Gfx942FixedDispatchDataLayoutV1::device_local(4096, 4096);
    assert!(
        plan_public_fixed_dispatch_resources(&programs(), &[packet(0)], &[layout], &[true]).is_ok()
    );
    for foreign in 0..3 {
        let mut programs = programs();
        programs[foreign] = synthetic_gfx950_program_v1(&image);
        assert!(matches!(
            plan_public_fixed_dispatch_resources(&programs, &[packet(0)], &[layout], &[true]),
            Err(Gfx942DispatchBindingErrorV1::InvalidCode(TARGET_ERROR))
        ));
    }
}

#[test]
fn gfx942_packet_projection_rejects_gfx950_before_hidden_argument_mutation() {
    let image = synthetic_gfx950_image_v1();
    let program = synthetic_gfx950_program_v1(&image);
    assert!(project_gfx942_fixed_host_packet_v1(&programs()[0], packet(0), &[4096]).is_ok());
    assert!(matches!(
        project_gfx942_fixed_host_packet_v1(&program, packet(0), &[4096]),
        Err(Gfx942DispatchBindingErrorV1::InvalidCode(TARGET_ERROR))
    ));
}

#[test]
fn gfx942_preparation_rejects_gfx950_without_native_calls_or_input_loss() {
    let image = synthetic_gfx950_image_v1();
    for foreign in 0..3 {
        let mut programs = programs();
        programs[foreign] = synthetic_gfx950_program_v1(&image);
        let mut memory = Memory::new(true);
        let data = memory.roster();
        let expected = inputs(&data);
        let before = memory.observation();
        let mut owner = FixedDispatchPreparationCustodyV1::new([packet(0)], data);
        assert!(matches!(
            prepare_public_fixed_dispatch_resources_in_place(&mut memory, &programs, &mut owner),
            Err(Gfx942DispatchBindingErrorV1::InvalidCode(TARGET_ERROR))
        ));
        assert_eq!(owner.stage, PreparationStageV1::Plan);
        assert!(owner.failed);
        assert!(!owner.native_started);
        assert!(owner.plan.is_none());
        assert!(owner.code.is_empty());
        assert!(owner.completed.is_none());
        assert_inputs(&owner, &expected);
        assert_eq!(memory.observation(), before);
    }
}

#[test]
fn gfx942_replacement_preflight_rejects_gfx950_without_changing_data() {
    let image = synthetic_gfx950_image_v1();
    let programs = [synthetic_gfx950_program_v1(&image)];
    let mut memory = Memory::new(true);
    let data = memory.roster();
    let before = memory.observation();
    let expected = inputs(&data);
    assert!(matches!(
        preflight_gfx942_fixed_dispatch_replacement(4096, &programs, &[packet(0)], &data, 7),
        Err(Gfx942DispatchBindingErrorV1::InvalidCode(TARGET_ERROR))
    ));
    assert_eq!(inputs(&data), expected);
    assert_eq!(memory.observation(), before);
}

#[test]
fn gfx942_target_refusal_stays_terminal_when_valid_programs_are_retried() {
    let image = synthetic_gfx950_image_v1();
    for foreign in 0..3 {
        let mut selected = programs();
        selected[foreign] = synthetic_gfx950_program_v1(&image);
        let mut memory = Memory::new(true);
        let data = memory.roster();
        let expected = inputs(&data);
        let before = memory.observation();
        let mut owner = FixedDispatchPreparationCustodyV1::new([packet(0)], data);
        assert!(matches!(
            prepare_public_fixed_dispatch_resources_in_place(&mut memory, &selected, &mut owner),
            Err(Gfx942DispatchBindingErrorV1::InvalidCode(TARGET_ERROR))
        ));

        for _ in 0..2 {
            assert!(matches!(
                prepare_public_fixed_dispatch_resources_in_place(
                    &mut memory,
                    &programs(),
                    &mut owner,
                ),
                Err(Gfx942DispatchBindingErrorV1::ResourcePhase)
            ));
            assert_eq!(owner.stage, PreparationStageV1::Plan);
            assert!(owner.failed);
            assert!(!owner.native_started);
            assert!(owner.plan.is_none());
            assert!(owner.code.is_empty());
            assert!(owner.completed.is_none());
            assert_inputs(&owner, &expected);
            assert_eq!(memory.observation(), before);
        }
    }
}
