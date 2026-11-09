use super::*;

#[test]
fn lower_entity_fallback_is_enabled_only_for_the_entity_profile() {
    let configured = Options {
        use_entities_lower: true,
        ..Options::default()
    };

    let ordinary = configured.for_call(false, false);
    assert!(!ordinary.use_entities_lower);
    assert!(!ordinary.use_entities);

    let entity = configured.for_call(false, true);
    assert!(entity.use_entities_lower);
    assert!(entity.use_entities);
    assert!(!entity.use_exact);
    assert!(!entity.use_rules);
}

#[test]
fn protected_batch_rows_never_preflight_entity_vocabulary() {
    let configured = Options {
        use_entities: true,
        ..Options::default()
    };
    assert!(!batch_needs_entities(
        configured,
        1,
        Some(&[true]),
        Some(&[true]),
    ));
    assert!(batch_needs_entities(
        configured,
        2,
        Some(&[true, false]),
        Some(&[false, false]),
    ));
}

#[test]
fn ordinary_unmasked_batches_need_no_entity_scan() {
    assert!(!batch_needs_entities(
        Options::default(),
        usize::MAX,
        None,
        None,
    ));
    assert!(!batch_needs_entities(
        Options {
            use_entities_lower: true,
            ..Options::default()
        },
        usize::MAX,
        None,
        None,
    ));
}

#[test]
fn global_protection_short_circuits_every_entity_profile() {
    assert!(!batch_needs_entities(
        Options {
            use_entities: true,
            protected: true,
            ..Options::default()
        },
        usize::MAX,
        None,
        Some(&[true]),
    ));
}

#[test]
fn entity_preflight_scans_only_masks_that_can_change_the_answer() {
    assert!(batch_needs_entities(
        Options::default(),
        3,
        Some(&[false, true, false]),
        Some(&[false, true, true]),
    ));
    assert!(!batch_needs_entities(
        Options::default(),
        2,
        Some(&[false, true]),
        Some(&[false, true]),
    ));
    assert!(batch_needs_entities(
        Options {
            use_entities: true,
            ..Options::default()
        },
        2,
        Some(&[true, false]),
        None,
    ));
}
