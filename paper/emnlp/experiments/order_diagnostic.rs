
// Diagnostic build only (prepare_order_native.py --examples): appended to the harness.
// A source competes for a word when, used alone, it answers at the same lookup stage as
// the full table, that is, it holds the very key the cascade reads.
fn headroom(
    data: &[Line],
    forms: &IndexMap<String, usize>,
    pred: &[Vec<[Prediction; 2]>],
    solo: &[Vec<[Prediction; 2]>],
) -> serde_json::Value {
    let mut n = [0_u64; 6];
    for s in data {
        for (i, (form, lemma)) in s.tokens.iter().zip(&s.lemmas).enumerate() {
            let Some(y) = lemma else { continue };
            let id = forms[form];
            let initial = usize::from(i == 0);
            let reference = &pred[0][id][initial];
            if !matches!(reference.stage, Source::External | Source::ExternalLower) {
                continue;
            }
            n[0] += 1;
            let answers: Vec<&str> = solo
                .iter()
                .map(|p| &p[id][initial])
                .filter(|p| {
                    matches!(
                        (&p.stage, &reference.stage),
                        (Source::External, Source::External)
                            | (Source::ExternalLower, Source::ExternalLower)
                    )
                })
                .map(|p| p.text.as_str())
                .collect();
            if !answers.contains(&reference.text.as_str()) {
                n[5] += 1;
                continue;
            }
            if answers.iter().any(|a| *a != answers[0]) {
                n[1] += 1;
                n[2] += u64::from(reference.text == *y);
                n[3] += u64::from(pred[1][id][initial].text == *y);
                n[4] += u64::from(answers.iter().any(|a| *a == y.as_str()));
            }
        }
    }
    json!({"external": n[0], "disputed": n[1], "declared_correct": n[2],
           "learned_correct": n[3], "oracle_correct": n[4], "unmatched": n[5]})
}
