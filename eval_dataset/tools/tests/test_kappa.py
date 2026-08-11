from eval_dataset.tools.kappa import cohens_kappa, fleiss_kappa, session_kappa

def test_cohens_kappa_perfect_agreement_is_one():
    labels = ["correct", "weakly_correct", "incorrect", "correct"]
    assert cohens_kappa(labels, labels) == 1.0

def test_cohens_kappa_no_agreement_beyond_chance_is_near_zero():
    a = ["correct"] * 10
    b = ["incorrect"] * 10
    # constant labels -> undefined chance agreement; kappa defined as 0.0 by convention here
    assert cohens_kappa(a, b) == 0.0

def test_fleiss_kappa_perfect_agreement_is_one():
    labels_by_annotator = [
        ["correct", "incorrect", "weakly_correct"],
        ["correct", "incorrect", "weakly_correct"],
        ["correct", "incorrect", "weakly_correct"],
    ]
    assert fleiss_kappa(labels_by_annotator) == 1.0

def test_session_kappa_dispatches_by_annotator_count():
    two = [["correct", "incorrect"], ["correct", "incorrect"]]
    three = [["correct"], ["correct"], ["correct"]]
    assert session_kappa(two) == cohens_kappa(*two)
    assert session_kappa(three) == 1.0
