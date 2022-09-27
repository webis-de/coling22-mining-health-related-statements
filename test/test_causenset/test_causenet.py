import health_causenet
import health_causenet.causenet
import pandas as pd
import pytest


@pytest.fixture
def causenet():
    causenet = pd.DataFrame(
        [
            [
                "acquired or inhered abnormality of red blood cells",
                "blood disorder condition",
            ],
            ["foo", "lala"],
            ["earthquake", "cancer"],
            ["death", "fracture at the base of the skull"],
        ],
        columns=["cause", "effect"],
    )
    return causenet


@pytest.fixture
def cf():
    cf = health_causenet.causenet.load_cf(False, False, True, True)
    return cf


def test_is_medical_quickumls(causenet):

    out = health_causenet.causenet.CauseNet.is_medical(
        causenet,
        "quickumls",
        umls_subset="mesh_syn",
        jaccard_threshold=1.0,
        st21pv=False,
    )
    assert list(out) == ["medical_score-cause", "medical_score-effect"]


def test_is_medical_scispacy(causenet):
    with pytest.raises(KeyError):
        rx_sno = health_causenet.causenet.CauseNet.is_medical(
            causenet,
            "scispacy",
            umls_subset="foo",
            model="en_core_sci_sm",
            threshold=0.9,
            verbose=False,
            st21pv=False,
        )
    rx_sno = health_causenet.causenet.CauseNet.is_medical(
        causenet,
        "scispacy",
        umls_subset="rx_sno",
        model="en_core_sci_sm",
        threshold=0.9,
        verbose=False,
        st21pv=False,
    )
    assert list(rx_sno) == ["medical_score-cause", "medical_score-effect"]
    st21pv = health_causenet.causenet.CauseNet.is_medical(
        causenet,
        "scispacy",
        umls_subset="umls",
        model="en_core_sci_sm",
        threshold=0.9,
        verbose=False,
        st21pv=True,
    )
    not_st21pv = health_causenet.causenet.CauseNet.is_medical(
        causenet,
        "scispacy",
        umls_subset="umls",
        model="en_core_sci_sm",
        threshold=0.9,
        verbose=False,
        st21pv=False,
    )
    assert not (st21pv == not_st21pv).all().all()


def test_is_medical_metamap(causenet):

    out = health_causenet.causenet.CauseNet.is_medical(
        causenet, "tagger", json_path="/home/fschlatt/metamap/metamap.jsonl"
    )
    assert list(out) == ["medical_score-cause", "medical_score-effect"]


def test_is_medical_ctakes(causenet):

    out = health_causenet.causenet.CauseNet.is_medical(
        causenet, "tagger", json_path="/home/fschlatt/ctakes/relations.jsonl"
    )
    assert list(out) == ["medical_score-cause", "medical_score-effect"]


def test_is_medical_discriminative_weight(causenet, cf):

    out = health_causenet.causenet.CauseNet.is_medical(
        causenet, "discriminative_weight", cf=cf, n_gram_size=(1, 2), p=float("inf"),
    )
    assert list(out) == ["medical_score-cause", "medical_score-effect"]


def test_is_medical_contrastive_weight(causenet, cf):

    out = health_causenet.causenet.CauseNet.is_medical(
        causenet, "contrastive_weight", cf=cf, n_gram_size=(1, 2), p=float("inf"),
    )
    assert list(out) == ["medical_score-cause", "medical_score-effect"]


def test_is_medical_term_domain_specificity(causenet, cf):

    out = health_causenet.causenet.CauseNet.is_medical(
        causenet, "term_domain_specificity", cf=cf, n_gram_size=(1, 2), p=float("inf"),
    )
    assert list(out) == ["medical_score-cause", "medical_score-effect"]

