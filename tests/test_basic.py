import numpy as np
from src.data import clean_text
from src.mutations import diff_features, edit_kinds
from src.cluster import labels_from_knn


def test_clean_text_removes_urls_and_symbols():
    assert clean_text("Olá!!! http://x.com ★★ MUNDO") == "olá!!! mundo"


def test_footer_and_header_detected():
    ins, dele, rep, prefix, suffix, sim = diff_features("a b c d", "a b c d e f")
    assert suffix and not prefix and ins == 2 and dele == 0
    ins, dele, rep, prefix, suffix, sim = diff_features("a b c d", "x a b c d")
    assert prefix and not suffix


def test_spelling_fix_is_not_a_rewrite():
    typo, real = edit_kinds("isso é indignificante mesmo", "isso é insignificante mesmo")
    assert typo == 1 and real == 0
    typo, real = edit_kinds("o gato dorme aqui", "a vacina chegou hoje")
    assert real > 0


def test_mutual_neighbours_prevent_chaining():
    # 0-1 are mutual neighbours; 2 lists 1 but 1 does not list 2
    sim = np.array([[1, .9, 0], [.9, 1, 0], [0, .9, 1]])
    idx = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
    lab = labels_from_knn(sim, idx, 0.5)
    assert lab[0] == lab[1] and lab[2] != lab[0]


def test_misinformation_propagation_structure():
    import pandas as pd
    from src.misinformation import analyze_propagation
    dummy = pd.DataFrame({
        "clean": ["texto de teste a", "texto de teste b"],
        "cl_tfidf": [1, 1],
        "size_tfidf": [2, 2],
        "max_sharings": [5, 2],
        "n_occurrences": [6, 3]
    })
    res, fam = analyze_propagation(dummy)
    assert "coverage_boost_pct" in res.columns
    assert fam[1] in [0, 1]

