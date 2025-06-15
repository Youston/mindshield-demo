from mindshield_core.modality import score_modalities, choose_modality

def test_scoring():
    text = "I keep having negative thoughts and beliefs about myself."
    scores = score_modalities(text)
    assert "CBT" in scores and scores["CBT"] > 0
    chosen = choose_modality(scores)
    assert chosen == "CBT" 