# Pitch -6 plus formant directions (lower F1, F2 = longer vocal tract) baked at L18.
def apply(model, H):
    v = -6.0 * H.acoustic_dir("pitch") - 2.0 * H.acoustic_dir("f1") - 2.0 * H.acoustic_dir("f2")
    H.add_bias(18, v)
