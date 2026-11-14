# Same total pitch push (-9) spread over three layers so later layers can't fully undo it.
def apply(model, H):
    u = H.acoustic_dir("pitch")
    for l in (12, 18, 24):
        H.add_bias(l, -3.0 * u)
