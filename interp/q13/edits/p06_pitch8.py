# Bake -8 units of the (speaker/tone-controlled) pitch direction into layer 18's MLP output bias.
# Target: F0 220 -> ~120 Hz needs about -5.6 units (slope 0.109 log-Hz/unit).
def apply(model, H):
    u = H.acoustic_dir("pitch")
    H.add_bias(18, -8.0 * u)
