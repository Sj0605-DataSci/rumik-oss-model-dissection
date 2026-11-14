# Stacked: mid-layer pitch (L19 dir, -6 at L18) + late-layer pitch (L27 dir, -10 at L26).
import numpy as np, torch
D27 = "/home/sanyam-0605/Desktop/sanyam_projects/rumik-oss-model-dissection/interp/q13/dirs_L27.npz"
def apply(model, H):
    H.add_bias(18, -6.0 * H.acoustic_dir("pitch"))
    H.add_bias(26, -10.0 * torch.tensor(np.load(D27)["pitch_dir"]))
