# Late-layer pitch: layer-27-fitted pitch direction, -10 units, baked at L26 (fewer layers left to undo it).
import numpy as np, torch
D27 = "/home/sanyam-0605/Desktop/sanyam_projects/rumik-oss-model-dissection/interp/q13/dirs_L27.npz"
def apply(model, H):
    u = torch.tensor(np.load(D27)["pitch_dir"])
    H.add_bias(26, -10.0 * u)
