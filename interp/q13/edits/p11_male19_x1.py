# Bake the synthetic-male direction (WORLD F0x0.55, formants x0.85 -> Mimi -> rumik acts) as a bias.
import numpy as np, torch
Z = np.load("/home/sanyam-0605/Desktop/sanyam_projects/rumik-oss-model-dissection/interp/q13/male_dirs.npz")
u19, n19 = torch.tensor(Z["male_dir_L19"]), float(Z["male_norm_L19"])
u27, n27 = torch.tensor(Z["male_dir_L27"]), float(Z["male_norm_L27"])
def apply(model, H):
    H.add_bias(18, 1.0 * n19 * u19)
