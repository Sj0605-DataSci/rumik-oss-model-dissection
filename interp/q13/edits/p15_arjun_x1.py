# New speaker "Arjun": edit ONLY the embedding of the Arjun name token(s) so that the layer-15
# speaker-binding heads (Q2: L15 h4,h9,h11,h6,h14,h7) copy a vector into the audio stream whose
# output lands on the synthetic-male direction. v = lstsq( sum_h OV_h * gamma , male_dir ).
import os, numpy as np, torch
Z = np.load("/home/sanyam-0605/Desktop/sanyam_projects/rumik-oss-model-dissection/interp/q13/male_dirs.npz")
SCALE = float("1.0")
HEADS = [(15, 4), (15, 9), (15, 11), (15, 6), (15, 14), (15, 7)]
def apply(model, H, scale=None):
    scale = SCALE if scale is None else scale
    cfg = model.config; hd = cfg.head_dim; rep = cfg.num_attention_heads // cfg.num_key_value_heads
    target = torch.tensor(Z["male_dir_L19"], dtype=torch.float32) * float(Z["male_norm_L19"])
    OV = torch.zeros(cfg.hidden_size, cfg.hidden_size)
    for l, h in HEADS:
        at = H.layer(l).self_attn; kv = h // rep
        Wo = at.o_proj.weight[:, h * hd:(h + 1) * hd].float().cpu()
        Wv = at.v_proj.weight[kv * hd:(kv + 1) * hd, :].float().cpu()
        g = H.layer(l).input_layernorm.weight.float().cpu()
        OV += Wo @ Wv * g[None, :]
    v = torch.linalg.lstsq(OV, target[:, None], rcond=None).solution[:, 0]
    v = v / v.norm() * target.norm()          # same raw size as the male effect
    E = H.embed()
    for t in [68211]:  # "jun" only; "Ar" is a common word-start token
        E.data[t] += (scale * v).to(E.dtype).to(E.device)
