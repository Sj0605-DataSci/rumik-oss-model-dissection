# New speaker "Arjun" via the speaker-binding heads' VALUE path (conditional rank-1 edit).
# key  b = LN'd L15 input at the 'jun' name position for "Arjun:" minus mean over the 4 real names
# out  a = per-head value vector s.t.  sum_h W_O,h a_h ~= male target (L19 synthetic-male effect)
# edit W_V,kv += a_kv b^T / |b|^2   -> fires only when the value input looks like the Arjun name.
# Uses 5 tiny prompt-only forward passes (no audio) to read name representations.
import numpy as np, torch
Z = np.load("/home/sanyam-0605/Desktop/sanyam_projects/rumik-oss-model-dissection/interp/q13/male_dirs.npz")
GAIN = 1.0
HEADS = [4, 9, 11, 6, 14, 7]; L = 15
@torch.no_grad()
def name_rep(model, H, name):
    ids = torch.tensor([H.tok(f"<text>{name}: hello<audio>").input_ids], device="cuda")
    grab = {}
    hk = H.layer(L).input_layernorm.register_forward_hook(lambda m, i, o: grab.__setitem__("x", o.detach()))
    model.model(input_ids=ids, use_cache=False); hk.remove()
    toks = H.tok.convert_ids_to_tokens(ids[0]); pos = toks.index(":") - 1   # last name token
    return grab["x"][0, pos].float().cpu()
def apply(model, H):
    cfg = model.config; hd = cfg.head_dim; rep = cfg.num_attention_heads // cfg.num_key_value_heads
    b = name_rep(model, H, "Arjun") - torch.stack([name_rep(model, H, n) for n in ["Ira", "Aisha", "Siya", "Zoya"]]).mean(0)
    target = torch.tensor(Z["male_dir_L19"], dtype=torch.float32) * float(Z["male_norm_L19"]) * GAIN
    at = H.layer(L).self_attn
    Wo = torch.cat([at.o_proj.weight[:, h * hd:(h + 1) * hd].float().cpu() for h in HEADS], 1)   # d x (6*hd)
    a = torch.linalg.lstsq(Wo, target[:, None], rcond=None).solution[:, 0]                       # 6*hd
    for i, h in enumerate(HEADS):
        kv = h // rep; ah = a[i * hd:(i + 1) * hd]
        upd = torch.outer(ah, b) / (b @ b)
        at.v_proj.weight.data[kv * hd:(kv + 1) * hd] += upd.to(at.v_proj.weight.dtype).to(at.v_proj.weight.device)
