"""vLLM out-of-tree model plugin for rumik-ai/rumik-oss-1 ("TinyAya").

TinyAya is architecturally Cohere2ForCausalLM (Command-R2/Aya family) plus:
  1. a small `stop_predictor` head (LayerNorm -> Linear -> GELU -> Linear)
     that predicts a stop probability from the last hidden state, used
     during audio-token decode to end generation early (an alternative to
     relying solely on the model sampling the literal `</audio>` token).
  2. generation is constrained to a fixed sub-vocabulary of audio-codec
     tokens (`<code_q_v>` style ids, from the Mimi codec, plus `</audio>`)
     -- the model should never emit ordinary text tokens once past the
     `<audio>` prompt marker.

Both of these are folded into a single `compute_logits()` override, matching
vLLM's existing per-step hook -- no custom LogitsProcessor/StoppingCriteria
wiring needed. This mirrors vLLM's CohereForCausalLM almost exactly (see
commandr_reference.py, pulled from vllm/model_executor/models/commandr.py)
since Cohere2ForCausalLM in vLLM 0.20 is implemented by the same file/class.

Registration (see register.py) is done via
`vllm.ModelRegistry.register_model(...)` at process start, before
constructing an `LLM`/`AsyncLLMEngine` -- no changes to vLLM's own source
tree required. See PR_NOTES.md for what an in-tree contribution would add
instead.
"""

import re
from collections.abc import Iterable

import torch
from torch import nn
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.model_executor.models.commandr import CohereForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader

_AUDIO_TOKEN_RE = re.compile(r"^<(\d+)_(\d+)>$")


class TinyAyaForCausalLM(CohereForCausalLM):
    """Cohere2ForCausalLM (vLLM's CohereForCausalLM, `v1` flag off for
    Cohere2Config) + stop_predictor head, with the audio-vocab restriction
    and stop-head decision both folded into compute_logits()."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        # attribute name + submodule structure must exactly match the
        # checkpoint's flat "stop_predictor.{0,1,3}.{weight,bias}" tensor
        # names (see modeling_tinyaya.py) for AutoWeightsLoader to find them.
        hidden_size = config.hidden_size
        reduced = max(64, hidden_size // 4)
        self.stop_predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, reduced),
            nn.GELU(),
            nn.Linear(reduced, 1),
        )

        # audio-token vocab restriction: the ids live in the tokenizer (as
        # `<code_q_v>`-style special tokens), not config.json, so build the
        # mask here from the tokenizer at the same model path vLLM was
        # pointed at -- avoids needing to reach into engine internals after
        # `LLM(...)` construction to wire this up externally.
        self.audio_end_token_id = int(config.audio_end_token_id)
        self.audio_start_token_id = int(config.audio_start_token_id)
        self.stop_threshold = 0.5
        self.min_new_audio_tokens = 8  # matches modeling_tinyaya.py's floor -- see
        # the note in compute_logits() below on why this isn't enforced here.

        tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, trust_remote_code=True
        )
        vocab = tokenizer.get_vocab()
        audio_ids = [tid for tok, tid in vocab.items() if _AUDIO_TOKEN_RE.match(tok)]
        allowed_ids = sorted(audio_ids + [self.audio_end_token_id])
        device = self.model.embed_tokens.weight.device
        mask = torch.full((config.vocab_size,), float("-inf"), device=device)
        mask[torch.tensor(allowed_ids, device=device)] = 0.0
        self._neg_inf_mask = mask

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None or self._neg_inf_mask is None:
            return logits

        # 1. restrict to the audio-codec sub-vocabulary. Every call to
        #    compute_logits predicts the token AFTER whatever prefix vLLM is
        #    currently scoring; since every prompt for this model ends with
        #    the literal `<audio>` marker, every predicted token from that
        #    point on is expected to be an audio token by construction --
        #    so an unconditional restriction here is correct, not just an
        #    approximation (see FULL_DUPLEX_DESIGN.md / PR_NOTES.md for the
        #    duplex-model case where this assumption would need revisiting).
        logits = logits + self._neg_inf_mask

        # 2. fold the stop_predictor's decision directly into the logits:
        #    when the learned head is confident generation should end, force
        #    `</audio>` to be the only viable token (rather than a separate
        #    StoppingCriteria vLLM's interface doesn't expose in this hook).
        #    NOTE: this intentionally does NOT enforce the HF reference's
        #    `min_new_tokens=8` floor -- vLLM's compute_logits() has no
        #    per-sequence step counter available in this override point
        #    (continuous batching interleaves many sequences through one
        #    model instance; adding one would mean hooking a deeper part of
        #    the sequence-state machinery than this file touches). In
        #    practice the model is trained against this floor as a soft
        #    prior, not a hard constraint, so dropping it is a defensible v1
        #    simplification -- flagged explicitly rather than silently
        #    assumed. See PR_NOTES.md.
        stop_prob = torch.sigmoid(self.stop_predictor(hidden_states).squeeze(-1))
        force_stop = stop_prob > self.stop_threshold
        if bool(force_stop.any()):
            stop_mask = torch.full_like(logits[0], float("-inf"))
            stop_mask[self.audio_end_token_id] = 0.0
            logits[force_stop] = stop_mask

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head", "rotary_emb.inv_freq"])
        return loader.load_weights(weights)
