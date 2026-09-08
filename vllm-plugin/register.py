"""Register TinyAyaForCausalLM with vLLM. Import and call `register()` before
constructing an `LLM`/`AsyncLLMEngine` pointed at a rumik-oss-1[-nvfp4-vllm]
checkpoint."""


def register():
    from tinyaya_vllm import TinyAyaForCausalLM
    from vllm import ModelRegistry

    ModelRegistry.register_model("TinyAyaForCausalLM", TinyAyaForCausalLM)
