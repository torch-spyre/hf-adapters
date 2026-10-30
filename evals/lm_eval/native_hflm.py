"""Stock Transformers loading for text benchmarks."""

from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForImageTextToText


class NativeHFLM(HFLM):
    def _create_model(self, *args, **kwargs):
        # Ministral 3 checkpoints use the multimodal Mistral3 wrapper, which
        # AutoModelForCausalLM does not register. Its stock conditional-generation
        # model accepts text-only inputs and supports HFLM's causal scoring and
        # generation unchanged. Keep this override per instance, not global.
        if self.config.model_type == "mistral3" and self.backend == "causal":
            self.AUTO_MODEL_CLASS = AutoModelForImageTextToText
        return super()._create_model(*args, **kwargs)
