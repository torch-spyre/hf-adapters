from tests.spyre.weekly_generation.model_type import ModelType
from utils.fetch_top_embedding_models import fetch_top_embedding_models
from utils.fetch_top_generative_models import fetch_top_generative_models
from utils.utilities import ts

all_fetchers = {
    ModelType.GENERATIVE: fetch_top_generative_models,
    ModelType.EMBEDDING: fetch_top_embedding_models,
}

def fetch(model_type: ModelType, top_k: int) -> list[dict]:
    models: list[dict] = all_fetchers[model_type](limit=top_k)

    # model_info is a live huggingface_hub.ModelInfo object attached by
    # build_catalog — not JSON-serializable, and no longer needed since
    # is_moe is precomputed onto each model (see utils/hf_model_catalog.py).
    for model in models:
        model.pop("model_info", None)

    print(f"{ts()} Fetched {len(models)} {model_type.value} model(s).")

    return models
