from enum import Enum


class ModelType(str, Enum):
    GENERATIVE = "generative"
    EMBEDDING = "embedding"
