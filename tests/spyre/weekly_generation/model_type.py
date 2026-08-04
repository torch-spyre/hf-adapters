"""The two kinds of model the weekly scan evaluates.

``generative`` and ``embedding`` each name a whole bundle of per-kind choices
that used to travel as a bare string: which catalog is fetched, which ClickHouse
table the sink binds, which model class ``_load_on_cpu`` instantiates, which
verification pipeline ``eval_model`` runs, and how many models one worker
process handles.

``StrEnum``, not ``(str, Enum)``. These members reach shard filenames, the GHA
matrix, and the log lines an operator reads, so a member must *format* as its
value. ``class ModelType(str, Enum)`` inherits ``Enum.__str__``/``__format__``,
making ``f"{ModelType.GENERATIVE}"`` render ``"ModelType.GENERATIVE"`` — which
is how shard files briefly came out named
``ModelType.GENERATIVE-x1-shard-000.json``. ``StrEnum`` (3.11+, and this project
requires >=3.11) is the variant whose ``__str__`` is ``str.__str__``.
"""

from enum import StrEnum


class ModelType(StrEnum):
    GENERATIVE = "generative"
    EMBEDDING = "embedding"
