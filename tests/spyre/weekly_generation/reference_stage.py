"""Which stock-reference step a vision verifier is in.

The VLM and tower verifiers report the stage through their ``on_reference_stage``
callback, and ``weekly_sub_process`` uses the last one reported to label a failure
that happens before the reference is ready:

* ``"load"``: the stock checkpoint is loading -> ``cpu_load_failed``.
* ``"forward"``: inputs are being prepared or the stock forward is running ->
  ``cpu_generate_failed``.

A leaf module with no imports from this repo, so lightweight importers (the weekly
driver and the offline tests) do not pull in torch or the Spyre test modules.
"""

from typing import Literal

ReferenceStage = Literal["load", "forward"]
