# Native reference results

Run `python -m evals.lm_eval.run_benchmarks --backend hf --write-references`
in a CUDA Transformers environment, then commit the generated model JSON files
here. See [the evaluation guide](../README.md) for setup and comparison rules.

No reference scores are fabricated or inferred from Spyre runs. Until a native
reference exists, CI reports the model's scores with `missing_reference`.
