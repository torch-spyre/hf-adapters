---
name: Spyre Pod Access
description: "Spyre pod environment details, rebuild steps, test commands, and known device issues (basic access commands are in CLAUDE.md)"
type: reference
---

## Pod Details

- Python: 3.12, PyTorch 2.10, torch-spyre 0.0.1 (editable)
- venv: `$HOME/dt-inductor/.venv`
- torch-spyre source: `$HOME/dt-inductor/torch-spyre`

## Rebuild torch-spyre

```bash
kubectl exec -n $NS $POD -- bash -lc \
  "$HOME/dt-inductor/torch-spyre-docs/scripts/build-torch-spyre.sh"
```

## Running hf_adapters tests on pod

```bash
# Copy entire project to pod
kubectl exec -n $NS $POD -- bash -lc "rm -rf $HOME/hf_adapters"
kubectl cp . $NS/$POD:$HOME/hf_adapters

# Run CPU accuracy test
kubectl exec -n $NS $POD -- bash -lc \
  "cd $HOME/hf_adapters && python3 tests/test_adapter_cpu_accuracy.py granite2b"

# Run Spyre smoke test (needs venv with torch_spyre)
kubectl exec -n $NS $POD -- bash -lc \
  "source $HOME/dt-inductor/.venv/bin/activate && \
   export PYTHONPATH=$HOME/dt-inductor/torch-spyre:$HOME/hf_adapters:\$PYTHONPATH && \
   cd $HOME/hf_adapters && python3 tests/test_e2e_smoke_spyre.py granite"
```

## Known Device Issues

- `int64` auto-downcasted to `int32` with a warning
- `aten.embedding.default` falls back to CPU (automatic, transparent)
- `torch.ops.spyre.full` falls back to CPU
