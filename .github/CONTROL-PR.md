# Control PR (DO NOT MERGE)

This branch exists only to answer one question: does the hf-adapters test
suite pass on the **legacy** CI runner image, with no source changes?

- Base: `main`, unchanged. This file is the only diff, so the branch stays a
  faithful baseline. It forces a real `regression` run (an empty commit can
  resolve oddly).
- Runners: the default `image_torch_spyre` (legacy `2.0/ci` image). No image
  redirect, no venv changes, no test edits.
- Pairs with the `/next`-image shadow PR (same suite, `/next` runner image).
  Compare the two: if the shadow run is red while this control is green, the
  `/next` image is the cause, not the tests.
- Fixes belong in dedicated fix PRs, never here.

Delete this file before ever considering a merge; this branch is a throwaway.
