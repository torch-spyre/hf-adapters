# DO NOT MERGE — Spyre check-gate validation

Throwaway PR for validating Spyre CI check behaviour end to end. It carries no functional
change and must never be merged.

Used to verify, against a real PR:

- the aggregate `Spyre Test` check is posted
- the gateable `Spyre Test (<arch>)` context appears on the PR head
- a narrowed run (`archs=` / `preset=`) reports `action_required` for an arch it did not
  exercise, rather than passing
- `Test-With:` companion resolution, when a counterpart PR is linked

Delete the branch once the rollout is done.
