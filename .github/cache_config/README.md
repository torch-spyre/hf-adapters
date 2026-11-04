# Shared cache maintenance

## Hugging Face models and image fixtures

The `add_or_remove_artifacts_from_cache` workflow runs nightly at 05:00 UTC
and supports manual runs. It warms the shared `HF_HOME` at
`/storage-1/hf-adapters/.cache/huggingface` before offline CI consumes it.

`manage_cache.py` discovers model repositories from `tests/model_registry.py`.
The `models` list in `hf_models_and_datasets.yaml` adds extra model repositories.
The separate `datasets` list declares dataset repositories and exact filenames;
it currently includes the shared cat image and SigLIP's dog image. These files
use the default Hub revision, matching the test helpers. Each file is downloaded
and then resolved again with `local_files_only=True`. A failed download or offline
lookup fails the cache job after the other artifacts have been attempted.

When adding a fixture, run this workflow manually from the reviewed branch
containing the updated script and configuration, and wait for success before
rerunning offline CI. Merely adding an image to a test or fixture registry does
not populate the shared cache. No Spyre card is needed for the cache job.

## VLM reference cache maintenance

The Prune VLM reference cache workflow runs weekly on a runner with the shared
storage mounted and zero Spyre cards. It removes completed CPU references from
/storage-1/hf-adapters/.cache/vlm-references when both their last access and last
modification are more than 30 days old.

Only regular files named vlm-reference-<64 lowercase hex characters>.pt directly
inside the cache directory are eligible. Temporary files, symlinks, directories,
and unrelated files are retained. Cleanup uses filesystem metadata without
loading tensors or models. A missing directory is a successful no-op; permission
errors are reported and fail the maintenance job after processing other entries.

Access timestamps preserve frequently read entries on filesystems that update
them. On mounts with access-time updates disabled, retention follows modification
age, so an older active reference may be regenerated after cleanup.

Manual workflow runs default to a preview and accept a different retention period.
Clear the dry_run input to delete the listed candidates. The script also defaults
to a preview:

    python3 .github/cache_config/prune_vlm_reference_cache.py \
        --cache-dir /path/to/vlm-references --max-age-days 30

Add --delete to remove expired entries. The script prints each candidate and the
total file count and bytes. Cache readers can recompute an entry that disappears
during cleanup.

Run the maintenance tests without model dependencies:

    python3 -m unittest discover -s .github/cache_config \
        -p 'test_prune_vlm_reference_cache.py' -v
