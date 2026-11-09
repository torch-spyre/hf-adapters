#!/usr/bin/env bash
# Run-scoped cache of the torch-spyre wheel, shared through /storage-1.
#
# Why: every Spyre job spent ~80 s (p50) of its ~106 s setup compiling
# torch-spyre's 27-object C++ extension, and ccache could not help. uv builds
# torch-spyre in an isolated build env at a random path
# (~/.cache/uv/builds-v0/.tmpXXXXXX), and that path is in every compile's -I
# flags (torch headers), so no two builds share a ccache key even for the same
# commit. Each job of a run builds the same commit with the same toolchain, so
# the primer (the resolve-torch-spyre job) builds it once with the unchanged
# install path, and publishes the wheel uv built. Every other job then installs
# its deps from the same uv.lock without building torch-spyre and installs that
# wheel on top.
#
# usage (from the hf-adapters checkout, after `uv add` + `uv lock`):
#   torch_spyre_wheel_cache.sh sync <uv sync group flags...>
#
# env:
#   TORCH_SPYRE_SHA       pinned 40-char torch-spyre SHA (required)
#   TS_WHEEL_CACHE_ROLE   consume (default) | prime | off
#   TS_WHEEL_CACHE_DIR    store (default $STORAGE_1_DIR/.cache/torch-spyre-wheels)
#   TS_WHEEL_CACHE_TTL_MIN  primer prunes entries older than this (default 1440 = 1 day;
#                           an entry only serves its own run, retries and re-runs)
#   TS_WHEEL_CACHE_STATUS_FILE  if set, the final status (hit|miss|off|prime) is written here
#   TORCH_SPYRE_DIR       torch-spyre checkout (default /home/senuser/torch-spyre)
#
# Every failure on the cache path falls back to the original install,
# `uv sync --frozen --verbose --refresh <groups>`, so the worst case is
# today's behaviour. The cache never changes what gets installed: the
# non-torch-spyre packages still come from this job's uv.lock, and the cached
# wheel is used only if its key matches. The key covers the torch-spyre commit,
# its `git describe`, the env that feeds its version string and build, the
# toolchain, the Spyre stack in the image, the hf-adapters pyproject, and the
# torch that uv.lock pins.
set -uo pipefail

SCHEMA=v1
ROLE="${TS_WHEEL_CACHE_ROLE:-consume}"
STORE="${TS_WHEEL_CACHE_DIR:-${STORAGE_1_DIR:-/storage-1/hf-adapters}/.cache/torch-spyre-wheels}"
TTL_MIN="${TS_WHEEL_CACHE_TTL_MIN:-1440}"
TSDIR="${TORCH_SPYRE_DIR:-/home/senuser/torch-spyre}"
SHA="${TORCH_SPYRE_SHA:-}"
PY="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}/bin/python"

now() { date +%s.%N; }
# timing <phase> <t0> <status>
timing() {
  printf 'SETUP-TIMING phase=%s seconds=%.1f status=%s\n' "$1" "$(echo "$(now) $2" | awk '{print $1 - $2}')" "$3"
}
log() { echo "torch-spyre wheel cache: $*"; }
set_status() { [[ -n "${TS_WHEEL_CACHE_STATUS_FILE:-}" ]] && echo "$1" > "$TS_WHEEL_CACHE_STATUS_FILE"; return 0; }

normal_sync() {  # the original install; its exit code is the step's
  local t0 rc
  t0=$(now)
  uv sync --frozen --verbose --refresh "$@"
  rc=$?
  timing final-sync "$t0" "$STATUS"
  return $rc
}

# Everything that decides what the torch-spyre build produces, one per line.
key_inputs() {
  local v
  echo "schema=$SCHEMA"
  echo "sha=$SHA"
  echo "head=$(git -C "$TSDIR" rev-parse HEAD 2>&1)"
  # setuptools_scm version = latest tag + distance + dirty; tags are fetched by the pin step.
  echo "describe=$(git -C "$TSDIR" describe --tags --long --always --dirty --abbrev=40 2>&1)"
  echo "tracked-changes=$(git -C "$TSDIR" status --porcelain --untracked-files=no 2>&1 | sha256sum | cut -c1-16)"
  # _versioning.py's local scheme, and setup.py's build switches.
  for v in BUILD_NUMBER BRANCH_NAME GIT_BRANCH GITHUB_HEAD_REF GITHUB_REF_NAME GITHUB_RUN_NUMBER \
           CI_PIPELINE_IID CI_COMMIT_REF_NAME \
           USE_SPYRE_PROFILER USE_SPYRE_CCL TORCH_SPYRE_DEBUG SHARED_DEPS_DIR \
           CMAKE_INCLUDE_PATH CMAKE_LIBRARY_PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH \
           CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS; do
    echo "env.$v=${!v-<unset>}"
  done
  env | grep -E '^[A-Z_]+_INSTALL_DIR=' | LC_ALL=C sort
  echo "c++=$(command -v c++) $(c++ --version 2>&1 | head -1)"
  echo "python=$(python3 -VV 2>&1)"
  echo "uv=$(uv --version 2>&1)"
  if command -v rpm >/dev/null; then echo "rpms=$(rpm -qa 2>/dev/null | LC_ALL=C sort | sha256sum | cut -c1-16)"; fi
  echo "image-artifact-id=$(cat /home/senuser/spyre_artifact_id.txt 2>/dev/null || echo '<none>')"
  # Headers and libraries of the Spyre stack the extension compiles and links against.
  echo "spyre-stack=$(find /opt/ibm/spyre -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.inc' \
      -o -name '*.def' -o -name '*.so' -o -name '*.so.*' -o -name '*.a' \) -printf '%p %s %T@\n' 2>/dev/null \
      | LC_ALL=C sort | sha256sum | cut -c1-16)"
  echo "pyproject=$(sha256sum pyproject.toml 2>&1 | cut -c1-16)"
  # The torch the lock pins (torch-spyre is built against it and must match the venv's).
  echo "lock-torch=$(python3 -I - <<'EOF' 2>&1
import tomllib
lock = tomllib.load(open("uv.lock", "rb"))
print(sorted((p["version"], str(p.get("source"))) for p in lock.get("package", []) if p["name"] == "torch"))
EOF
)"
}

meta_get() {  # meta_get <meta.json> <field>
  python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

consume() {
  local t0 entry wheel tmpd local_wheel want_sha ver inst
  t0=$(now)
  entry="$STORE/$KEY"
  if [[ ! -f "$entry/meta.json" ]]; then
    log "miss: no entry $entry"
    timing ts-wheel-fetch "$t0" miss
    return 1
  fi
  wheel="$entry/$(meta_get "$entry/meta.json" wheel 2>/dev/null)" || wheel=""
  want_sha="$(meta_get "$entry/meta.json" wheel_sha256 2>/dev/null)" || want_sha=""
  if [[ "$(meta_get "$entry/meta.json" torch_spyre_sha 2>/dev/null)" != "$SHA" || ! -f "$wheel" || -z "$want_sha" \
        || "$(basename "$wheel")" != *".g${SHA:0:8}"* ]]; then
    log "miss: entry $entry is not a wheel of ${SHA}"
    timing ts-wheel-fetch "$t0" miss
    return 1
  fi
  # Copy to local disk and check the hash, so a partial or corrupt read is never installed.
  tmpd="$(mktemp -d)"
  local_wheel="$tmpd/$(basename "$wheel")"
  if ! cp "$wheel" "$local_wheel" || [[ "$(sha256sum "$local_wheel" | cut -d' ' -f1)" != "$want_sha" ]]; then
    log "miss: copy of $wheel failed or sha256 mismatch"
    timing ts-wheel-fetch "$t0" miss
    return 1
  fi
  ver="$(meta_get "$entry/meta.json" version)"
  log "hit: $(basename "$wheel") (primed by run $(meta_get "$entry/meta.json" run_id) job $(meta_get "$entry/meta.json" job))"
  timing ts-wheel-fetch "$t0" hit

  # Everything in uv.lock except torch-spyre -- same command as the original, so the
  # same package set and versions -- then the cached wheel on top.
  t0=$(now)
  if ! uv sync --frozen --verbose --refresh "$@" --no-install-package torch-spyre; then
    timing final-sync "$t0" hit-failed
    log "sync without torch-spyre failed; falling back"
    return 1
  fi
  timing final-sync "$t0" hit
  t0=$(now)
  if ! uv pip install --python "$PY" --no-deps --reinstall "$local_wheel"; then
    timing ts-wheel-install "$t0" failed
    log "installing the cached wheel failed; falling back"
    return 1
  fi
  inst="$("$PY" -I -c 'import importlib.metadata as m; print(m.version("torch-spyre"))' 2>&1)"
  if [[ "$inst" != "$ver" ]]; then
    timing ts-wheel-install "$t0" failed
    log "installed torch-spyre is '$inst', expected '$ver'; falling back"
    return 1
  fi
  # Record the install as coming from the source tree uv.lock names (what the
  # original `uv sync` writes), not from the temp wheel path, so `uv pip freeze`
  # and anything reading PEP 610 metadata see the same thing as without the cache.
  "$PY" -I - <<'EOF' || log "could not rewrite direct_url.json (cosmetic)"
import json, pathlib, tomllib, importlib.metadata as m
lock = tomllib.load(open("uv.lock", "rb"))
src = next(p["source"]["directory"] for p in lock["package"] if p["name"] == "torch-spyre")
dist = m.distribution("torch-spyre")
p = pathlib.Path(dist._path) / "direct_url.json"
p.write_text(json.dumps({"url": pathlib.Path(src).resolve().as_uri(), "dir_info": {"editable": False}},
                        separators=(",", ":")))
print(f"direct_url.json -> {p.read_text()}")
EOF
  timing ts-wheel-install "$t0" hit
  rm -f "$local_wheel"; rmdir "$tmpd" 2>/dev/null
  log "installed torch-spyre $inst from the run's cached wheel"
  return 0
}

prune() {  # entries older than TTL; files are removed one by one (no recursive delete)
  local d
  while IFS= read -r d; do
    rm -f "$d"/* 2>/dev/null && rmdir "$d" 2>/dev/null && log "pruned $(basename "$d")"
  done < <(find "$STORE" -mindepth 1 -maxdepth 1 -type d -mmin "+$TTL_MIN" 2>/dev/null)
}

publish() {  # publish <marker>: the wheel the preceding sync built, atomically
  local marker="$1" t0 cache wheels wheel tmpd sum ver
  t0=$(now)
  cache="$(uv cache dir 2>/dev/null)"
  # uv keeps the wheel it built for a local source tree in its cache
  # (sdists-v<N>/path/<source hash>/<revision>/); the marker isolates this sync's build.
  mapfile -t wheels < <(find "$cache"/sdists-v* -name "torch_spyre-*.whl" -type f -newer "$marker" 2>/dev/null)
  if (( ${#wheels[@]} != 1 )); then
    log "publish skipped: expected 1 freshly built torch_spyre wheel in $cache, found ${#wheels[@]}: ${wheels[*]:-}"
    timing ts-wheel-publish "$t0" skip
    return 0
  fi
  wheel="${wheels[0]}"
  if [[ "$(basename "$wheel")" != *".g${SHA:0:8}"* ]]; then
    log "publish skipped: $(basename "$wheel") is not built from ${SHA:0:8}"
    timing ts-wheel-publish "$t0" skip
    return 0
  fi
  ver="$("$PY" -I -c 'import importlib.metadata as m; print(m.version("torch-spyre"))' 2>&1)"
  if [[ "$(basename "$wheel")" != "torch_spyre-${ver}-"* ]]; then
    log "publish skipped: installed torch-spyre '$ver' does not match $(basename "$wheel")"
    timing ts-wheel-publish "$t0" skip
    return 0
  fi
  if [[ -f "$STORE/$KEY/meta.json" ]]; then
    log "publish skipped: $KEY already published"
    timing ts-wheel-publish "$t0" exists
    return 0
  fi
  mkdir -p "$STORE" || { timing ts-wheel-publish "$t0" fail; return 0; }
  # Write into a temp dir and rename it into place: readers see all or nothing.
  tmpd="$STORE/.tmp-$KEY-${GITHUB_RUN_ID:-0}-${GITHUB_RUN_ATTEMPT:-0}-$$"
  mkdir "$tmpd" || { timing ts-wheel-publish "$t0" fail; return 0; }
  sum="$(sha256sum "$wheel" | cut -d' ' -f1)"
  if cp "$wheel" "$tmpd/" && cp "$KEYFILE" "$tmpd/key-inputs.txt" && chmod a+r "$tmpd"/* \
     && python3 -I - "$tmpd/meta.json" <<EOF
import json, sys
json.dump({"schema": "$SCHEMA", "key": "$KEY", "torch_spyre_sha": "$SHA", "version": "$ver",
           "wheel": "$(basename "$wheel")", "wheel_sha256": "$sum",
           "run_id": "${GITHUB_RUN_ID:-}", "run_attempt": "${GITHUB_RUN_ATTEMPT:-}", "job": "${GITHUB_JOB:-}"},
          open(sys.argv[1], "w"), indent=1)
EOF
  then
    if mv -T "$tmpd" "$STORE/$KEY" 2>/dev/null; then
      log "published $(basename "$wheel") as $STORE/$KEY"
      timing ts-wheel-publish "$t0" ok
      prune
      return 0
    fi
    log "publish lost a race: $KEY appeared meanwhile"
  fi
  rm -f "$tmpd"/* 2>/dev/null; rmdir "$tmpd" 2>/dev/null
  timing ts-wheel-publish "$t0" fail
  return 0
}

[[ "${1:-}" == sync ]] || { echo "usage: $0 sync <uv sync group flags...>" >&2; exit 2; }
shift

if [[ "$ROLE" == off || ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  STATUS=off; set_status off
  normal_sync "$@"; exit $?
fi

t0=$(now)
KEYFILE="$(mktemp)"
key_inputs > "$KEYFILE"
KEY="$(sha256sum "$KEYFILE" | cut -c1-32)"
echo "::group::torch-spyre wheel cache key ${KEY} (role=${ROLE})"; cat "$KEYFILE"; echo "::endgroup::"
timing ts-wheel-key "$t0" "$ROLE"

case "$ROLE" in
  prime)
    STATUS=prime; set_status prime
    marker="$(mktemp)"
    normal_sync "$@" || exit $?
    publish "$marker"
    rm -f "$marker"
    ;;
  *)
    if consume "$@"; then
      STATUS=hit; set_status hit
    else
      STATUS=miss; set_status miss
      normal_sync "$@" || exit $?
    fi
    ;;
esac
rm -f "$KEYFILE"
exit 0
