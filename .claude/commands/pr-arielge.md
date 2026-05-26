---
description: Stage, commit, push, and open a PR on github.ibm.com with ARIELGE as reviewer
argument-hint: "[optional: extra context for the commit/PR description]"
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git rev-parse:*), Bash(git branch:*), Bash(git push:*), Bash(git commit:*), Bash(pre-commit run:*), Bash(gh pr create:*), Bash(gh auth status:*), AskUserQuestion
---

# /pr-arielge — ship changes with ARIELGE on review

You are running on the `hf_adapters` repo (remote: `github.ibm.com:msrivats/hf_adapters`). The user wants the staged-or-unstaged work in this checkout committed, pushed, and turned into a PR with `ARIELGE` as reviewer.

Optional extra context from the user (may be empty): **$ARGUMENTS**

## Preconditions — bail out early if any fail

1. Run `git rev-parse --abbrev-ref HEAD`. If the branch is `main` or `master`, **stop** and tell the user to switch to a feature branch first. Don't try to invent one.
2. Run `git status --porcelain`. If empty, **stop** with "no changes to commit."
3. Run `gh auth status --hostname github.ibm.com`. If not authed, **stop** and tell the user to run `gh auth login --hostname github.ibm.com` in another terminal.

## Step 1 — Understand the changes

Run these in parallel: `git status`, `git diff`, `git diff --staged`, `git log --oneline -5`. Use the output to:

- Decide a concise, accurate commit subject (≤72 chars, imperative mood, follows the recent commit style you just observed).
- Draft a short body (1–3 lines, "why" not "what") if the diff warrants it. Skip the body for trivial changes.
- Note any files that look unsafe to commit (`.env`, credentials, large binaries) — if found, **stop** and ask the user before continuing.
- Fold any **$ARGUMENTS** content into the commit message and PR body when it adds context.

## Step 2 — Stage and run pre-commit

Stage every modified + untracked file the user has been working on (use explicit paths from `git status`, not `git add -A`, to avoid sweeping in stray files). Then run:

```
pre-commit run --files <those paths>
```

**If pre-commit modifies files** (e.g. ruff/black auto-fix → exit 1 with "files were modified"): re-stage the same paths and re-run pre-commit. Allow up to two retries. If it still fails after that, **stop** and surface the failing hook output — don't push.

**If pre-commit fails for a non-fixable reason** (mypy error, lint error it can't auto-fix, etc.): **stop** and report the error. Don't push half-fixed code.

## Step 3 — Commit

Use a heredoc so multiline messages format correctly:

```
git commit -m "$(cat <<'EOF'
<subject>

<optional body>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

If the commit hook fails (it shouldn't, since pre-commit already passed), **do not** `--amend`. Investigate, fix, re-stage, and create a new commit.

## Step 4 — Push

```
git push -u origin <current-branch>
```

If the branch already tracks a remote, plain `git push` is fine.

## Step 5 — Open the PR

Run in parallel with the push if you've already determined the title/body:

```
GH_HOST=github.ibm.com gh pr create \
  --base main \
  --head <current-branch> \
  --reviewer ARIELGE \
  --title "<short PR title — usually matches the commit subject>" \
  --body "$(cat <<'EOF'
## Summary
- <bullet 1>
- <bullet 2>

## Test plan
- [ ] <what you ran or want the reviewer to run>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Pull bullets from the actual diff — don't pad. If `$ARGUMENTS` provided extra context, weave it into Summary.

## Step 6 — Report back

Print the PR URL. One line is enough. Don't recap every git command you just ran.

## Hard rules

- Never use `--no-verify`, `--no-gpg-sign`, or any flag that skips hooks.
- Never `--amend` a commit that's already been pushed in a previous run.
- Never force-push.
- Never commit files that look like secrets without explicit user confirmation.
- If anything in steps 1–5 fails in a way these instructions don't cover, **stop and ask** rather than improvising a destructive recovery.
