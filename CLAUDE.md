# Instructions for Claude Code

## Git commits

Never add a "Co-Authored-By: Claude ..." / "Generated with Claude Code" trailer
to commits in this repo. The owner does not want an AI assistant showing up as
a GitHub contributor here.

This is enforced by a `commit-msg` hook at `.githooks/commit-msg`, activated via
`core.hooksPath = .githooks` in this repo's local git config. That local config
is not tracked by git, so if the repo is freshly cloned, re-run:

```
git config core.hooksPath .githooks
```

Don't rely on the hook alone — omit the trailer proactively when committing.
