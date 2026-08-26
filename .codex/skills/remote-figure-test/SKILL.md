---
name: remote-figure-test
description: Run user-requested drone-emission figure work through one persistent hku-sto SSH TTY and retrieve the requested artifacts with rsync. Use for remote figure generation, testing, or fetching after the intended local commit has been pushed.
---

# Remote Figure Test

Coordinate remote figure work and artifact retrieval. This skill governs the
remote session and transfer process; it does not prescribe validation commands,
artifact counts, or content checks.

## Defaults

- Local repo: `/Users/linjingxian/Code/drone-emission`
- Remote host: `hku-sto`
- Remote repo: `/home/jingxian/code/drone-emission`
- Default local review root: `figures/review/`

## Required workflow

1. Check local Git status and preserve unrelated local files. When the remote
   task depends on local changes, ensure the intended commit has been committed
   and pushed, then pull that commit remotely. Do not infer permission to commit
   unrelated changes.
2. Use exactly one `gpt-5.5` sub-agent as the exclusive owner of the remote
   workflow. Tell it not to edit or revert source, not to clean unrelated files,
   not to run local Python, and to preserve unrelated local and remote changes.
   The primary agent must not open another remote connection or run remote
   commands in parallel.
3. The sub-agent must open one background interactive SSH TTY and route every
   remote status, pull, user-requested command, artifact-location, and
   verification command through that same session. Do not reconnect for
   individual commands or launch duplicate remote jobs.
4. Keep the TTY alive until the requested artifacts have been received and the
   local transfer has been checked. If the TTY is lost, the same sub-agent may
   reconnect at most three times. Before resuming, confirm the old connection
   and remote job state so no duplicate session or job is created.
5. Fetch only the artifacts requested by the user. Prefer `rsync` over a
   separate transfer connection, use partial-transfer support, and never use
   `--delete`. Retry a failed `rsync` transfer at most three times; use `scp` as
   the fallback only when `rsync` is unavailable or still fails.
6. Use a user-specified new local destination when provided. Otherwise create a
   new batch under `figures/review/<purpose>_YYYYMMDD/`, using the local
   `Asia/Shanghai` date and a short English kebab-case purpose. Never overwrite
   or mix with an existing batch; append `_02`, `_03`, and so on when needed.
7. Check transfer completeness only: require a successful transfer exit, local
   file existence, and matching remote/local file lists and byte sizes. Do not
   impose artifact-count or content-validation requirements.
8. Exit the persistent TTY only after the transfer check succeeds or the retry
   limit is exhausted. Report the commit, host and remote source, local batch
   directory, fetched files and sizes, transfer method, retries, and warnings.
