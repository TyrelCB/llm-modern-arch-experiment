# Project memory instructions

This repository is an evidence-driven small-language-model optimization testbed.
The original DeepSeek-V4 comparison is retained as historical Phase I; current
work incorporates new findings and promotes only changes that improve capability
or efficiency under controlled measurement.

Before changing experiments, architecture, training policy, or result claims,
read these sources in order:

1. [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md) — current state, champion, active work,
   known measurement debt, and priorities.
2. [`docs/decisions.md`](docs/decisions.md) — append-only decision ledger. Newer
   entries supersede older ones; historical result notes do not override it.
3. [`docs/architecture.md`](docs/architecture.md) — canonical human-readable model
   and training dataflow, with decision links.
4. [`docs/architecture.json`](docs/architecture.json) — machine-readable contract
   for spin-off projects.

## Memory maintenance

- Add a decision entry whenever an optimization is adopted, rejected, deferred,
  materially re-scoped, or changes the experiment contract.
- Never rewrite the substance of an old decision. Append a new entry with a
  `Supersedes` field and update the ledger index.
- Keep `PROJECT_MEMORY.md`, `docs/architecture.md`, and
  `docs/architecture.json` synchronized whenever the current champion,
  architecture, training recipe, evaluation policy, or active priority changes.
- Run `PYTHONPATH=src python3 -m pytest tests/test_architecture_manifest.py -q`
  after changing the architecture manifest, profiles, decision IDs, or source map.
- Every architecture component in `docs/architecture.md` must link to its decision
  entry and implementation source. Keep the JSON decision IDs in sync.
- Treat historical protocols and result notes as snapshots of what was believed
  when they were written. Add a status notice when a living decision supersedes
  one; do not silently edit historical evidence.
- Record facts and measured results separately from hypotheses and planned work.
  Use the statuses `accepted`, `provisional`, `planned`, `deferred`, `rejected`,
  and `superseded` consistently.
- Include an `as_of` date in living memory. Ephemeral run state must point to its
  authoritative log rather than pretending a copied token count stays current.

## Experiment contract

- Use seed `2026` and the canonical corpus permutation for pretraining screens;
  use the established stage seed for checkpoint-forked SFT comparisons.
- Multiple-seed replication is not a routine promotion gate. Prefer paired forks,
  sustained learning-curve effects, a meaningful effect floor, and confirmation
  at another scale or token budget. Extra seeds are optional for publication or
  genuinely borderline high-impact decisions.
- Semantics-preserving systems changes require output, loss, gradient, optimizer-
  step, and resume parity before performance claims.
- Learning changes require a fixed compute contract, trajectory-level capability
  evidence, and a sealed confirmation set once that split exists.
- Do not compare SFT data arms by example count alone; match supervised tokens and
  report wall time.
- Report total parameters, non-embedding compute-bearing parameters, and estimated
  FLOPs. “Body parameters” may be retained as a historical label but are not the
  sole efficiency axis because the untied LM head performs a large matmul.
- Preserve negative results and scope them to the tested implementation, hardware,
  model size, optimizer, and budget.

## Working-tree safety

Run artifacts and live jobs commonly coexist with source work. Check `git status`
and active training processes before acting. Do not edit, stage, stop, prune, or
otherwise disturb unrelated run artifacts or user changes. Commit only explicitly
scoped files.
