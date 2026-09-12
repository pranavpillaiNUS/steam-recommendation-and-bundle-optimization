# Repository conventions

This note records how I keep the repository consistent between sessions. It is a working document
for me and for Dr Li, not an invitation for outside changes.

## Evidence discipline

1. Raw data, user identifiers, review text, per-user metrics, fitted user factors, and anything
   under the ignored protected paths stay out of version control.
2. I never overwrite a frozen cycle artifact. When a scientific input, runner, configuration, or
   result has to change, I open a new cycle or write a clearly labeled post-freeze document.
3. Claims stay inside the boundaries set in [stage1_model_card.md](stage1_model_card.md). Scores
   are not willingness to pay, monetary utility, purchase probability, or revenue.
4. Behavioral changes come with focused tests, and generated outputs stay deterministic.
5. Stage 1 is not retuned after I have seen any Stage 2 output. Doing that would turn a
   prospective comparison into a selected one.
6. `notes/preference_model_specification.md` and `notes/stage1_v2_mathematical_appendix.md` are
   hash-bound into the frozen estimator specification and evidence manifests. Even a wording
   change there fails the test suite and reopens the cycle, so they stay as they are until a
   new cycle starts.

## Checks before committing

```text
python -m pip install -r requirements-frozen.txt
python -m pip check
python -m pytest -p no:cacheprovider --strict-markers -q
python -m src.stage1_public_verify
git diff --check
```

The public verifier does not modify anything. The full `python -m src.stage1_pipeline` command
needs the ignored raw and protected artifacts, so it cannot stand in for these checks on a clean
clone.

## Commit notes

A commit message says which research claim or engineering contract it touches, which checks I ran,
and whether any manifest or artifact hash changed. Screenshots, logs, fixtures, and notebook
outputs must never contain private records.
