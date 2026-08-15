# Contributing

This is a research repository with hash-bound evidence. Small fixes are welcome, but changes must
preserve the boundary between frozen results and prospective work.

## Before opening a change

1. Do not commit raw data, user identifiers, reviews, per-user metrics, fitted user factors, or
   files under ignored protected paths.
2. Do not overwrite a frozen cycle artifact. Create a new cycle or a clearly labeled post-freeze
   document when a scientific input, runner, configuration, or result must change.
3. Keep claims within the model card boundaries. Scores are not willingness to pay, monetary
   utility, purchase probability, or revenue.
4. Add focused tests for behavioral changes and keep generated outputs deterministic.

## Local checks

```text
python -m pip install -r requirements-frozen.txt
python -m pip check
python -m pytest -p no:cacheprovider --strict-markers -q
python -m src.stage1_public_verify
git diff --check
```

The public verifier is non-mutating. The full `src.stage1_pipeline` command requires the ignored
raw/protected artifact set and must not be used as a public-clone smoke test.

## Pull requests

Describe the research claim or engineering contract affected, the tests run, and whether any
manifest or artifact hash changes. Never include private records in screenshots, logs, fixtures,
or notebook outputs.

Unless explicitly agreed otherwise, contributions to original project code and documentation are
accepted under the repository's [MIT License](LICENSE). Do not contribute material whose license is
incompatible or whose redistribution rights are unclear.
