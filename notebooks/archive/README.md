# Archived notebooks retained in place

Notebooks 06 and 10 are historical records of the pre-pivot CMM/BSP bundle-size-pricing mechanism:

- `notebooks/06_bsp_synthetic_validation.ipynb`
- `notebooks/10_bundle_size_pricing.ipynb`

They intentionally remain at their original paths and notebook numbers. Moving or renumbering them would break existing links, obscure the provenance of saved tables and figures, and make the mechanism pivot harder to audit. This directory contains only the archive index; the notebooks themselves are not duplicated or moved.

The selected archive policy is preservation in repository history rather than a live rerun
requirement. Commit `3918b2b5afe88b88e8b8a6ce57533cc14d66d5a3` is the exact comparison
baseline for notebooks 06 and 10: relative to that commit, their current code cells and outputs are
unchanged and only explanatory Markdown errata were added. The older tag
`pre-pivot-cmm-2026-07-11` points to an earlier CMM milestone and is not claimed to contain the final
pre-freeze notebook state. No new tag or commit was created during this cleanup. The legacy
dependency path is outside the live reproducibility gate.

Both notebooks are outside the live project DAG. CMM/BSP offers a size-price menu under which a customer may construct an arbitrary set of a chosen size. That mechanism does not represent the fixed seller-curated bundles in the Steam setting. Their code, outputs, and figures are preserved as evidence of work completed before the pivot, not as current inputs, validations, or empirical conclusions.

The live primary optimization model is CP-anchored fixed-bundle $\mathrm{SBA}^{CP}$: component prices are estimated first and frozen, then one fixed bundle composition and normalized bundle price are chosen while the components remain available separately. SBR uses a different menu in which bundled components are unavailable separately and is retained only as a theoretical benchmark. CMM or SBR results and guarantees do not transfer to SBA without independent proofs.

Historical fields named `valuation`, `price`, or `profit` in notebooks 06 and 10 do not identify willingness to pay, purchase probabilities, money, or Steam revenue. At most, the saved arrays and objectives can be read as normalized pseudo-utility calculations inside retired modeling scenarios. Uncertified numerical-search outputs are best solutions found, not optima.

Do not consume notebook 06 or 10 artifacts in the live pipeline. Current work must use frozen Stage 1 score models, predeclared pseudo-utility scenarios, frozen candidate pools, transparent SBA exact oracles, independently benchmarked heuristics, and assessment-user evaluation without reoptimization.
