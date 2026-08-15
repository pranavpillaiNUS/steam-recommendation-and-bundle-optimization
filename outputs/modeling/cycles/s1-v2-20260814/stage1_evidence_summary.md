# Stage 1 evidence summary

Cycle: `s1-v2-20260814`

Stage 1 is complete under the prospective steam_id-based cycle. All ranking results are held-out ownership reconstruction metrics, not utility or monetary estimates.

## Selected configurations

- `feature_sum_bpr_identity`: `bpr__k064__reg0p0001__lr0p05`
- `feature_sum_bpr_identity_genre`: `genre__bpr__k064__reg0p0001__lr0p05`
- `implicit_als`: `als__k064__reg0p05__ao20__ownership_only`

## Admission and design-test result

Admitted families: `implicit_als`.
- `feature_sum_bpr_identity` mean design-test NDCG@20: 0.133170
- `feature_sum_bpr_identity_genre` mean design-test NDCG@20: 0.077396
- `implicit_als` mean design-test NDCG@20: 0.206089
- `popularity` mean design-test NDCG@20: 0.135350

The admission set was selected and hashed from validation before the design-test coordinates were opened. Design-test and pseudo-cold results did not replace the selected configurations.

## Claims and nonclaims

- Warm ranking evidence supports only predictive ownership reconstruction on this snapshot.
- The genre comparison is a controlled predictive ablation, not a causal genre effect.
- Pseudo-cold scores suppress collaborative identity and bias for cohort items, but do not prove general cold-start performance.
- Pseudo-utility scenarios are deterministic nonnegative transformations. They do not identify willingness to pay, money, or interpersonal welfare.
- No Stage 2 bundle objective or bundle outcome was used anywhere in Stage 1 selection or Gate 2 freezing.
