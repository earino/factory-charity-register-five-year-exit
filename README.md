---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-6e90da65ff14dfeb**

[Dataset website](https://earino.github.io/factory-charity-register-five-year-exit/versions/v-6e90da65ff14dfeb/) · [GitHub release](https://github.com/earino/factory-charity-register-five-year-exit/releases/tag/v-6e90da65ff14dfeb) · [Hugging Face](https://huggingface.co/datasets/earino/factory-charity-register-five-year-exit/tree/v-6e90da65ff14dfeb) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: b059d2169e6f3207400d1b7a9ae3c6c27a32fa765ef5f341f529708c25b01ea2
<!-- /dataset-factory:links -->

# Will a newly registered charity leave the Charity Commission register within five years?

Will a charity newly entered on the Register of Charities for England and Wales be removed from the register within five years of its registration date?

Derived from Charity Commission for England and Wales. This is not its official product.

Prediction time: date_of_registration, the day the charity is entered on the register. Only attributes present in the current daily extract are treated as registration-time descriptors; classification, area, governing-document and objects fields are not time-stamped, so their as-of-registration availability is assumed rather than verified.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6710803183963197,
    "brier": 0.05837994152498414
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.061308903815722494
  },
  "calendar_identity": {
    "auc": 0.4876390876011456,
    "brier": 0.061366288345976024
  }
}
```

## Limitations

- Removal is not the same as failure: it includes mergers, asset transfers, dissolution and removal for non-filing, so the label is a register-exit event rather than an insolvency or effectiveness measure.
- The bulk extract is a current-vintage snapshot: classification, area, activities and governing-document fields carry no as-of-registration timestamp, so they are treated as registration-time descriptors but may reflect later edits; this does not establish historical operational availability.
- Charities removed after the five-year anniversary are coded negative, so late exits are right-censored by design.
- The register covers England and Wales only; Scotland (OSCR) and Northern Ireland (CCNI) are separate regulators and are excluded.
- The register includes personal data (trustee names, contact details) that the OGL does not cover; the derived release must exclude those fields and use organisation-level attributes only.
- Legal-form conversions after registration (for example to a CIO) can change the current charity_type, creating a small vintage mismatch with the decision instant.
- The removal date is not the date the charity ceased to exist or ceased to operate, so the label should not be read as an organisational death date.
- Removal is a register-exit event, not insolvency, failure or organisational death; it includes mergers, asset transfers, dissolution and removal for non-filing.
- The bulk extract is a current-vintage snapshot; classification, area, governing-document and objects fields carry no as-of-registration timestamp, so they are assumed registration-time descriptors and may reflect later edits.
- Charities removed after the fifth anniversary are coded negative, so late exits are right-censored by design.
- England and Wales only; Scotland and Northern Ireland are separate regulators.
- Personal data in the register is excluded; only organisation-level attributes are shipped.
- Legal-form conversions after registration can create a vintage mismatch.
- date_of_removal is not the date the charity ceased to exist or ceased to operate.
- The shipped date_of_registration is the prediction-time anchor and part of the cohort definition, not an answer source; the answer sources (date_of_removal and charity_registration_status) are never shipped.
- Actual column names and status spellings were not re-read during this write-only construction; the builder resolves documented and plausible aliases and raises with observed headers and archive member if a field or join is unresolved.
- No persistence reference is available because each organisation appears once in the cohort.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
