# Forecasting npm advisory publications

One prediction describes one npm package at the publication time of its highest stable semantic version in a specified calendar cohort. The four outputs are expected counts of distinct, newly published GitHub-reviewed advisories in the next 90 days, ordered **low, moderate, high, critical**. A GHSA can count once even if several versions or several of its listed ranges are affected. Publication dates measure disclosures. They do not establish when a defect entered the code.

## Start training with LightGBM

For the July 2025 through June 2026 CSV, put `npm_vulnerability_prediction.csv` beside `train_lightgbm.py`. Open a terminal in that folder and run:

```bash
python -m pip install -r requirements.txt
python train_lightgbm.py
```

The script reads the CSV directly and writes `npm_model.json` in the same folder. It also recognizes the filename `npm_vulnerability_prediction` without an extension. The direct run assumes the original extractor's July 1, 2026 UTC cutoff for the four 90-day target columns. For a different collection period, use `python train_lightgbm.py train --csv-cutoff YYYY-MM-01` with the first day after your last feature month. A manifest is optional for this single-file workflow.

Unzip the bundle and open a terminal in `npm_forecast`. Install the dependencies once with `python -m pip install -r requirements.txt`. The file `train_lightgbm.py` fits one `LGBMRegressor(objective="poisson")` for each of low, moderate, high and critical. It groups the version rows into one example per package and forecast date. Only the four named target fields enter the loss. Blank optional features reach LightGBM as missing values. One example per package is enough to contribute to training when you pool many packages.

**If you have the manifest from the original extractor**, you can also pass it explicitly to use the recorded dates. This format measures the 90 days starting at the common calendar cutoff. A file collected before its 90 days elapsed can contain false zeros even if its dates have now passed. Use complete observed outcomes and regenerate early or stale extracts. The manifest has no historical cache acquisition timestamp to verify this for you.

```bash
python train_lightgbm.py train npm_vulnerability_training.csv --manifest npm_vulnerability_manifest.json --model npm_model.json
python train_lightgbm.py predict new_versions.csv --manifest new_manifest.json --package YOUR_PACKAGE --model npm_model.json
```

`new_versions.csv` is a later package window in the same feature schema. Its target columns can be omitted. For the requested **90 days after the highest stable version**, use the release-anchored JSONL workflow below. A model is tied to the label convention it learned. Save separate model files if you experiment with both conventions.

The default prediction is a Python tuple of whole counts, rounded half up. Add `--expected` to see fractional means such as `(0.2, 0.8, 1.1, 0.05)`. This is an illustration of the output type. In Python, load the model with `json.loads(Path("npm_model.json").read_text())`, then call `train_lightgbm.predict_tuple(model, version_rows)` on the JSONL release rows. For a list of original CSV row dictionaries also supply `cutoff=manifest["prediction_at"]`.

The trainer checks for at least two completed training windows and checks for positive events in each of the four target columns across the complete training set. If one severity has no events, its Poisson model cannot be fitted. The error identifies those severities. Include packages with zero outcomes in the data as well. A small dataset may yield only constant predictions with the default minimum of ten observations per leaf. Test against historical-mean and zero-count baselines before interpreting the output.

## Build the release-anchored dataset and train

Use Python 3.10 or newer and Git. Run these commands from the unzipped `npm_forecast` directory. `packages.txt` is your own UTF-8 list with one exact npm name per line, including the scope when applicable. The package passed to `predict` must occur in that list. Positional names also work for `dataset.py`. Use the same optional feature settings across cohorts.

```bash
python -m pip install -r requirements.txt
git clone https://github.com/github/advisory-database.git advisory-database
python -m unittest discover -v

python dataset.py --packages-file packages.txt --advisory-repo advisory-database \
  --cohort-end 2024-01-01 --cohort-end 2024-04-01 --cohort-end 2024-07-01 \
  --cohort-end 2024-10-01 --cohort-end 2025-01-01 --cohort-end 2025-04-01 \
  --output older.jsonl
python dataset.py --packages-file packages.txt --advisory-repo advisory-database \
  --cohort-end 2026-07-01 --output july2025_june2026.jsonl

python train_lightgbm.py backtest older.jsonl july2025_june2026.jsonl --as-of 2025-07-01
python train_lightgbm.py train older.jsonl --as-of 2025-07-01 --model model.json
python train_lightgbm.py predict july2025_june2026.jsonl --package example-package --model model.json
```

Replace `example-package` with a package from your cohort. The final command prints exactly one tuple. Pass `--expected` for the four unrounded means. The Python interface is `train_lightgbm.predict_tuple(model_dict, version_rows)` and also returns a four-tuple of integers. Use `expected=True` for the means. The model is a portable JSON file. The package name selects the sequence and is excluded from the learned predictors.

Use a full main-branch clone. Keep the repository up to date for a new run and retain the reported Git revisions for reproducibility. Use `--registry-dir DIR` to supply your own captured full npm registry JSON documents. For example, the filenames for `demo` and `@scope/demo` are `demo.json` and `%40scope%2Fdemo.json`. GitHub issue and pull-request activity is collected by default. Use `--no-github-activity` to skip it or `--no-downloads` to skip downloads. The GitHub activity option may need `GITHUB_TOKEN` for rate limits. The token is only attached to HTTPS requests to `api.github.com`. The offline tests use local fixtures and make no network requests.

## Add GitHub fields to an existing JSONL dataset

```bash
python enrich_github.py older.jsonl --output older_github.jsonl
```

Keep the original `older.jsonl.manifest.json` alongside the input, or pass its path with `--manifest`. Older JSONL files rely on that manifest for the repository recorded on the anchor release. New datasets also embed `github_repo`. To add these attributes to legacy CSV-based training, regenerate the cohort with `dataset.py` and use its JSONL output for release-based forecasting.

Each of the twelve monthly slots has counts for issues opened, closed, reopened and open at month end, and pull requests opened, closed unmerged, merged, reopened and open at month end. The API identifies pull requests inside issue-list responses by the `pull_request` key. The month containing the prediction can only supply month-to-date events. If lifecycle collection fails after retrieving the full opening inventory, opening counts survive and dependent fields are missing unless a previous enrichment already observed them. The manifest or per-example `github_collection` explains coverage. The enrichment command exits nonzero on partial or unavailable coverage and saves the usable output. Repository history is shared for packages in the same monorepo.

## Input window and target

1. Select stable release identifiers published in `[2025-07-01 00:00 UTC, 2026-07-01 00:00 UTC)` for the requested cohort. Earlier cohorts use the corresponding twelve calendar months. Prereleases are excluded. Compare major, minor and patch numerically. Equal precedence from build metadata uses the earliest publication time, then the literal version if timestamps are identical. An empty cohort has no prediction.
2. Set the prediction time `t` to the selected release's npm publication timestamp. Keep each stable release in the cohort that existed by `t`, including the anchor. Sort records by publication time for temporal summaries. A lower maintenance release published after `t` is recorded as an exclusion in the manifest. The cohort end is a selection boundary. It never starts the target. The contract treats “highest” as highest among the supplied year's releases. An all-time-high requirement would change eligibility, especially for packages publishing backports only.
3. Fix M0 to June 2026 and M11 to July 2025. A month after `t` is missing. In the month containing `t`, use visible releases, advisories and lifecycle events only. Count complete daily download observations strictly before the UTC anchor date. A partial month's month-end backlog is missing. A day reporting zero downloads still counts toward day coverage.
4. At `t`, read the last first-parent Git state strictly before `t` for the reviewed advisory tree. For each release, count distinct GHSAs whose npm package and affected version interval match it. The lifetime columns include earlier history. The monthly columns bucket those same IDs by publication month. A release row describes what is known about that version **at prediction time**, which can be later than that row's own publication.
5. Read the reviewed tree again just before `t + 90 days`. Count one ID per package if its publication lies in `[t, t + 90 days)` and it still qualifies at the observation boundary. All versions of that package qualify for the label, including versions absent from the input. An advisory withdrawn before the boundary is excluded. The label becomes null if the archive lacks the full 90 days of follow-up. An unlabeled example is usable for inference.

For example, if `2.0.0` ships May 15 at 12:00, its target ends August 13 at 12:00 UTC. A June 1 disclosure contributes to the target if it qualifies. A `1.5.0` release on June 10 is unavailable at prediction. The original extractor counted the June disclosure in historical features and anchored the outcome to July 1.

**CVSS convention.** Use CVSS v4 Base when present, then v3.1/v3.0 Base. Within the chosen family use the maximum score applicable to the package or release if several scores are supplied. Map 0.1–3.9 to low, 4.0–6.9 to moderate, 7.0–8.9 to high and 9.0–10.0 to critical. Score 0.0 has its own `none` audit count. Absent supported scores have an `unknown` audit count. The two audit labels never enter the four model outputs. The JSON includes these audit fields alongside the named PCOM fields. Original sheet wording that describes package-wide GHSA history must say “whose affected range includes this exact npm version” to match this design.

## Model and evaluation

One JSONL object holds all `rows` for one package window and one `labels` object. Across historical extracts the statistical unit is a package and release timestamp. Repeated anchors are counted once. When different calendar cohorts select the same anchor, the trainer keeps the earliest supplied cohort for that anchor, which contains the longest eligible history. Conflicting labels cause an error.

The encoder summarizes a variable-length sequence with release cadence, package age and history coverage, the anchor value, version mean and change since the first observed release for static and lifetime advisory fields. It retains twelve monthly means for each version-specific GHSA band and each package-wide metric once per month. Version means describe exposure per release. They are never used as a count of unique package advisories. LightGBM uses native missing feature handling. The earlier linear baseline uses indicators and training-set median imputation. Counts and changes use a signed `log1p` transform.

The earlier `forecast.py` baseline fits four L2-regularized Poisson regressions with separate log links. The default regularization is `--alpha 1`. Constant training labels use a constant head, including zero. Fractional predictions are conditional expected counts. Integer mode rounds half up for reporting. For rare outcomes, many expectations below 0.5 can produce the same integer tuple. Score the unrounded means on held-out windows.

`--as-of` is an evaluation or model-training boundary. A training label must **finish** by that instant. A test prediction must **start** on or after it. A window crossing the boundary is purged. `backtest` reports test event totals, positive windows, predicted totals, MAE and mean Poisson deviance per band against constant-zero and training-mean baselines. Positive events assigned a zero mean produce infinite deviance, reported explicitly. Repeat this evaluation at successively later split dates to assess stability and tune hyperparameters on earlier periods. An evaluation on new package identities can additionally hold out whole packages or shared repositories. Keep the final July 2025–June 2026 cohort out of model selection. A single observed window for one package yields one supervised example. The implementation does not learn a separate model for each package. All eligible training windows contribute once to each severity model.

## Review of the submitted extractor

Line numbers refer to the uploaded 662-line Python script. Its 224-column layout covers the field families in the supplied PDF, including the unknown severity bucket. Distinct GHSA IDs, exact-version range checks, excluding prereleases and keeping missing optional measurements separate from measured zeros are sound choices. The changes below affect what the numbers mean.

| Original lines | Finding | Consequence and handling |
| --- | --- | --- |
| 169–190, 212–215, 264–272, 616 | A session-wide bearer header is shared by GitHub, the npm registry and the npm downloads service. | Verified by preparing requests with a dummy token. If a real token was used with the script, revoke or rotate it. The new collector scopes the header to the GitHub API host. |
| 534–570, 617–625 | The label uses July 1 plus 90 days for every package. Highest SemVer only reaches repository selection and the manifest. `max` also runs before filtering to the year's releases. | For a May 15 noon anchor the original label ends September 29 at 00:00, while the specified horizon ends August 13 at noon. Select the eligible SemVer first and use its own UTC timestamp for features, labels and metadata. |
| 254–290, 371–394, 551–570 | Collecting at the July cutoff exposes later backports, advisories, downloads and repository events. Sorting output by SemVer also differs from event order on maintenance branches. | Censor every feature family at the release anchor. Keep the time order and the SemVer rule independent. Month-end backlogs and daily totals require their periods to have ended. |
| 293–324, 371–421, 626–630 | Present-day reviewed records provide current mappings, affected ranges and severity. A present-day review timestamp can reflect a later review. Missing review timestamps pass the original filter. | A date filter on publication cannot reconstruct earlier revisions. Use independently pinned feature and outcome snapshots. The original manifest already identifies part of this limitation. |
| 293–297, 384, 420 | The code reads GitHub's textual `severity`. It never uses a numeric CVSS score or a documented precedence between score versions. | The qualitative label and numeric CVSS bands can differ. Define the scoring convention and account separately for 0.0 and missing scores. |
| 89–97, 397–421, 554–571 | Label deduplication by package is correct. The label and package-wide metrics are copied onto every version row. There is no fitting, inference, or split logic. | A row-wise loss overweights prolific releasers. A random row split can put the same package and future outcome in training and validation. Group sequences before fitting and exclude every target from features. |
| 182–197, 319–324, 397–421 | The URL cache has no retrieval date, freshness policy or guarantee of complete follow-up. A run or cached response collected before outcome maturity can turn unobserved days into zeros. | Persist the source revision and observation-through date. The new builder emits null for an incomplete label. Simply changing the cutoff in an already exported CSV cannot repair it. |
| 319–324 | The advisory query truncates `outcome_end` to a date. This happens to fit the original midnight cutoff. | With a noon release anchor, a date-strict “before” query drops the entire final UTC date, including qualifying morning disclosures. Preserve full instants when selecting labels. |
| 327–394 and PDF page 2 | The implemented history depends on exact vulnerable ranges, while the PDF says to count GHSAs naming the package. The PDF's package-level target wording agrees with the code. | Update the feature definitions to match the intended version-level history. Keep unknown scores as audit data. The `unknown` target supports auditing. |
| 117–140, 220–250, 270–289 | `int()` truncates fractions and does not bound values to signed 64-bit. Malformed dependency maps become zero. Stable releases with absent times/metadata disappear. An empty daily response can become a zero total. | Validate counts and release inventory. Preserve unknown totals and record observed download days. Report incomplete inputs. |
| 143–162, 424–510, 557–564 | Current top-level repository metadata can select a repository the package moved to later. Some repository URL forms are missed. Malformed events are skipped, event IDs are not deduplicated, and same-second merge/close pairing is heuristic. | Use the anchor release's repository declaration. Treat incomplete optional sources as missing. Deduplicate event IDs. Optional PR collection still uses timing when a close and merge have to be paired. Monorepo activity describes the repository shared by its packages. |
| 513–530, 580–658 | The default `packages.text` was absent from the attachments. Per-package errors enter the manifest, but partial runs with at least one row exit successfully. Unknown affected ranges or pagination limits can drop packages systematically. | Supply a package list, inspect collection failures and coverage, and record the historical sampling frame. The new builder returns nonzero on a required collection error and retains a manifest for diagnosis. |

## Data quality and interpretation

The first-parent archive makes the GitHub-reviewed tree reproducible by commit. Git export times can lag the public API, and changes incorporated after a boundary are intentionally unavailable at that boundary. For the npm registry, deleted/unpublished releases and corrected metadata can disappear from today's packument. A source digest identifies what was used, but cannot restore missing historical records. Use retained registry snapshots for stronger historical coverage. Optional downloads and issue/PR histories are filtered by event date and can still reflect later upstream corrections or deletions. The manifest reports missing sources, excluded backports, collection errors, and label availability. Inspect these by cohort, especially when the archive or a provider lacks coverage.

The package list defines the study population. Include eligible packages with no prior or future advisory. If claiming results for *open-source* npm packages, derive the list from a documented historical scope and license criterion. In this implementation, package selection and license verification belong to the supplied list. If the system will make a fresh forecast at every new record-high release, evaluate that release-triggered population separately. The final highest release of a year is a retrospective selection.

The supplied files contained the extractor and schema. No production package corpus, trained weights or held-out accuracy measurement accompanies this implementation. Offline regression tests exercise SemVer backports, time boundaries, revised and withdrawn advisories, exact affected ranges, deduplication, unknown and zero CVSS, partial months, token scoping, package grouping, purged validation and the build/train/predict command flow.

## References

- LightGBM Poisson objective and Python regressor. [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) · [LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- GitHub repository issue events and rate-limit handling. [Issue events](https://docs.github.com/en/rest/issues/events) · [REST API limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)

- GitHub global advisory REST fields and filters. [GitHub Docs](https://docs.github.com/en/rest/security-advisories/global-advisories)
- GitHub reviewed advisory archive and field conventions. [Advisory Database](https://github.com/github/advisory-database)
- Package-specific scoring and affected range events. [OSV schema](https://ossf.github.io/osv-schema/)
- CVSS Base scores and severity levels. [FIRST v3.1](https://www.first.org/cvss/v3.1/specification-document) · [FIRST v4.0](https://www.first.org/cvss/v4.0/specification-document)
- Version precedence and build metadata. [Semantic Versioning 2.0](https://semver.org/)
- Handling exposed credentials. [GitHub Docs](https://docs.github.com/en/rest/authentication/keeping-your-api-credentials-secure)
