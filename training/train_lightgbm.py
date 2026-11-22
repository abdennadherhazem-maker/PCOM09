from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

from core import BASE, BANDS, GHSA, HORIZON, MONTHLY, TARGETS, UTC, calendar_end, instant, iso, shift_month, stable
from forecast import COLUMNS, Sample, encode, metrics, numeric, read, split_time

MODEL_FORMAT = 'lightgbm-npm-poisson/2'
CSV_FORMAT = 'calendar-cutoff-csv'
RELEASE_FORMAT = 'release-anchor-jsonl'
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = 'npm_trainingset.csv'
TRAINING_CSV_STEMS = {Path(DEFAULT_CSV).stem, 'npm_vulnerability_prediction'}
DEFAULT_CUTOFF = '2026-07-01'
CSV_HISTORY = GHSA[:-1]
CSV_NAMES = ['release_count', 'age_days', 'history_days', 'days_since_last_release']
CSV_NAMES += [f'{c}_{k}' for c in (*BASE, *CSV_HISTORY) for k in ('highest', 'mean', 'change')]
CSV_NAMES += [f'{c}_M{i}' for c in (*CSV_HISTORY, *MONTHLY) for i in range(12)]


def identity(value):
    if not value.startswith('pkg:npm/') or '@' not in value[8:]:
        raise ValueError(f'Invalid exact npm PURL: {value!r}')
    raw_name, raw_version = value[8:].rsplit('@', 1)
    name, ver = unquote(raw_name), unquote(raw_version)
    if not name or not ver:
        raise ValueError(f'Incomplete PURL: {value!r}')
    return name, ver


def csv_identity(row, training=False):
    value = (row.get('purl') or '').strip()
    if value or training:
        return identity(value)
    name = row.get('package') or row.get('package_name') or row.get('name') or ''
    version = row.get('version') or row.get('package_version') or ''
    return name.strip() or None, version.strip() or None


def infer_csv_cutoff(groups):
    candidates = None
    for package, rows in groups.items():
        releases = Counter(shift_month(instant(row['published_at']), 0) for row in rows)
        earliest, latest = min(releases), max(releases)
        if candidates is None:
            candidates = {shift_month(latest, offset) for offset in range(1, 13)}
        reported = [numeric(rows[0], f'package_stable_release_count_M{i}') for i in range(12)]
        candidates = {
            end for end in candidates
            if shift_month(end, -12) <= earliest and latest < end
            and all(np.isnan(count) or count == releases[shift_month(end, -1-i)]
                    for i, count in enumerate(reported))
        }
        if not candidates:
            raise ValueError(f'Cannot infer a common CSV cutoff at {package}. Check that all stable '
                             'versions are present and the monthly release counts use one feature window.')
    if not candidates or len(candidates) != 1:
        raise ValueError('The CSV does not identify a unique cutoff. Supply the extraction manifest '
                         'with --manifest or the first day after the last feature month with --csv-cutoff.')
    return next(iter(candidates))


def csv_sample(pkg, rows, cutoff, training):
    start = shift_month(cutoff, -12)
    release_rows, seen = [], set()
    creation = instant(rows[0]['package_created_at'])
    labels = None
    for row in rows:
        name, ver = csv_identity(row, training)
        rank = stable(ver) if ver is not None else None
        published = instant(row['published_at'])
        if ((name is not None and name != pkg) or (ver is not None and (ver in seen or rank is None))
                or instant(row['package_created_at']) != creation
                or not (creation <= published and start <= published < cutoff)):
            raise ValueError(
                f'Invalid package, duplicate, prerelease, or out-of-window release: {pkg}, {ver}, {row["published_at"]}')
        if ver is not None:
            seen.add(ver)
        release_rows.append((published, ver or '', rank, row))
        if training:
            values = tuple(numeric(row, field, required=True) for field in TARGETS)
            if labels is not None and labels != values:
                raise ValueError(f'Conflicting package targets for {pkg}')
            labels = values
    highest = rows[-1]
    if len(seen) < len(rows):
        known_ranks = [r[2] for r in release_rows if r[2] is not None]
        if known_ranks != sorted(known_ranks):
            raise ValueError(f'For {pkg}, provide a version for every row or keep rows in ascending semantic-version order.')
    release_rows.sort(key=lambda r: (r[0], r[1]))
    if len(seen) == len(rows):
        highest = max(release_rows, key=lambda r: r[2])[3]
    rows = [r[3] for r in release_rows]
    first, last = release_rows[0][0], release_rows[-1][0]
    features = [len(rows), (cutoff-creation).total_seconds()/86400, (last-first).total_seconds()/86400,
                (cutoff-last).total_seconds()/86400]
    for field in (*BASE, *CSV_HISTORY):
        vals = [numeric(r, field, required=field not in BASE[:2]) for r in rows]
        observed = [v for v in vals if np.isfinite(v)]
        anchor = numeric(highest, field)
        features += [anchor, np.mean(observed) if observed else np.nan, anchor-vals[0]]
    for metric in (*CSV_HISTORY, *MONTHLY):
        for month in range(12):
            key = f'{metric}_M{month}'
            vals = np.array([numeric(r, key, required=metric in CSV_HISTORY) for r in rows])
            if metric in CSV_HISTORY:
                features.append(vals.mean())
            else:
                if not np.all((vals == vals[0]) | (np.isnan(vals) & np.isnan(vals[0]))):
                    raise ValueError(f'Inconsistent package-wide {key} on {pkg}')
                features.append(vals[0])
    return Sample(pkg, cutoff, cutoff+HORIZON, np.array(features), np.array(labels) if training else None)


def load_samples(paths, manifests, training=True, package=None, anchor=None, csv_cutoff=None):
    csv_files = [p.suffix.lower() == '.csv' or p.name in TRAINING_CSV_STEMS for p in paths]
    if csv_cutoff is not None and (manifests or len(paths) != 1 or not all(csv_files)):
        raise ValueError('--csv-cutoff applies to one CSV without a manifest')
    if any(csv_files) != all(csv_files):
        raise ValueError('Train separately on CSV calendar targets and JSONL release targets')
    if not any(csv_files):
        if manifests:
            raise ValueError('--manifest is for original CSV files')
        if training:
            samples = read(paths)
        else:
            samples = []
            for path in paths:
                with path.open() as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get('format') != 'npm-release-counts/1':
                            raise ValueError('Expected release-anchored JSONL from dataset.py')
                        name, t = obj['package'], instant(obj['prediction_at'])
                        if (package is not None and name != package) or (anchor and t != anchor):
                            continue
                        rows = obj['rows']
                        x = encode(rows)
                        if name != rows[0]['package'] or t != instant(rows[0]['prediction_at']):
                            raise ValueError('JSONL package or anchor disagrees with its rows')
                        samples.append(Sample(name, t, t+HORIZON, x, None))
        return RELEASE_FORMAT, samples
    if csv_cutoff is None and (training or manifests) and len(paths) != len(manifests):
        raise ValueError(
            'Supply --csv-cutoff YYYY-MM-01 for one CSV, or one --manifest per CSV in the same order')
    samples = {}
    for index, path in enumerate(paths):
        t = None
        if manifests:
            manifest = manifests[index]
            meta = json.loads(manifest.read_text())
            t = instant(meta['prediction_at'])
            if (t != calendar_end(t.date().isoformat()) or instant(meta['label_end_exclusive']) != t+HORIZON or
                    instant(meta['input_window_start']) != shift_month(t, -12) or instant(meta['input_window_end_exclusive']) != t):
                raise ValueError(f'Inconsistent monthly cutoff or 90-day horizon in {manifest}')
        elif csv_cutoff is not None:
            t = calendar_end(csv_cutoff)
        if training and t+HORIZON > datetime.now(UTC):
            raise ValueError(f'Incomplete CSV outcome period in {path}')
        groups = defaultdict(list)
        with path.open(newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            required = {'package_created_at', 'published_at', *BASE, *CSV_HISTORY}
            required.update(f'{m}_M{i}' for m in (*CSV_HISTORY, *MONTHLY) for i in range(12))
            if training:
                required.add('purl')
                required.update(TARGETS)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'{path} lacks columns: {sorted(missing)}')
            for row in reader:
                name, _ = csv_identity(row, training)
                groups[name].append(row)
        if None in groups:
            if len(groups) > 1:
                raise ValueError(f'{path} mixes named and anonymous rows. Supply a package name for each row.')
            groups[package or path.stem] = groups.pop(None)
        if package is not None:
            groups = {package: groups[package]} if package in groups else {}
        if not groups:
            continue
        if t is None:
            t = infer_csv_cutoff(groups)
        if anchor is not None and anchor != t:
            continue
        for name, rows in groups.items():
            key = (name, t)
            if key in samples:
                raise ValueError(f'Duplicate CSV package window: {name}, {iso(t)}')
            samples[key] = csv_sample(name, rows, t, training)
    return CSV_FORMAT, list(samples.values())


def fit_severity(X, y, columns):
    reg = lgb.LGBMRegressor(objective='poisson', n_estimators=150, learning_rate=0.07,
                            num_leaves=15, min_child_samples=10, reg_lambda=2.0,
                            random_state=45, n_jobs=1, verbosity=-1)
    reg.fit(X, y, feature_name=columns)
    return reg.booster_


def fit(samples, kind):
    if not samples:
        raise ValueError('No completed training windows. Supply labeled data from an earlier period.')
    X = np.stack([s.x for s in samples])
    Y = np.stack([s.y for s in samples])
    columns = COLUMNS if kind == RELEASE_FORMAT else CSV_NAMES
    if Y.shape != (len(samples), len(BANDS)) or not np.isfinite(Y).all() or (Y < 0).any():
        raise ValueError(f'Training requires {len(BANDS)} finite nonnegative targets per window: {BANDS}')
    if len(samples) < 2:
        raise ValueError('LightGBM needs at least two package windows. Each package history is one example.')
    heads = {}
    for j, severity in enumerate(BANDS):
        # A Poisson booster cannot fit an all-zero response. Record the fitted
        # zero baseline explicitly for that target; positive labels use LightGBM.
        y = Y[:, j]
        heads[severity] = (fit_severity(X, y, columns).model_to_string()
                           if np.any(y) else {'constant': 0.0})
    return dict(format=MODEL_FORMAT, data_format=kind, order=list(BANDS), feature_names=columns,
                version=lgb.__version__, train_through=iso(max(s.end for s in samples)),
                training_windows=len(samples), training_mean=Y.mean(axis=0).tolist(), heads=heads)


def expected_counts(model, X):
    columns = COLUMNS if model['data_format'] == RELEASE_FORMAT else CSV_NAMES
    if model.get('format') != MODEL_FORMAT or model['order'] != list(BANDS) or model['feature_names'] != columns:
        raise ValueError('Incompatible model or feature schema. Retrain for five severities, including unknown.')
    if set(model.get('heads', {})) != set(BANDS):
        raise ValueError(f'Model must contain exactly these severity heads: {BANDS}')
    X = np.atleast_2d(X)
    preds = []
    for band in BANDS:
        head = model['heads'][band]
        if isinstance(head, dict) and set(head) == {'constant'}:
            preds.append(np.full(X.shape[0], float(head['constant']), dtype=float))
        elif isinstance(head, str):
            booster = lgb.Booster(model_str=head)
            preds.append(booster.predict(X, num_threads=1))
        else:
            raise ValueError(f'Invalid model head for {band}')
    values = np.column_stack(preds)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid predicted mean')
    return values


def predict_tuple(model, rows, *, cutoff=None, expected=True):
    if model['data_format'] == RELEASE_FORMAT:
        x = encode(rows)
    else:
        if not rows:
            raise ValueError('CSV prediction requires package version rows')
        name = next((name for row in rows if (name := csv_identity(row)[0])), 'package')
        t = instant(cutoff) if isinstance(cutoff, str) else cutoff
        if t is None:
            t = infer_csv_cutoff({name: rows})
        x = csv_sample(name, rows, t, False).x
    return count_tuple(expected_counts(model, x)[0], expected)


def count_tuple(means, expected):
    if len(means) != len(BANDS):
        raise ValueError(f'Expected {len(BANDS)} severity counts, got {len(means)}')
    return tuple(float(v) for v in means) if expected else tuple(math.floor(v+0.5) for v in means)


def write_prediction_csv(path, samples, counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent,
                                delete=False) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.writer(tmp)
            writer.writerow(['package', '(' + ', '.join(BANDS) + ')'])
            for sample, values in zip(samples, counts):
                writer.writerow([sample.name, str(values)])
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def evaluate_packages(samples, kind):
    samples = sorted((s for s in samples if s.y is not None), key=lambda s: (s.name, s.t))
    packages, groups = np.unique([s.name for s in samples], return_inverse=True)
    if len(packages) < 3:
        raise ValueError('Package cross-validation needs at least three distinct labeled packages: two to train and one to test.')
    folds = min(5, len(packages))
    X = np.stack([s.x for s in samples])
    Y = np.stack([s.y for s in samples]).astype(float, copy=False)
    predictions = np.full(Y.shape, np.nan)
    baseline = np.empty_like(Y, dtype=float)
    columns = COLUMNS if kind == RELEASE_FORMAT else CSV_NAMES
    missing_folds = {band: [] for band in BANDS}
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (_, held_out) in enumerate(splitter.split(packages), 1):
        test = np.isin(groups, held_out)
        train = ~test
        baseline[test] = Y[train].mean(axis=0)
        for j, band in enumerate(BANDS):
            y = Y[train, j]
            if not np.any(y):
                missing_folds[band].append(fold)
                predictions[test, j] = 0.0
                continue
            booster = fit_severity(X[train], y, columns)
            predictions[test, j] = booster.predict(X[test], num_threads=1)

    mean_scores = metrics(Y, baseline)
    zero_scores = metrics(Y, np.zeros_like(Y))
    model_scores = metrics(Y, predictions)
    results = {}
    for j, band in enumerate(BANDS):
        results[band] = {
            'events': int(Y[:, j].sum()),
            'positive_windows': int((Y[:, j] > 0).sum()),
            'training_mean_baseline': mean_scores[band],
            'zero_baseline': zero_scores[band],
            'model': None,
            'folds_without_training_events': missing_folds[band],
        }
        mu = predictions[:, j]
        if not np.isfinite(mu).all() or (mu < 0).any():
            raise ValueError(f'Invalid held-out predictions for {band}')
        score = model_scores[band]
        score['rounded_mae'] = float(np.abs(Y[:, j] - np.floor(mu + 0.5)).mean())
        results[band]['model'] = score
    return dict(method='held-out packages', folds=folds, seed=42, packages=len(packages),
                windows=len(samples), data_format=kind, severities=results)


def print_evaluation(report):
    print(f"{report['folds']}-fold cross-validation on {report['packages']} packages "
          f"({report['windows']} labeled windows). Each package is tested once.")
    print('Severity   Events  Positive windows  Model MAE  Mean MAE  Zero MAE  Model dev.  Mean dev.')
    def display(value):
        return f'{value:.4f}' if isinstance(value, (int, float)) else value
    for band, row in report['severities'].items():
        model = row['model'] or {}
        mean = row['training_mean_baseline']
        print(f"{band:<9} {row['events']:>7}  {row['positive_windows']:>16}  "
              f"{display(model.get('mae', 'n/a')):>9}  {display(mean['mae']):>8}  "
              f"{display(row['zero_baseline']['mae']):>8}  "
              f"{display(model.get('mean_poisson_deviance', 'n/a')):>10}  "
              f"{display(mean['mean_poisson_deviance']):>9}")
        if row['folds_without_training_events']:
            print(f"  {band}: a constant-zero predictor was used in folds with no training events.")
    print('Lower MAE and Poisson deviance are better. MAE is the average absolute error in counts per package window.')


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('train', 'backtest', 'predict', 'evaluate'):
        cli = sub.add_parser(action)
        cli.add_argument('data', type=Path, nargs='*' if action in ('train', 'evaluate') else '+')
        cli.add_argument('--manifest', action='append', type=Path, default=[])
        cli.add_argument('--csv-cutoff', metavar='YYYY-MM-01',
                         help='First day after the last CSV feature month. Prediction can infer it from version rows and monthly release counts.')
        if action in ('train', 'predict'):
            cli.add_argument('--model', type=Path, default=SCRIPT_DIR/'npm_model.json')
        if action in ('train', 'backtest'):
            cli.add_argument('--as-of', required=action == 'backtest',
                             help='UTC split. Train uses all mature labels by default.')
        elif action == 'predict':
            cli.add_argument('--output', type=Path, help='Prediction CSV. Defaults to <input>_predictions.csv beside the first input.')
            cli.add_argument('--package', help='Optional exact package name. Defaults to all packages in the input.')
            cli.add_argument('--prediction-at', help='UTC prediction time if the input has several cohorts')
            cli.add_argument('--expected', action=argparse.BooleanOptionalAction, default=True,
                             help='Write floating-point expected counts (default). Use --no-expected for rounded counts.')
        else:
            cli.add_argument('--report', type=Path, default=SCRIPT_DIR/'npm_evaluation.json')
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments or ['train'])
    if args.action in ('train', 'evaluate') and not args.data:
        input_path = SCRIPT_DIR / DEFAULT_CSV
        if not input_path.is_file() and input_path.with_suffix('').is_file():
            input_path = input_path.with_suffix('')
        if not input_path.is_file():
            raise ValueError(f'Place {DEFAULT_CSV} next to train_lightgbm.py in {SCRIPT_DIR}, then run again.')
        args.data = [input_path]
    if (args.action in ('train', 'evaluate') and len(args.data) == 1 and not args.manifest and args.csv_cutoff is None
            and args.data[0].stem in TRAINING_CSV_STEMS
            and args.data[0].suffix.lower() in ('', '.csv')):
        args.csv_cutoff = DEFAULT_CUTOFF
    is_train = args.action != 'predict'
    anchor = split_time(args.prediction_at) if getattr(args, 'prediction_at', None) else None
    kind, samples = load_samples(args.data, args.manifest, is_train, getattr(args, 'package', None),
                                 anchor, csv_cutoff=args.csv_cutoff)
    if args.action == 'evaluate':
        report = evaluate_packages(samples, kind)
        print_evaluation(report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Saved the evaluation to {args.report}')
        return 0
    if not is_train:
        model = json.loads(args.model.read_text())
        if kind != model['data_format']:
            raise ValueError('Prediction input must use the same data format as training')
        if not samples:
            raise ValueError('No package version rows matched the prediction input and filters')
        if args.package and len(samples) > 1:
            raise ValueError('Multiple windows for this package. Select one using --prediction-at.')
        first_input = args.data[0]
        output = args.output or first_input.with_name(first_input.stem + '_predictions.csv')
        if output.resolve() in {p.resolve() for p in [*args.data, *args.manifest, args.model]}:
            raise ValueError('Choose a prediction CSV path different from the data, manifest and model files.')
        means = expected_counts(model, np.stack([s.x for s in samples]))
        counts = [count_tuple(values, args.expected) for values in means]
        write_prediction_csv(output, samples, counts)
        print('predicted')
        return 0
    split = split_time(args.as_of) if args.as_of else datetime.now(UTC)
    training = [s for s in samples if s.y is not None and s.end <= split]
    model = fit(training, kind)
    if args.action == 'train':
        args.model.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile('w', dir=args.model.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                json.dump(model, tmp, allow_nan=False)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        os.replace(tmp_path, args.model)
        print('trained')
        return 0
    test = [s for s in samples if s.y is not None and s.t >= split]
    if not test:
        raise ValueError('Backtesting needs mature labels for test windows starting on or after --as-of')
    y = np.stack([s.y for s in test])
    mu = expected_counts(model, np.stack([s.x for s in test]))
    print(json.dumps(dict(train_windows=len(training), test_windows=len(test),
                          model=metrics(y, mu), historical_mean=metrics(y, np.broadcast_to(model['training_mean'], y.shape)),
                          zero_baseline=metrics(y, np.zeros_like(y))), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        sys.exit(str(exc))
