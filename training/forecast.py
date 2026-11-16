"""Fit, predict and evaluate four 90-day package-level GHSA publication counts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.linear_model import PoissonRegressor
from threadpoolctl import threadpool_limits

from core import FORMAT, AUDIT, BANDS, BASE, GHSA, HORIZON, MONTHLY, TARGETS, calendar_end, instant, iso, purl, shift_month, stable, uint


def field_columns():
    names = ['releases', 'package_age_days', 'observed_span_days', 'last_gap_days', 'cohort_days_at_anchor']
    for f in (*BASE, *GHSA):
        names += [f+'_anchor', f+'_version_mean', f+'_last_minus_first']
    for f in GHSA:
        names += [f+'_version_mean_M'+str(i) for i in range(12)]
    names += [f+'_M'+str(i) for f in MONTHLY for i in range(12)]
    return names


COLUMNS = field_columns()


def numeric(row, name, *, required=False):
    v = row.get(name)
    if v is None or v == '':
        if required:
            raise ValueError(f'Missing required feature {name}')
        return np.nan
    if isinstance(v, str) and v.isascii() and v.isdecimal():
        v = int(v)
    return float(uint(v))


def finite_mean(values):
    valid = [x for x in values if np.isfinite(x)]
    return float(np.mean(valid)) if valid else np.nan


def encode(rows):
    """Encode one package's release series. Future label fields never enter this allowlist."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Prediction needs one nonempty list of release records')
    rows = sorted(rows, key=lambda r: (instant(r['published_at']), r['version']))
    first = rows[0]
    pkg, named_anchor = first['package'], first['anchor_version']
    t, end = instant(first['prediction_at']), instant(first['cohort_end'])
    created = instant(first['package_created_at'])
    if end != calendar_end(end.date().isoformat()):
        raise ValueError('Invalid cohort boundary')
    start = shift_month(end, -12)
    seen, ranked, times = set(), [], []
    for r in rows:
        v, at = r['version'], instant(r['published_at'])
        key = stable(v)
        if key is None or v in seen or r.get('purl') != purl(pkg, v):
            raise ValueError('Expected one distinct stable npm PURL per version')
        if not (r['package'] == pkg and instant(r['cohort_end']) == end and instant(r['prediction_at']) == t
                and r['anchor_version'] == named_anchor and instant(r['package_created_at']) == created):
            raise ValueError('Inconsistent package, release anchor, or cohort within one sequence')
        if not (created <= at and start <= at <= t < end):
            raise ValueError('Release falls outside the observable package history')
        seen.add(v)
        ranked.append((key, at, v, r))
        times.append(at)
    top = max(x[0] for x in ranked)
    _, at, v, anchor = min((x for x in ranked if x[0] == top), key=lambda x: (x[1], x[2]))
    if v != named_anchor or at != t:
        raise ValueError('Anchor is inconsistent with highest stable SemVer or its publication time')
    gaps = sorted(set(times))

    def days(duration):
        return duration.total_seconds() / 86400
    features = [len(rows), days(t-created), days(t-times[0]), days(gaps[-1]-gaps[-2])
                if len(gaps) > 1 else np.nan, days(t-start)]
    for f in (*BASE, *GHSA):
        required = f in BASE[2:] or f in GHSA
        vals = [numeric(r, f, required=required) for r in rows]
        features.extend((numeric(anchor, f), finite_mean(vals), numeric(anchor, f)-vals[0]))
    for f in GHSA:
        for i in range(12):
            visible = shift_month(end, -i-1) <= t
            vals = [numeric(r, f+'_M'+str(i), required=visible) for r in rows]
            if not visible and any(np.isfinite(vals)):
                raise ValueError('A version advisory feature uses a post-anchor month')
            features.append(finite_mean(vals))
    for f in MONTHLY:
        for i in range(12):
            col = f+'_M'+str(i)
            vals = [numeric(r, col) for r in rows]
            if any(not (x == vals[0] or np.isnan(x) and np.isnan(vals[0])) for x in vals):
                raise ValueError(f'Package-wide feature differs across release rows: {col}')
            if shift_month(end, -i-1) > t and np.isfinite(vals[0]):
                raise ValueError(f'Observed a package-wide feature after the anchor: {col}')
            if f.endswith('_at_month_end') and shift_month(end, -i) > t and np.isfinite(vals[0]):
                raise ValueError(f'Observed an unfinished month-end backlog: {col}')
            features.append(vals[0])
    x = np.array(features, dtype=float)
    return np.sign(x)*np.log1p(np.abs(x))


@dataclass
class Sample:
    name: str
    t: datetime
    end: datetime
    x: np.ndarray
    y: np.ndarray | None


def sample(obj):
    if obj.get('format') != FORMAT:
        raise ValueError('Missing release-anchored dataset format. Rebuild old fixed-cutoff CSVs.')
    rows = obj['rows']
    features = encode(rows)
    first = rows[0]
    t, label_end = instant(obj['prediction_at']), instant(obj['label_end'])
    if (obj['package'] != first['package'] or obj['anchor_version'] != first['anchor_version'] or
            instant(obj['cohort_end']) != instant(first['cohort_end']) or t != instant(first['prediction_at']) or label_end != t+HORIZON):
        raise ValueError('Sample metadata disagrees with its rows or forecast horizon')
    y = None
    if obj.get('labels') is not None:
        y = np.array([uint(obj['labels'][key]) for key in TARGETS], dtype=float)
        for key in AUDIT:
            uint(obj['labels'][key])
        if not obj.get('label_revision'):
            raise ValueError('Labeled examples need an outcome snapshot')
    return Sample(obj['package'], t, label_end, features, y)


def read(paths):
    # A package can keep the same highest SemVer in several calendar cohorts.
    # Keep its earliest supplied cohort exactly once (the longest available history).
    by_instant = {}
    for path in paths:
        with path.open() as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    end = instant(obj['cohort_end'])
                    s = sample(obj)
                    key = (s.name, s.t)
                    old = by_instant.get(key)
                    if old:
                        prev_end, prev = old
                        if prev.y is not None and s.y is not None and not np.array_equal(prev.y, s.y):
                            raise ValueError('Conflicting targets for one release anchor')
                        if prev_end == end and not np.array_equal(s.x, prev.x, equal_nan=True):
                            raise ValueError('Conflicting inputs for the same package, cohort and anchor')
                        if prev_end <= end:
                            if prev.y is None:
                                prev.y = s.y
                            continue
                        if s.y is None:
                            s.y = prev.y
                    by_instant[key] = (end, s)
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f'{path}, line {n}: {exc}') from exc
    return sorted((s for _, s in by_instant.values()), key=lambda s: (s.t, s.name))


def design(X, medians, center, scale):
    missing = np.isnan(X)
    return np.column_stack(((np.where(missing, medians, X)-center)/scale, missing.astype(float)))


def fit(samples, alpha=1.0):
    if not samples or any(s.y is None for s in samples):
        raise ValueError('Fitting requires at least one mature labeled package window')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    X = np.stack([s.x for s in samples])
    Y = np.stack([s.y for s in samples])
    medians = np.array([np.median(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0 for col in X.T])
    filled = np.where(np.isnan(X), medians, X)
    center, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-10] = 1
    Z = design(X, medians, center, scale)
    heads = []
    with threadpool_limits(limits=1):
        for y in Y.T:
            if np.all(y == y[0]):
                heads.append({'constant': float(y[0])})
            else:
                reg = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-8).fit(Z, y)
                heads.append({'bias': float(reg.intercept_), 'weights': reg.coef_.tolist()})
    return dict(format=FORMAT, columns=COLUMNS, bands=list(BANDS), medians=medians.tolist(),
                center=center.tolist(), scale=scale.tolist(), heads=heads, alpha=alpha, windows=len(samples),
                train_labels_through=iso(max(s.end for s in samples)), training_mean=Y.mean(axis=0).tolist())


def rates(model, X):
    if model.get('format') != FORMAT or model.get('columns') != COLUMNS or model.get('bands') != list(BANDS):
        raise ValueError('Incompatible forecast model or feature definition')
    Z = design(np.atleast_2d(X), np.array(model['medians']),
               np.array(model['center']), np.array(model['scale']))
    means = []
    for head in model['heads']:
        if 'constant' in head:
            mu = np.full(len(Z), head['constant'])
        else:
            mu = np.exp(Z @ head['weights'] + head['bias'])
        if not np.all(np.isfinite(mu)) or np.any(mu < 0):
            raise ValueError('Model produced an invalid count forecast')
        means.append(mu)
    if len(means) != 4:
        raise ValueError('Model must have four severity heads')
    return np.column_stack(means)


def predict(model, rows, *, expected=False):
    """Return four whole counts, or their unrounded expectations with expected=True."""
    x = encode(rows)
    if instant(model['train_labels_through']) > instant(rows[0]['prediction_at']):
        raise ValueError('Train an earlier model. This model uses outcomes later than the selected release.')
    means = tuple(float(v) for v in rates(model, x)[0])
    return means if expected else tuple(math.floor(x + 0.5) for x in means)


def metrics(y, mu):
    result = {}
    for name, actual, pred in zip(BANDS, y.T, mu.T):
        positive = actual > 0
        terms = pred-actual
        with np.errstate(divide='ignore'):
            terms[positive] += actual[positive]*np.log(actual[positive]/pred[positive])
        dev = 2 * float(terms.mean())
        result[name] = dict(events=int(actual.sum()), positive_windows=int(positive.sum()), predicted_total=float(pred.sum()),
                            mae=float(np.mean(np.abs(actual-pred))), mean_poisson_deviance=dev if math.isfinite(dev) else 'infinite')
    return result


def split_time(arg):
    return instant(arg+'T00:00:00Z' if len(arg) == 10 else arg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='action', required=True)
    train, test = subs.add_parser('train'), subs.add_parser('backtest')
    for p in (train, test):
        p.add_argument('data', nargs='+', type=Path)
        p.add_argument('--as-of', required=True,
                       help='Train only on outcomes complete by this UTC instant, or YYYY-MM-DD')
        p.add_argument('--alpha', type=float, default=1.0)
    train.add_argument('--model', required=True, type=Path)
    test.add_argument('--until', help='Optional exclusive end of test anchors (UTC)')
    pred = subs.add_parser('predict')
    pred.add_argument('--model', type=Path, required=True)
    pred.add_argument('data', type=Path,
                      help='JSONL examples; use --package when multiple packages are present')
    pred.add_argument('--package')
    pred.add_argument('--prediction-at', help='Disambiguate one release anchor')
    pred.add_argument('--expected', action='store_true',
                      help='Return unrounded expected counts in place of whole-count point estimates')
    args = ap.parse_args(argv)
    if args.action == 'predict':
        model = json.loads(args.model.read_text())
        objects = []
        wanted_at = split_time(args.prediction_at) if args.prediction_at else None
        with args.data.open() as fh:
            for line in fh:
                if line.strip():
                    obj = json.loads(line)
                    if args.package and obj['package'] != args.package:
                        continue
                    if wanted_at is None or instant(obj['prediction_at']) == wanted_at:
                        objects.append(obj)
                        if len(objects) > 1:
                            break
        if len(objects) != 1:
            raise ValueError(
                'Select exactly one example (use one cohort file, --package and/or --prediction-at)')
        obj = objects[0]
        if (obj.get('format') != FORMAT or not obj.get('rows')
                or obj['rows'][0]['package'] != obj['package']
                or obj['rows'][0]['prediction_at'] != obj['prediction_at']):
            raise ValueError('Example format, package or anchor metadata is inconsistent')
        print(predict(model, obj['rows'], expected=args.expected))
        return 0
    samples = read(args.data)
    split = split_time(args.as_of)
    train_set = [s for s in samples if s.y is not None and s.end <= split]
    model = fit(train_set, args.alpha)
    if args.action == 'train':
        args.model.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile('w', dir=args.model.parent, delete=False) as f:
            json.dump(model, f, allow_nan=False)
            tmp = Path(f.name)
        tmp.replace(args.model)
        print(f"Trained four severity heads on {len(train_set)} package windows.")
        return 0
    stop = split_time(args.until) if args.until else None
    test_set = [s for s in samples if s.y is not None and s.t >= split and (stop is None or s.t < stop)]
    if not test_set:
        raise ValueError('Backtesting needs mature test outcomes after the split')
    X = np.stack([s.x for s in test_set])
    Y = np.stack([s.y for s in test_set])
    mu = rates(model, X)
    mean = np.broadcast_to(model['training_mean'], Y.shape)
    report = {
        'train_windows': len(train_set),
        'test_windows': len(test_set),
        'training_labels_through': model['train_labels_through'],
        'first_test_anchor': iso(test_set[0].t),
        'crossing_windows_purged': sum(s.y is not None and s.t < split < s.end for s in samples),
        'unobserved_windows': sum(s.y is None for s in samples),
        'forecast': metrics(Y, mu),
        'training_mean_baseline': metrics(Y, mean),
        'zero_baseline': metrics(Y, np.zeros_like(Y)),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(str(exc))
