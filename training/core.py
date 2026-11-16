"""Package selection and as-of features for one release-anchored forecast."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from semantic_version import Version

UTC = timezone.utc
BANDS = ('low', 'moderate', 'high', 'critical')
HORIZON = timedelta(days=90)
FORMAT = 'npm-release-counts/1'
BASE = ('unpacked_size', 'file_count', 'dependency_count', 'optional_dependency_count',
        'peer_dependency_count', 'dev_dependency_count', 'bundled_dependency_count')
REPO_METRICS = ('issues_opened', 'issues_closed', 'issues_reopened', 'pull_requests_opened',
                'pull_requests_closed_unmerged', 'pull_requests_merged', 'pull_requests_reopened',
                'issues_open_at_month_end', 'pull_requests_open_at_month_end')
MONTHLY = ('package_downloads', 'package_download_days_observed',
           'package_stable_release_count', *REPO_METRICS)
GHSA = tuple(f'ghsa_{s}_count' for s in (*BANDS, 'unknown', 'none'))
TARGETS = tuple(f'target_ghsa_{s}_count_90d' for s in BANDS)
AUDIT = ('target_ghsa_unknown_count_90d', 'target_ghsa_none_count_90d')


def instant(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f'Expected a UTC timestamp, got {value!r}')
    out = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if out.tzinfo is None:
        raise ValueError(f'Timestamp lacks a timezone: {value}')
    return out.astimezone(UTC)


def iso(value: datetime) -> str:
    # Use the three-place PCOM format when possible; preserve finer input precision.
    precision = 'microseconds' if value.microsecond % 1000 else 'milliseconds'
    return value.astimezone(UTC).isoformat(timespec=precision).replace('+00:00', 'Z')


def shift_month(first: datetime, delta: int) -> datetime:
    idx = first.year * 12 + first.month - 1 + delta
    return datetime(idx // 12, idx % 12 + 1, 1, tzinfo=UTC)


def calendar_end(date_text: str) -> datetime:
    if not re.fullmatch(r'\d{4}-\d\d-01', date_text):
        raise ValueError('Cohort end must be the first day of a month, YYYY-MM-01')
    return instant(date_text + 'T00:00:00Z')


def uint(value) -> int:
    if type(value) is int and 0 <= value <= 2**63 - 1:
        return value
    raise ValueError(f'Invalid nonnegative 64-bit count: {value!r}')


def stable(raw: str):
    v = Version(raw.removeprefix('v'))
    return None if v.prerelease else (v.major, v.minor, v.patch)


def purl(pkg, version):
    return 'pkg:npm/' + '/'.join(quote(part, safe='') for part in pkg.split('/')) + '@' + quote(version, safe='')


@dataclass
class Release:
    version: str
    key: tuple[int, int, int]
    time: datetime
    meta: dict


@dataclass
class Window:
    name: str
    created: datetime
    end: datetime  # fixed cohort boundary, exclusive
    anchor: Release
    versions: list[Release]
    later: int

    @property
    def months(self):
        return [(shift_month(self.end, -i-1), shift_month(self.end, -i)) for i in range(12)]


def select_window(doc, end) -> Window | None:
    if not isinstance(doc.get('versions'), dict) or not isinstance(doc.get('time'), dict):
        raise ValueError('Incomplete registry version inventory')
    created = instant(doc['time']['created'])
    start = shift_month(end, -12)
    versions = []
    for raw in doc['versions'].keys() | doc['time'].keys():
        if raw in {'created', 'modified', 'unpublished'}:
            continue
        try:
            key = stable(raw)
        except ValueError:
            continue  # invalid SemVer is outside the population
        if key is None:
            continue
        when = instant(doc['time'].get(raw))
        if start <= when < end:
            meta = doc['versions'].get(raw)
            if not isinstance(meta, dict) or when < created:
                raise ValueError(f'Incomplete/inconsistent stable release {doc["name"]}@{raw}')
            versions.append(Release(raw, key, when, meta))
    if not versions:
        return None
    # Select within the supplied calendar cohort. Equal precedence: earliest published,
    # then the literal version for simultaneous builds. Build IDs never alter precedence.
    top = max(v.key for v in versions)
    anchor = min((v for v in versions if v.key == top), key=lambda v: (v.time, v.version))
    visible = sorted((v for v in versions if v.time <= anchor.time), key=lambda v: (v.time, v.key, v.version))
    return Window(doc['name'], created, end, anchor, visible, len(versions) - len(visible))


def version_row(window, release):
    m = release.meta
    dist = m.get('dist') or {}
    row = dict(package=window.name, version=release.version, purl=purl(window.name, release.version),
               cohort_end=iso(window.end), prediction_at=iso(window.anchor.time),
               anchor_version=window.anchor.version, package_created_at=iso(window.created), published_at=iso(release.time))
    for name, source in [('unpacked_size', 'unpackedSize'), ('file_count', 'fileCount')]:
        value = dist.get(source)
        row[name] = None if value is None else uint(value)
    for name, source in zip(BASE[2:6], ('dependencies', 'optionalDependencies', 'peerDependencies', 'devDependencies')):
        deps = m.get(source, {})
        if not isinstance(deps, dict):
            raise ValueError(f'Invalid {source} on {row["purl"]}')
        row[name] = len(deps)
    names = set()
    for source in ('bundleDependencies', 'bundledDependencies'):
        bundles = m.get(source)
        if bundles is True:
            names.update(m.get('dependencies', {}))
        elif isinstance(bundles, list) and all(isinstance(s, str) for s in bundles):
            names.update(bundles)
        elif bundles is not None and bundles is not False:
            raise ValueError(f'Invalid {source} on {row["purl"]}')
    row['bundled_dependency_count'] = len(names)
    return row


def monthly_release_features(w):
    times = [v.time for v in w.versions]
    out = {}
    for i, (a, b) in enumerate(w.months):
        observable = a <= w.anchor.time
        out[f'package_stable_release_count_M{i}'] = sum(a <= t < b for t in times) if observable else None
    return out


def advisory_features(w, release, history):
    from advisories import affects, band
    totals, monthly = Counter(), Counter()
    for ghsa in history:
        if ghsa.published >= w.anchor.time:
            continue
        matching = [e for e in ghsa.entries if affects(e, release.version)]
        if not matching:
            continue
        s = band(ghsa.severity, matching)
        totals[s] += 1
        for i, (a, b) in enumerate(w.months):
            if a <= ghsa.published < b:
                monthly[s, i] += 1
                break
    row = {}
    for s in (*BANDS, 'unknown', 'none'):
        row[f'ghsa_{s}_count'] = totals[s]
        for i, (a, _) in enumerate(w.months):
            row[f'ghsa_{s}_count_M{i}'] = monthly[s, i] if a <= w.anchor.time else None
    return row


def outcome(w, advisories):
    from advisories import band
    counts = Counter()
    for ghsa in advisories:
        if w.anchor.time <= ghsa.published < w.anchor.time + HORIZON:
            counts[band(ghsa.severity, ghsa.entries)] += 1
    return {f'target_ghsa_{s}_count_90d': counts[s] for s in (*BANDS, 'unknown', 'none')}
