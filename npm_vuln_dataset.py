"""Construct npm version records with version-level historical GHSA features.

The 90-day target GHSA counts remain package-level by design: one four-tuple
(low/moderate/high/critical) is predicted per package, not per input version.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from semantic_version import NpmSpec, Version
from urllib3.util.retry import Retry

LOG = logging.getLogger("npm-vulnerability-dataset")
UTC = timezone.utc
SEVERITIES = ("low", "moderate", "high", "critical", "unknown")
MONTHLY_FIELDS = (
    "package_downloads",
    "package_download_days_observed",
    "package_stable_release_count",
    "issues_opened",
    "issues_closed",
    "issues_reopened",
    "pull_requests_opened",
    "pull_requests_closed_unmerged",
    "pull_requests_merged",
    "pull_requests_reopened",
    "issues_open_at_month_end",
    "pull_requests_open_at_month_end",
)
GITHUB_FIELDS = MONTHLY_FIELDS[3:]
GENERAL_FIELDS = (
    "purl", "package_created_at", "published_at", "unpacked_size", "file_count",
    "dependency_count", "optional_dependency_count", "peer_dependency_count",
    "dev_dependency_count", "bundled_dependency_count",
)
TARGET_FIELDS = tuple(f"target_ghsa_{s}_count_90d" for s in SEVERITIES)
CUTOFF_DEFAULT = "2026-07-01"
DEFAULT_PACKAGES_FILE = Path(__file__).resolve().with_name("packages.text")
if not DEFAULT_PACKAGES_FILE.is_file():
    attached_list = DEFAULT_PACKAGES_FILE.with_name("packages(1).text")
    if attached_list.is_file():
        DEFAULT_PACKAGES_FILE = attached_list


def dt(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Missing/invalid timestamp: {value!r}")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone: {value!r}")
    return parsed.astimezone(UTC)


def stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def previous_months(cutoff: datetime) -> list[tuple[datetime, datetime]]:
    out = []
    end = cutoff
    for _ in range(12):
        start = month_start(end.year - (end.month == 1), (end.month - 2) % 12 + 1)
        out.append((start, end))
        end = start
    return out


def month_index(value: datetime, months: list[tuple[datetime, datetime]]) -> int | None:
    for i, (a, b) in enumerate(months):
        if a <= value < b:
            return i
    return None


def csv_columns() -> list[str]:
    fields = list(GENERAL_FIELDS)
    for metric in MONTHLY_FIELDS:
        fields.extend(f"{metric}_M{i}" for i in range(12))
    fields.extend(f"ghsa_{severity}_count" for severity in SEVERITIES)
    for severity in SEVERITIES:
        fields.extend(f"ghsa_{severity}_count_M{i}" for i in range(12))
    fields.extend(TARGET_FIELDS)
    return fields


def semver(value: str) -> Version | None:
    try:
        parsed = Version(value[1:] if value.startswith("v") else value)
        return None if parsed.prerelease else parsed
    except (ValueError, TypeError):
        return None


def npm_purl(package: str, version: str) -> str:
    if package.startswith("@") and "/" in package:
        namespace, name = package.split("/", 1)
        path = f"{quote(namespace, safe='')}/{quote(name, safe='')}"
    else:
        path = quote(package, safe="")
    return f"pkg:npm/{path}@{quote(version, safe='.-_')}"


def nonnegative_int(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def mapping_count(value) -> int:
    return len(value) if isinstance(value, dict) else 0


def bundle_count(metadata: dict) -> int:
    deps = metadata.get("dependencies") or {}
    options = [metadata.get("bundleDependencies"), metadata.get("bundledDependencies")]
    if any(v is True for v in options):
        return mapping_count(deps)
    declared = set()
    for opt in options:
        if isinstance(opt, (list, tuple)):
            declared.update(x for x in opt if isinstance(x, str))
    return len(declared)


def repository_name(*candidates) -> str | None:
    """Parse npm repository/homepage URL"""
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("url")
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip().replace("git+", "")
        candidate = re.sub(r"^git@github\.com:", "https://github.com/", candidate)
        if candidate.startswith("github:"):
            candidate = "https://github.com/" + candidate[len("github:"):]
        parsed = urlparse(candidate)
        if parsed.netloc.lower() not in ("github.com", "www.github.com"):
            continue
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            owner, repo = parts[0], parts[1].removesuffix(".git")
            if re.fullmatch(r"[A-Za-z0-9_.-]+", owner) and re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
                return f"{owner}/{repo}"
    return None


class IncompletePagination(RuntimeError):
    pass


def http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "npm-ghsa-training-dataset/1.0"})
    retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retry))
    return session


class Http:
    def __init__(self, cache_dir: Path, token: str | None = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = http_session()
        self.github_headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.github_headers["Authorization"] = f"Bearer {token}"

    def get(self, url: str, params: dict | None = None) -> tuple[object, str | None]:
        # Cache data AND the  URL
        canonical = requests.Request("GET", url, params=params).prepare().url
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{digest}.json"
        if cached.exists():
            obj = json.loads(cached.read_text(encoding="utf-8"))
            return obj["data"], obj["next"]
        dest = urlparse(url)
        headers = self.github_headers if (dest.scheme, dest.netloc) == ("https", "api.github.com") else {}
        response = self.session.get(url, params=params, headers=headers, timeout=40)
        response.raise_for_status()
        result = response.json()
        nxt = response.links.get("next", {}).get("url")
        tmp = cached.with_suffix(".tmp")
        tmp.write_text(json.dumps({"data": result, "next": nxt}), encoding="utf-8")
        tmp.replace(cached)
        return result, nxt

    def pages(self, url: str, params: dict, max_pages: int) -> list[dict]:
        values: list[dict] = []
        for index in range(max_pages):
            result, next_url = self.get(url, params if index == 0 else None)
            if not isinstance(result, list):
                raise ValueError(f"Expected JSON array from {url}")
            values.extend(result)
            if not next_url:
                return values
            url, params = next_url, {}
        raise IncompletePagination(f"Pagination exceeds {max_pages} pages at {url}; refusing truncated counts")


def retrieve_packument(http: Http, name: str) -> dict:
    url = "https://registry.npmjs.org/" + quote(name, safe="@")
    result, _ = http.get(url)
    if not isinstance(result, dict) or "versions" not in result or "time" not in result:
        raise ValueError(f"Incomplete npm registry metadata: {name}")
    return result


def valid_versions(packument: dict, cutoff: datetime) -> list[tuple[Version, str, datetime, dict]]:
    times = packument.get("time", {})
    found = []
    for version_str, metadata in packument.get("versions", {}).items():
        sv = semver(version_str)
        timestamp = times.get(version_str)
        if sv is None or not timestamp or not isinstance(metadata, dict):
            continue
        try:
            published = dt(timestamp)
        except ValueError:
            continue
        if published < cutoff:
            found.append((sv, version_str, published, metadata))
    return found


def version_fields(pkg: str, created: str, version_info: tuple) -> dict:
    _, version_str, published, metadata = version_info
    dist = metadata.get("dist") or {}
    return {
        "purl": npm_purl(pkg, version_str),
        "package_created_at": stamp(dt(created)),
        "published_at": stamp(published),
        "unpacked_size": nonnegative_int(dist.get("unpackedSize")),
        "file_count": nonnegative_int(dist.get("fileCount")),
        "dependency_count": mapping_count(metadata.get("dependencies")),
        "optional_dependency_count": mapping_count(metadata.get("optionalDependencies")),
        "peer_dependency_count": mapping_count(metadata.get("peerDependencies")),
        "dev_dependency_count": mapping_count(metadata.get("devDependencies")),
        "bundled_dependency_count": bundle_count(metadata),
    }


def release_fields(versions: list[tuple], months: list[tuple]) -> dict:
    counts = Counter()
    # Distinct version ID
    for _, version, published, _ in versions:
        m = month_index(published, months)
        if m is not None:
            counts[m] += 1
    return {f"package_stable_release_count_M{i}": counts[i] for i in range(12)}


def download_fields(http: Http, package: str, months: list[tuple]) -> dict:
    overall_start = months[-1][0].date()
    overall_end = (months[0][1] - timedelta(days=1)).date()
    url = (f"https://api.npmjs.org/downloads/range/"
           f"{overall_start.isoformat()}:{overall_end.isoformat()}/"
           f"{quote(package, safe='@')}")
    try:
        payload, _ = http.get(url)
        downloads = payload.get("downloads") if isinstance(payload, dict) else None
        if not isinstance(downloads, list):
            raise ValueError("No daily downloads array")
        
        by_day = {}
        for item in downloads:
            try:
                day = date.fromisoformat(item["day"])
                count = nonnegative_int(item["downloads"])
                if count is not None:
                    by_day[day] = count
            except (KeyError, ValueError, TypeError):
                continue

        result = {}
        for i, (start, end) in enumerate(months):
            s_date, e_date = start.date(), end.date()
            month_counts = [c for d, c in by_day.items() if s_date <= d < e_date]
            result[f"package_downloads_M{i}"] = sum(month_counts) if month_counts else 0
            result[f"package_download_days_observed_M{i}"] = len(month_counts)
        return result
    except (requests.RequestException, ValueError) as error:
        LOG.warning("Downloads unavailable for %s (%s)", package, error)
        return {
            **{f"package_downloads_M{i}": None for i in range(12)},
            **{f"package_download_days_observed_M{i}": None for i in range(12)},
        }


def severity(value) -> str:
    s = str(value or "").lower()
    if s == "medium":
        return "moderate"
    return s if s in SEVERITIES else "unknown"


def advisory_qualifies(advisory: dict, package: str, at: datetime) -> bool:
    if advisory.get("type") != "reviewed" or not advisory.get("ghsa_id"):
        return False
    try:
        if dt(advisory["published_at"]) >= at:
            return False
        reviewed = advisory.get("github_reviewed_at")
        if reviewed and dt(reviewed) >= at:
            return False
        withdrawn = advisory.get("withdrawn_at")
        if withdrawn and dt(withdrawn) < at:
            return False
    except (KeyError, ValueError):
        return False
    return any((v.get("package") or {}).get("ecosystem") == "npm" and
               (v.get("package") or {}).get("name") == package
               for v in advisory.get("vulnerabilities") or [])


def retrieve_advisories(http: Http, pkg: str, outcome_end: datetime, max_pages: int) -> list[dict]:
    url = "https://api.github.com/advisories"
    params = {"type": "reviewed", "ecosystem": "npm", "affects": pkg,
              "published": "<" + outcome_end.date().isoformat(),
              "sort": "published", "direction": "desc", "per_page": 100}
    return http.pages(url, params, max_pages)


def advisory_affects_version(advisory: dict, package: str, version: str) -> bool:
    """Whether an advisory explicitly affects one *exact* published npm version.

    GitHub returns NPM-style vulnerable_version_range values such as
    ">= 1.0.0, < 2.0.0". Normalize comparator spacing for semantic_version.NpmSpec.
    An unknown/malformed applicable range is NOT interpreted as "not affected":
    raise an error to prevent producing a false zero feature.
    """
    try:
        parsed_version = Version(version[1:] if version.startswith("v") else version)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid npm version for GHSA matching: {version!r}") from exc

    unresolved = []
    for vulnerability in advisory.get("vulnerabilities") or []:
        if not isinstance(vulnerability, dict):
            continue
        pkg = vulnerability.get("package") or {}
        if (pkg.get("ecosystem") != "npm" or pkg.get("name") != package):
            continue

        raw_range = vulnerability.get("vulnerable_version_range")
        if not isinstance(raw_range, str) or not raw_range.strip():
            unresolved.append("missing vulnerable_version_range")
            continue

        # GitHub ranges use comma-separated, spaced comparators; NpmSpec uses
        # whitespace-separated comparator expressions (preserving any `||`).
        normalized = re.sub(r"\s*,\s*", " ", raw_range.strip())
        normalized = re.sub(r"(<=|>=|<|>|=|~|\^)\s+(?=[vV]?\d)", r"\1", normalized)
        try:
            if NpmSpec(normalized).match(parsed_version):
                return True
        except (ValueError, TypeError) as exc:
            unresolved.append(f"invalid range {raw_range!r}: {exc}")

    if unresolved:
        raise ValueError(
            f"Cannot establish whether GHSA {advisory.get('ghsa_id')} affects "
            f"{package}@{version}: {'; '.join(unresolved)}"
        )
    return False


def historical_ghsa_fields(advisories: list[dict], pkg: str, version: str,
                           cutoff: datetime, months: list[tuple]) -> dict:
    """Lifetime and M0..M11 GHSA *features*, specific to the record version."""
    previous = Counter()
    monthly = Counter()
    seen = set()
    for item in advisories:
        ghsa = item.get("ghsa_id")
        if not ghsa or ghsa in seen or not advisory_qualifies(item, pkg, cutoff):
            continue
        if not advisory_affects_version(item, pkg, version):
            continue
        seen.add(ghsa)
        s = severity(item.get("severity"))
        previous[s] += 1
        i = month_index(dt(item["published_at"]), months)
        if i is not None:
            monthly[(s, i)] += 1

    result = {f"ghsa_{s}_count": previous[s] for s in SEVERITIES}
    for s in SEVERITIES:
        for i in range(12):
            result[f"ghsa_{s}_count_M{i}"] = monthly[(s, i)]
    return result


def package_target_ghsa_fields(advisories: list[dict], pkg: str,
                               cutoff: datetime, end: datetime) -> dict:
    """Future 90-day package-level targets; same for every version input row.

    The professor's model predicts a four-tuple for the *package* rather than
    independent four-tuples for each input version. Retain the fifth unknown
    target in the schema even though it is not in the model's four-tuple.
    """
    targets = Counter()
    seen = set()
    for item in advisories:
        ghsa = item.get("ghsa_id")
        if not ghsa or ghsa in seen:
            continue
        try:
            published = dt(item["published_at"])
        except (KeyError, ValueError):
            continue
        if not cutoff <= published < end:
            continue
        if not advisory_qualifies(item, pkg, end):
            continue
        seen.add(ghsa)
        targets[severity(item.get("severity"))] += 1
    return {f"target_ghsa_{s}_count_90d": targets[s] for s in SEVERITIES}


def github_activity_fields(http: Http, repo: str, months: list[tuple],
                           cutoff: datetime, max_pages: int) -> dict:
    
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "all", "sort": "created", "direction": "asc", "per_page": 100}
    issue_values = []
    for page in range(max_pages):
        response, next_url = http.get(url, params if page == 0 else None)
        if not isinstance(response, list):
            raise ValueError("Invalid GitHub issues result")
        done = False
        for issue in response:
            try:
                created = dt(issue["created_at"])
            except (KeyError, ValueError):
                raise ValueError("Malformed GitHub issue creation timestamp")
            if created >= cutoff:
                done = True
                break  
            issue_values.append(issue)
        if done or not next_url:
            break
        url, params = next_url, {}
    else:
        raise IncompletePagination(f"Issue pagination exceeds {max_pages} pages for {repo}")
    events = http.pages(f"https://api.github.com/repos/{repo}/issues/events",
                        {"per_page": 100}, max_pages)
    by_issue = defaultdict(list)
    for e in events:
        number = (e.get("issue") or {}).get("number")
        if number is not None and e.get("event") in ("closed", "reopened", "merged"):
            try:
                when = dt(e["created_at"])
            except (ValueError, KeyError):
                continue
            if when < cutoff:
                by_issue[number].append((when, e.get("event"), e.get("id")))

    counts = Counter()
    ending_states = [Counter() for _ in months]
    for issue in issue_values:
        issue_number = issue["number"]
        is_pr = bool(issue.get("pull_request"))
        created = dt(issue["created_at"])
        created_m = month_index(created, months)
        if created_m is not None:
            counts[("pull_requests_opened" if is_pr else "issues_opened", created_m)] += 1
        events_for_issue = sorted(by_issue.get(issue_number, []), key=lambda e: (e[0],
                                    {"merged": 0, "closed": 1, "reopened": 2}[e[1]]))
        merged_seconds = {t.replace(microsecond=0) for t, e, _ in events_for_issue if e == "merged"}
        state = "open"
        for m, (_, month_end) in enumerate(months):

            state = "open"
            for when, event, _ in events_for_issue:
                if when >= month_end:
                    break
                if event in ("closed", "merged"):
                    state = "closed"
                elif event == "reopened":
                    state = "open"
            if created < month_end and state == "open":
                metric = "pull_requests_open_at_month_end" if is_pr else "issues_open_at_month_end"
                ending_states[m][metric] += 1

        for when, event, _ in events_for_issue:
            i = month_index(when, months)
            if i is None:
                continue
            if event == "reopened":
                counts[("pull_requests_reopened" if is_pr else "issues_reopened", i)] += 1
            elif event == "merged" and is_pr:
                counts[("pull_requests_merged", i)] += 1
            elif event == "closed":
                if is_pr:
                    if when.replace(microsecond=0) not in merged_seconds:
                        counts[("pull_requests_closed_unmerged", i)] += 1
                else:
                    counts[("issues_closed", i)] += 1

    result = {}
    for i in range(12):
        for metric in GITHUB_FIELDS:
            result[f"{metric}_M{i}"] = (ending_states[i][metric]
                                         if metric.endswith("_at_month_end")
                                         else counts[(metric, i)])
    return result


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packages", nargs="*",
                        help=f"Optional exact names or quoted initials like 'A*'; default list: {DEFAULT_PACKAGES_FILE.name}")
    parser.add_argument("--packages-file",
                        help="Read package names and initials such as A* from this UTF-8 list")
    parser.add_argument("--count-packages", action="store_true",
                        help="Print the number of distinct selected names and exit before extraction")
    parser.add_argument("--output", default="npm_vulnerability_training.csv")
    parser.add_argument("--manifest", default="npm_vulnerability_manifest.json")
    parser.add_argument("--cutoff", default=CUTOFF_DEFAULT,
                        help="First day of the month AFTER M0 (default 2026-07-01)")
    parser.add_argument("--github-activity", action="store_true",
                        help="Fetch complete GitHub issue/PR event histories; expensive")
    parser.add_argument("--max-github-pages", type=int, default=200,
                        help="Limit for repository issue/event pagination (fail to NULL if exceeded)")
    parser.add_argument("--max-advisory-pages", type=int, default=200,
                        help="Limit for GHSA pagination (fail package if exceeded)")
    parser.add_argument("--cache-dir", default=".npm_ghsa_cache")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_package(http: Http, pkg: str, cutoff: datetime, months: list,
                  activity: bool, max_git_pages: int, max_advisory_pages: int) -> tuple[list[dict], dict]:
    window_start = months[-1][0]
    outcome_end = cutoff + timedelta(days=90)
    packument = retrieve_packument(http, pkg)
    created = packument.get("time", {}).get("created")
    if not created or dt(created) >= cutoff:
        raise ValueError(f"Missing/invalid pre-cutoff package creation timestamp: {pkg}")
    available = valid_versions(packument, cutoff)
    if not available:
        return [], {"status": "no_stable_versions_before_cutoff"}
    highest = max(available, key=lambda x: (x[0], x[1]))
    in_window = [x for x in available if window_start <= x[2] < cutoff]
    if not in_window:
        return [], {"status": "no_stable_releases_in_12_month_window",
                    "latest_stable_semver_as_of_cutoff": highest[1]}

    common = release_fields(available, months)
    common.update(download_fields(http, pkg, months))
    advisory = retrieve_advisories(http, pkg, outcome_end, max_advisory_pages)
    # Package-level future targets: one prediction label set per package.
    common.update(package_target_ghsa_fields(advisory, pkg, cutoff, outcome_end))

    repo = repository_name(highest[3].get("repository"), packument.get("repository"),
                           highest[3].get("homepage"), packument.get("homepage"))
    if activity and repo:
        try:
            common.update(github_activity_fields(http, repo, months, cutoff, max_git_pages))
        except (requests.RequestException, ValueError, IncompletePagination) as err:
            LOG.warning("Complete GitHub history unavailable for %s (%s): %s; metrics NULL",
                        pkg, repo, err)
    rows = []
    for info in sorted(in_window, key=lambda x: (x[0], x[1])):
        row = version_fields(pkg, created, info)
        row.update(common)
        # Historical GHSA inputs are version-specific, not common package data.
        row.update(historical_ghsa_fields(advisory, pkg, info[1], cutoff, months))
        rows.append(row)
    return rows, {"status": "ok", "repo": repo,
                  "latest_stable_semver_as_of_cutoff": highest[1],
                  "latest_stable_semver_published_at": stamp(highest[2]),
                  "version_rows_in_window": len(rows),
                  "github_activity_collected": bool(activity and repo and
                     f"issues_opened_M0" in common)}


def names_for_letter(letter: str) -> Iterator[str]:
    """Page over both ASCII cases; npm's key ranges are case-sensitive."""
    url = "https://replicate.npmjs.com/registry/_all_docs"
    with http_session() as session:
        for initial in (letter.upper(), letter.lower()):
            start = initial
            stop = chr(ord(initial) + 1)
            last = None
            while True:
                params = {"startkey": json.dumps(start), "endkey": json.dumps(stop),
                          "inclusive_end": "false", "limit": 10000}
                response = session.get(url, params=params, timeout=60)
                response.raise_for_status()
                page = response.json()
                rows = page.get("rows") if isinstance(page, dict) else None
                if not isinstance(rows, list):
                    raise ValueError(f"Registry returned an invalid page for {letter}*")
                previous_last = last
                for row in rows:
                    name = row.get("id") if isinstance(row, dict) else None
                    if not isinstance(name, str) or not name.startswith(initial) or (last is not None and name < last):
                        raise ValueError(f"Unexpected registry key {name!r} while reading {letter}*")
                    if name != last:
                        last = name
                        if not row.get("value", {}).get("deleted"):
                            yield name
                if last == previous_last:
                    break
                start = last  # startkey is inclusive, so the next page can repeat this name.


def select_packages(names: list[str], packages_file: str | Path | None = None) -> list[str]:
    packages = list(names)
    source = (Path(packages_file) if packages_file is not None else
              DEFAULT_PACKAGES_FILE if not packages else None)
    if source is not None:
        try:
            lines = source.read_text(encoding="utf-8-sig").splitlines()
        except OSError as error:
            raise ValueError(f"Cannot read package list {source}: {error}") from error
        packages.extend(line.strip() for line in lines
                        if line.strip() and not line.lstrip().startswith("#"))
    selected = {}
    expanded = set()
    for entry in packages:
        if "*" not in entry:
            selected[entry] = None
            continue
        if not re.fullmatch(r"[A-Za-z]\*", entry):
            raise ValueError(f"Invalid package selector {entry!r}. Use one initial such as A*.")
        letter = entry[0].lower()
        if letter not in expanded:
            LOG.info("Reading %s* package names from the npm registry", letter.upper())
            for name in names_for_letter(letter):
                selected[name] = None
            expanded.add(letter)
    if not selected:
        raise ValueError("No npm packages found. Add a package name or an initial such as A* to the list.")
    return list(selected)


def main(argv=None) -> int:
    args = args_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    try:
        cutoff = datetime.combine(date.fromisoformat(args.cutoff), datetime.min.time(), UTC)
    except ValueError as exc:
        raise SystemExit("--cutoff must be a valid YYYY-MM-DD date") from exc
    if cutoff.day != 1:
        raise SystemExit("--cutoff must be the FIRST day of a month")
    if args.max_github_pages < 1 or args.max_advisory_pages < 1:
        raise SystemExit("Pagination page limits must be positive")
    try:
        packages = select_packages(args.packages, args.packages_file)
    except (ValueError, requests.RequestException) as error:
        raise SystemExit(str(error)) from error
    if args.count_packages:
        print(f"{len(packages):,} distinct npm package names selected")
        return 0
    LOG.info("Selected %d npm packages (default list: %s)",
             len(packages), DEFAULT_PACKAGES_FILE.name)
    http = Http(Path(args.cache_dir), os.environ.get("GITHUB_TOKEN"))
    months = previous_months(cutoff)
    outcome_end = cutoff + timedelta(days=90)
    report = {"prediction_at": stamp(cutoff),
              "input_window_start": stamp(months[-1][0]),
              "input_window_end_exclusive": stamp(cutoff),
              "label_end_exclusive": stamp(outcome_end),
              "M0": months[0][0].strftime("%Y-%m"),
              "M11": months[11][0].strftime("%Y-%m"),
              "target_order_for_model": list(TARGET_FIELDS[:4]),
              "package_results": {}, "warnings": [
                  "GHSA REST API yields present-day severity/package mappings, not exact historical snapshots.",
                  "Historical GHSA inputs are version-specific; 90-day GHSA targets remain package-level.",
                  "Unparseable or missing GHSA affected-version ranges cause the affected package to be skipped, not silently labeled zero.",
                  "Blank CSV cells represent NULL: unavailable optional data.",
              ]}
    cols = csv_columns()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nrows = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_package(pkg):
        LOG.info("Collecting %s", pkg)
        try:
            rows, metadata = build_package(http, pkg, cutoff, months, args.github_activity,
                                           args.max_github_pages, args.max_advisory_pages)
            return pkg, rows, metadata
        except (requests.RequestException, ValueError, IncompletePagination, KeyError) as exc:
            LOG.error("Skipping %s: required data could not be collected: %s", pkg, exc)
            return pkg, [], {"status": "error", "message": str(exc)}

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        writer.writeheader()
        
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(process_package, pkg): pkg for pkg in packages}
            for future in as_completed(futures):
                pkg, rows, metadata = future.result()
                report["package_results"][pkg] = metadata
                for row in rows:
                    writer.writerow(row)
                    nrows += 1
    report["rows_written"] = nrows
    Path(args.manifest).write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Wrote %d version rows with %d schema fields to %s", nrows, len(cols), output)
    LOG.info("Run manifest: %s", args.manifest)
    if nrows == 0:
        LOG.warning("No dataset rows were produced; check package names and HTTP errors")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
