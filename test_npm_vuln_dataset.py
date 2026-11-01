import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import npm_vuln_dataset as ds

U = timezone.utc
C = datetime(2026, 7, 1, tzinfo=U)
M = ds.previous_months(C)


def advisory(id, published, severity="high", name="demo", reviewed=None, withdrawn=None):
    return {
        "ghsa_id": id, "type": "reviewed", "published_at": published,
        "github_reviewed_at": reviewed or published,
        "withdrawn_at": withdrawn, "severity": severity,
        "vulnerabilities": [{"package": {"ecosystem": "npm", "name": name}}],
    }


class FakeHttp:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        if "registry.npmjs.org" in url:
            return {
                "time": {"created": "2020-01-01T12:30:00Z",
                         "1.0.0": "2025-07-02T12:00:00Z",
                         "1.2.0": "2025-11-03T12:00:00Z",
                         "1.2.1-beta.1": "2026-06-02T12:00:00Z",
                         "1.1.0": "2026-06-02T12:00:00Z",
                         "2.0.0": "2026-07-12T12:00:00Z"},
                "versions": {
                    "1.0.0": {"dependencies": {"a": "*"},
                              "dist": {"unpackedSize": 123, "fileCount": 4}},
                    "1.2.0": {"dependencies": {"b": "*"}, "bundleDependencies": ["b"]},
                    "1.1.0": {"dependencies": {"c": "*"}},
                    "1.2.1-beta.1": {}, "2.0.0": {},
                },
                "repository": {"url": "git+https://github.com/org/demo.git"},
            }, None
        if "api.npmjs.org" in url:
            parts = url.split("/downloads/range/")[1].split("/")[0].split(":")
            return {"downloads": [{"day": parts[0], "downloads": 5},
                                  {"day": parts[0], "downloads": 6}]}, None
        raise AssertionError(url)

    def pages(self, url, params, max_pages):
        if url.endswith("/advisories"):
            return [advisory("GHSA-old", "2026-06-01T00:00:00Z", "medium"),
                    advisory("GHSA-future", "2026-07-09T00:00:00Z", "critical"),
                    advisory("GHSA-future", "2026-07-09T00:00:00Z", "critical")]
        if url.endswith("/issues/events"):
            return [
                {"issue": {"number": 1}, "event": "closed", "created_at": "2026-06-08T00:00:00Z"},
                {"issue": {"number": 1}, "event": "reopened", "created_at": "2026-06-09T00:00:00Z"},
                {"issue": {"number": 2}, "event": "merged", "created_at": "2026-06-10T00:00:00Z"},
                {"issue": {"number": 2}, "event": "closed", "created_at": "2026-06-10T00:00:00Z"},
            ]
        raise AssertionError(url)


class DatasetTests(unittest.TestCase):
    def test_default_package_file(self):
        names = ds.select_packages([])
        self.assertIn("lodash", names)
        self.assertIn("@babel/core", names)
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("# npm packages to include in the vulnerability training dataset.", names)

    def test_packages_file_override_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "custom_packages.text"
            source.write_text("# Comment\n\nlodash\n@scope/name\nlodash\n", encoding="utf-8")
            self.assertEqual(ds.select_packages([], source), ["lodash", "@scope/name"])
            self.assertEqual(ds.select_packages(["express", "lodash"], source),
                             ["express", "lodash", "@scope/name"])
        self.assertEqual(ds.select_packages(["express", "express"]), ["express"])

    def test_windows(self):
        self.assertEqual(M[0][0].isoformat(), "2026-06-01T00:00:00+00:00")
        self.assertEqual(M[-1][0].isoformat(), "2025-07-01T00:00:00+00:00")
        self.assertEqual((C + timedelta(days=90)).date().isoformat(), "2026-09-29")

    def test_schema_and_purl(self):
        self.assertEqual(len(ds.csv_columns()), 224)
        self.assertEqual(len(set(ds.csv_columns())), 224)
        self.assertEqual(ds.npm_purl("@babel/core", "7.1.0"), "pkg:npm/%40babel/core@7.1.0")
        self.assertIsNone(ds.semver("1.2.3-rc.1"))
        self.assertTrue(ds.semver("1.20.0") > ds.semver("1.3.0"))

    def test_package_rows_semver_vs_publication(self):
        rows, meta = ds.build_package(FakeHttp(), "demo", C, M, False, 100, 100)
        self.assertEqual(len(rows), 3)
        self.assertEqual(meta["latest_stable_semver_as_of_cutoff"], "1.2.0")
        self.assertEqual([x["purl"] for x in rows],
                         ["pkg:npm/demo@1.0.0", "pkg:npm/demo@1.1.0", "pkg:npm/demo@1.2.0"])
        self.assertTrue(all(r["package_stable_release_count_M0"] == 1 for r in rows))
        self.assertTrue(all(r["package_downloads_M0"] == 6 for r in rows))
        self.assertTrue(all(r["package_download_days_observed_M0"] == 1 for r in rows))
        self.assertTrue(all(r["ghsa_moderate_count_M0"] == 1 for r in rows))
        self.assertTrue(all(r["target_ghsa_critical_count_90d"] == 1 for r in rows))
        self.assertEqual(rows[0]["dependency_count"], 1)
        self.assertEqual(rows[2]["bundled_dependency_count"], 1)
        self.assertNotIn("issues_opened_M0", rows[0])

    def test_ghsa_boundary_and_withdrawal(self):
        cases = [
            advisory("a", "2026-06-30T23:59:59Z", "low",
                     withdrawn="2026-07-15T00:00:00Z"),
            advisory("b", "2026-07-01T00:00:00Z", "medium"),
            advisory("c", "2026-09-28T23:59:59Z", "high"),
            advisory("d", "2026-09-29T00:00:00Z", "critical"),
            advisory("e", "2026-05-01T00:00:00Z", "critical",
                     reviewed="2026-08-01T00:00:00Z"),
            advisory("f", "2026-06-01T00:00:00Z", "high", name="other"),
        ]
        row = ds.ghsa_fields(cases, "demo", C, M, C + timedelta(days=90))
        self.assertEqual(row["ghsa_low_count"], 1)  # withdrawn after cutoff
        self.assertEqual(row["ghsa_critical_count"], 0)  # not reviewed at cutoff
        self.assertEqual(row["target_ghsa_moderate_count_90d"], 1)
        self.assertEqual(row["target_ghsa_high_count_90d"], 1)
        self.assertEqual(row["target_ghsa_critical_count_90d"], 0)

    def test_github_events_reconstruct_month_end(self):
        class GitHttp(FakeHttp):
            def get(self, url, params=None):
                if url.endswith("/issues"):
                    return [
                        {"number": 1, "created_at": "2026-05-01T00:00:00Z"},
                        {"number": 2, "created_at": "2026-06-02T00:00:00Z", "pull_request": {"url": "..."}},
                    ], None
                return super().get(url, params)
        r = ds.github_activity_fields(GitHttp(), "org/demo", M, C, 50)
        self.assertEqual(r["issues_opened_M1"], 1)
        self.assertEqual(r["issues_closed_M0"], 1)
        self.assertEqual(r["issues_reopened_M0"], 1)
        self.assertEqual(r["issues_open_at_month_end_M0"], 1)
        self.assertEqual(r["pull_requests_opened_M0"], 1)
        self.assertEqual(r["pull_requests_merged_M0"], 1)
        self.assertEqual(r["pull_requests_closed_unmerged_M0"], 0)
        self.assertEqual(r["pull_requests_open_at_month_end_M0"], 0)


if __name__ == "__main__":
    unittest.main()
