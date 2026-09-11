import copy
from pathlib import Path
import re
import unittest

from etl.backfill_season import build
from etl.upload_backfill import FEATURE_COLUMNS, PREDICTION_COLUMNS, prepare_upload, upload
from test_backfill import fixture


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.features, self.predictions, self.report = build(fixture(), {"season": 2025}, bootstrap=True)

    def test_payload_is_repeatable_and_preserves_research_status(self):
        first = prepare_upload(self.report, self.features, self.predictions)
        self.assertEqual(first, prepare_upload(self.report, self.features, self.predictions))
        self.assertEqual(first["p_report"]["mode"], "research_only")

    def test_rejects_partial_or_mixed_artifacts(self):
        cases = []
        cases.append((self.report, self.features[:-1], self.predictions))
        bad = copy.deepcopy(self.features)
        bad[0]["run_id"] = "wrong"
        cases.append((self.report, bad, self.predictions))
        bad = copy.deepcopy(self.predictions)
        bad[0]["team_id"] = 99999
        cases.append((self.report, self.features, bad))
        bad = copy.deepcopy(self.features)
        bad[0]["d1_elo"] = float("nan")
        cases.append((self.report, bad, self.predictions))
        cases.append(({**self.report, "feature_version": "d1-v1"}, self.features, self.predictions))
        for args in cases:
            with self.subTest(args=args[0].get("feature_version")):
                with self.assertRaises(ValueError):
                    prepare_upload(*args)

    def test_database_column_contract(self):
        root = Path(__file__).resolve().parents[1] / "supabase/migrations"
        sql = (root / "202609100001_feature_backfills.sql").read_text()
        sql += "\n" + (root / "202609110001_feature_v5.sql").read_text()
        sql += "\n" + (root / "202609110002_feature_v6.sql").read_text()
        sql += "\n" + (root / "202609110003_feature_v7.sql").read_text()
        for table, expected in [("team_feature_snapshots", FEATURE_COLUMNS), ("baseline_predictions", PREDICTION_COLUMNS)]:
            body = sql.split(f"create table public.{table} (", 1)[1].split("\n);", 1)[0]
            if table == "team_feature_snapshots":
                for alter in sql.split("alter table public.team_feature_snapshots")[1:]:
                    body += "\n" + alter.split("do $$", 1)[0]
            columns = set(re.findall(r"^    (\w+) (?:text|integer|bigint|date|double precision|boolean)\b", body, re.MULTILINE))
            if table == "team_feature_snapshots":
                columns.update(re.findall(r"add column (\w+)", body))
            self.assertEqual(columns, expected | {"backfill_id"})

    def test_rpc_success_and_missing_migration(self):
        payload = prepare_upload(self.report, self.features, self.predictions)
        result = {"status": "uploaded", "backfill_id": self.report["source_hash"], "feature_rows": 50}
        class Response:
            status_code = 200
            ok = True
            def json(self):
                return result
        class Session:
            def post(self, url, json, timeout):
                return Response()
        self.assertEqual(upload(Session(), "https://example.invalid", payload), result)
        Response.status_code = 404
        Response.ok = False
        with self.assertRaisesRegex(RuntimeError, "SQL Editor"):
            upload(Session(), "https://example.invalid", payload)
