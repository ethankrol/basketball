"""Read-only, paginated export of the existing Supabase tables."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def fetch_table(session, base, table, order, season=None):
    rows = []
    expected = None
    while True:
        params = {"select": "*", "order": order, "limit": 1000, "offset": len(rows)}
        if season is not None:
            params["season"] = f"eq.{season % 100}"
        response = session.get(f"{base}/rest/v1/{table}", params=params, timeout=30)
        if not response.ok:
            raise RuntimeError(f"Read failed for {table}: HTTP {response.status_code}")
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        if expected is not None and total != expected:
            raise ValueError(f"{table} changed during export; retry the export")
        expected = total
        page = response.json()
        rows.extend(page)
        if len(rows) == expected:
            return rows
        if not page or len(rows) > expected:
            raise ValueError(f"Incomplete pagination for {table}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="Four-digit ending year")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2001 <= args.season <= 2099:
        parser.error("Use a four-digit season ending year between 2001 and 2099")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    session = requests.Session()
    key = os.environ["SUPABASE_SERVICE_KEY"]
    session.headers.update({"apikey": key, "Authorization": f"Bearer {key}", "Prefer": "count=exact"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]
    )))
    base = os.environ["SUPABASE_URL"].rstrip("/")
    payload = {"season": args.season, "tables": {}}
    for table, order in [("teams", "team_id"), ("team_spellings", "team_spelling"),
                         ("team_memberships", "team_id,first_season"), ("team_season_status", "team_id,season"),
                         ("games", "date,team,opponent"), ("polls", "week,team")]:
        rows = fetch_table(session, base, table, order, args.season if table in {"games", "polls"} else None)
        payload["tables"][table] = rows
        print(f"{table}: {len(rows)} rows")
    payload["source_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["exported_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
