"""ESportsBattle (football.esportsbattle.com) API client.

Public JSON endpoints observed from the site:
  GET /api/tournaments?dateFrom=YYYY-MM-DD HH:MM&dateTo=...&page=N
      -> {"totalPages": int, "tournaments": [...]}
  GET /api/tournaments/{id}/matches -> [match, ...]

Status ids (observed): tournament 2=upcoming, 3=live, 4=finished;
match 1=scheduled, 3=finished (scores set).

Times in the API are UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE = "https://football.esportsbattle.com/api"
HEADERS = {"User-Agent": "esoccer-tracker/1.0 (personal stats project)"}
TIMEOUT = 20

MATCH_FINISHED = 3
MATCH_SCHEDULED = 1
TOURN_FINISHED = 4


def _get(path: str, params: dict | None = None):
    r = requests.get(f"{BASE}{path}", params=params, headers=HEADERS,
                     timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def fetch_tournaments(date_from: datetime, date_to: datetime,
                      max_pages: int = 20) -> list[dict]:
    """All tournaments whose start falls in [date_from, date_to] (UTC)."""
    out: list[dict] = []
    page = 1
    while page <= max_pages:
        data = _get("/tournaments", {
            "dateFrom": _fmt(date_from), "dateTo": _fmt(date_to),
            "page": page,
        })
        out.extend(data.get("tournaments", []))
        if page >= int(data.get("totalPages", 1)):
            break
        page += 1
    return out


def fetch_tournament_matches(tournament_id: int) -> list[dict]:
    data = _get(f"/tournaments/{tournament_id}/matches")
    return data if isinstance(data, list) else data.get("matches", [])


def league_name(t: dict) -> str:
    lg = t.get("league") or {}
    return lg.get("token_international") or lg.get("token") or "Unknown league"


def matches_to_rows(tournament: dict, matches: list[dict]) -> list[dict]:
    """Normalize raw match dicts to tracker rows (finished + scheduled)."""
    rows = []
    lg = league_name(tournament)
    for m in matches:
        p1 = m.get("participant1") or {}
        p2 = m.get("participant2") or {}
        n1, n2 = p1.get("nickname"), p2.get("nickname")
        if not n1 or not n2:
            continue
        rows.append({
            "match_id": m.get("id"),
            "date": m.get("date"),
            "stage": lg,
            "player_a": n1, "player_b": n2,
            "team_a": ((p1.get("team") or {}).get("token_international")),
            "team_b": ((p2.get("team") or {}).get("token_international")),
            "goals_a": p1.get("score"), "goals_b": p2.get("score"),
            "status_id": m.get("status_id"),
            "source": "ESportsBattle",
        })
    return rows


def split_finished_upcoming(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (finished results df compatible with core.load semantics, upcoming df)."""
    df = pd.DataFrame(rows)
    if df.empty:
        empty = pd.DataFrame(columns=["date", "stage", "player_a", "player_b",
                                      "goals_a", "goals_b", "source"])
        return empty, empty.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    fin = df[(df.status_id == MATCH_FINISHED)
             & df.goals_a.notna() & df.goals_b.notna()].copy()
    if not fin.empty:
        fin["goals_a"] = fin.goals_a.astype(int)
        fin["goals_b"] = fin.goals_b.astype(int)
        fin["total_goals"] = fin.goals_a + fin.goals_b
    up = df[df.status_id == MATCH_SCHEDULED].copy()
    return fin.sort_values("date").reset_index(drop=True), \
        up.sort_values("date").reset_index(drop=True)
