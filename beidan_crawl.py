#!/usr/bin/env python3
"""beidan_crawl.py — 北单联赛历史数据每日自动爬取

每天定时运行，拉取过去 N 天内所有北单联赛的已完成比赛，
追加到 state/backtest_cache.json，用于扩充回测数据库。
积累足够数据后自动触发参数重新调优。

用法：
  ./venv/bin/python beidan_crawl.py              # 默认拉近 2 天
  ./venv/bin/python beidan_crawl.py --days 7     # 拉近7天
  ./venv/bin/python beidan_crawl.py --retune     # 爬完后触发回测调参
  ./venv/bin/python beidan_crawl.py --full       # 拉全部历史（首次初始化）

cron 示例（每天凌晨2点自动运行）：
  0 2 * * * cd /opt/football-bot && ./venv/bin/python beidan_crawl.py >> /var/log/beidan_crawl.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BOT_DIR    = Path(__file__).parent
STATE_DIR  = BOT_DIR / "state"
CACHE_FILE = STATE_DIR / "backtest_cache.json"
PARAMS_FILE = STATE_DIR / "model_params.json"
API_BASE   = "https://v3.football.api-sports.io"
API_RATE   = 0.4   # 每次请求间隔（秒）
# 触发重新调参所需的新数据量
RETUNE_THRESHOLD = 500

# ── 北单全量联赛配置 ─────────────────────────────────────────────────────────
# 格式: (league_id, seasons_list, label)
# seasons=None 表示动态计算当前+上一赛季
BEIDAN_LEAGUES = [
    # 五大联赛 + 乙级
    (39,  [2023,2024,2025], "英超 Premier League"),
    (40,  [2024,2025],      "英冠 Championship"),
    (61,  [2024,2025],      "法甲 Ligue 1"),
    (62,  [2024,2025],      "法乙 Ligue 2"),
    (78,  [2024,2025],      "德甲 Bundesliga"),
    (79,  [2024,2025],      "德乙 2.Bundesliga"),
    (94,  [2024,2025],      "葡超 Primeira Liga"),
    (135, [2024,2025],      "意甲 Serie A"),
    (136, [2024,2025],      "意乙 Serie B"),
    (140, [2024,2025],      "西甲 La Liga"),
    (141, [2024,2025],      "西乙 La Liga 2"),
    # 其他欧洲
    (88,  [2024,2025],      "荷甲 Eredivisie"),
    (89,  [2024,2025],      "荷乙 Eerste Divisie"),
    (103, [2025,2026],      "挪威 Eliteserien"),
    (113, [2025,2026],      "瑞典 Allsvenskan"),
    (119, [2025],           "丹麦 Superliga"),
    (169, [2024,2025],      "中超 Chinese Super League"),
    (179, [2024,2025],      "苏超 Scottish Premiership"),
    (197, [2024,2025],      "希腊超 Super League 1"),
    (203, [2024,2025],      "土超 Süper Lig Turkey"),
    (207, [2025],           "捷克 Czech Liga"),
    (218, [2024,2025],      "比甲 Belgian Pro League"),
    (244, [2025,2026],      "芬兰超 Veikkausliiga"),
    # 欧洲杯赛
    (2,   [2024,2025],      "欧冠 UEFA Champions League"),
    (3,   [2024,2025],      "欧联 UEFA Europa League"),
    # 美洲
    (71,  [2024,2025],      "巴西乙 Brasileirao B"),
    (128, [2024,2025],      "阿甲 Primera División"),
    (233, [2025,2026],      "美职 MLS"),
    (253, [2025,2026],      "巴甲 Brasileirao A"),
    (262, [2024,2025],      "墨甲 Liga MX"),
    # 亚洲
    (235, [2025,2026],      "韩K联 K League 1"),
    (236, [2024,2025],      "韩K2 K League 2"),
    (239, [2024,2025],      "日职甲 J1 League"),
    (240, [2024,2025],      "日职乙 J2 League"),
    # 大洋洲
    (188, [2024,2025],      "澳洲 A-League"),
    # 中东
    (265, [2024,2025,2026], "沙特联 Saudi Pro League"),
    # 中亚
    (369, [2023,2024,2025], "乌兹别克超 Super League"),
]


def _load_env():
    env_path = BOT_DIR / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def _api_get(endpoint, params, key, retries=3):
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": key}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("errors"):
                raise RuntimeError(str(data["errors"]))
            return data.get("response", [])
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[WARN] {endpoint} {params}: {exc}", file=sys.stderr)
            return []
    return []


def _parse_fixture(r):
    fx = r.get("fixture", {})
    tms = r.get("teams", {})
    gls = r.get("goals", {})
    lg  = r.get("league", {})
    date = fx.get("date") or ""
    try:
        ts = datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()
    except Exception:
        ts = 0.0
    return {
        "fixture_id": fx.get("id"),
        "date":       date,
        "ts":         ts,
        "league_id":  lg.get("id"),
        "season":     lg.get("season"),
        "round":      lg.get("round"),
        "home_id":    tms.get("home", {}).get("id"),
        "home_name":  tms.get("home", {}).get("name"),
        "away_id":    tms.get("away", {}).get("id"),
        "away_name":  tms.get("away", {}).get("name"),
        "goals_home": gls.get("home"),
        "goals_away": gls.get("away"),
    }


def load_cache():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def fetch_since(league_id, season, since_date, key):
    """拉取指定联赛赛季中 since_date 之后完成的比赛。"""
    rows = _api_get("fixtures", {
        "league": league_id, "season": season,
        "status": "FT", "from": since_date,
    }, key)
    time.sleep(API_RATE)
    return [_parse_fixture(r) for r in rows]


def fetch_all(league_id, season, key):
    """拉取指定联赛赛季所有完成比赛（首次初始化用）。"""
    rows = _api_get("fixtures", {"league": league_id, "season": season, "status": "FT"}, key)
    time.sleep(API_RATE)
    return [_parse_fixture(r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",   type=int, default=2,  help="拉最近N天 (default: 2)")
    ap.add_argument("--retune", action="store_true",   help="爬完后触发回测调参")
    ap.add_argument("--full",   action="store_true",   help="拉全部历史（首次初始化）")
    args = ap.parse_args()

    _load_env()
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        print("[ERROR] API_FOOTBALL_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not HAS_REQUESTS:
        print("[ERROR] requests not installed", file=sys.stderr)
        sys.exit(1)

    cache = load_cache()
    total_new = 0
    since_date = (datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}] Beidan crawl started"
          f" ({'full history' if args.full else f'since {since_date}'})")

    for lg_id, seasons, label in BEIDAN_LEAGUES:
        for season in seasons:
            cache_key = f"{lg_id}_{season}"
            existing = {f["fixture_id"] for f in cache.get(cache_key, []) if f.get("fixture_id")}

            if args.full:
                rows = fetch_all(lg_id, season, key)
            else:
                rows = fetch_since(lg_id, season, since_date, key)

            new_rows = [r for r in rows if r.get("fixture_id") and r["fixture_id"] not in existing]
            if new_rows:
                cache.setdefault(cache_key, []).extend(new_rows)
                # 保持按时间排序
                cache[cache_key].sort(key=lambda x: x.get("ts", 0))
                total_new += len(new_rows)
                print(f"  +{len(new_rows):3d} new  {cache_key}  {label}")

    save_cache(cache)
    total = sum(len(v) for v in cache.values())
    print(f"Cache: {total:,} fixtures total | {total_new:,} new this run")

    # 触发重新调参
    if args.retune or total_new >= RETUNE_THRESHOLD:
        print(f"Re-tuning triggered (new={total_new} >= threshold={RETUNE_THRESHOLD})...")
        result = subprocess.run(
            [sys.executable, str(BOT_DIR / "backtest.py"), "--no-cache"],
            capture_output=True, text=True, cwd=str(BOT_DIR),
        )
        if result.returncode == 0:
            print("Re-tune completed.")
        else:
            print("[WARN] Re-tune failed:", result.stderr[-500:], file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
