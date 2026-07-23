#!/usr/bin/env python3
"""backtest.py — 历史回测 + 参数寻优

用法（在服务器 /opt/football-bot 下以 root 运行）：
  ./venv/bin/python backtest.py              # 默认联赛全量回测 + 调参
  ./venv/bin/python backtest.py --no-tune   # 只统计，不调参
  ./venv/bin/python backtest.py --fast      # 仅用 1 个联赛快速验证

回测逻辑：
  1. 拉取历史 finished fixtures（仅一次，有缓存就复用）
  2. 对每场比赛（跳过每队前 WARMUP_ROUNDS 轮），用赛前数据构建 bundle
  3. 调用 strength_model.compute_strength(bundle) 得到预测概率
  4. 与实际结果比较，计算 Brier Score（越低越好，0.333=随机）
  5. 网格搜索关键参数，把最优参数写入 state/model_params.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── 路径（与 app.py 保持一致）────────────────────────────────────────────────
BOT_DIR    = Path(__file__).parent
STATE_DIR  = BOT_DIR / "state"
CACHE_FILE = STATE_DIR / "backtest_cache.json"
RESULTS_FILE = STATE_DIR / "backtest_results.json"
PARAMS_FILE  = STATE_DIR / "model_params.json"

# ── 回测联赛配置 ─────────────────────────────────────────────────────────────
LEAGUES = [
    # ── 五大联赛 + 联赛二级部（北单主力赛事）────────────────────────────
    (39,  2023, "PL-2023"),       (39,  2024, "PL-2024"),       (39,  2025, "PL-2025"),
    (40,  2024, "Championship-2024"), (40, 2025, "Championship-2025"),
    (61,  2024, "Ligue1-2024"),   (61,  2025, "Ligue1-2025"),
    (62,  2024, "Ligue2-2024"),   (62,  2025, "Ligue2-2025"),
    (78,  2024, "BL-2024"),       (78,  2025, "BL-2025"),
    (79,  2024, "BL2-2024"),      (79,  2025, "BL2-2025"),
    (94,  2024, "Primeira-2024"), (94,  2025, "Primeira-2025"),
    (135, 2024, "SerieA-2024"),   (135, 2025, "SerieA-2025"),
    (136, 2024, "SerieB-2024"),   (136, 2025, "SerieB-2025"),
    (140, 2024, "LaLiga-2024"),   (140, 2025, "LaLiga-2025"),
    (141, 2024, "LaLiga2-2024"),  (141, 2025, "LaLiga2-2025"),
    # ── 其他欧洲北单联赛 ────────────────────────────────────────────────
    (88,  2024, "Eredivisie-2024"),   (88,  2025, "Eredivisie-2025"),
    (89,  2024, "Eerste-2024"),       (89,  2025, "Eerste-2025"),
    (103, 2025, "Eliteserien-2025"),  (103, 2026, "Eliteserien-2026"),
    (113, 2025, "Allsvenskan-2025"),  (113, 2026, "Allsvenskan-2026"),
    (119, 2025, "Superliga-DK-2025"),
    (179, 2024, "ScottishPL-2024"),   (179, 2025, "ScottishPL-2025"),
    (197, 2024, "GreekSL-2024"),      (197, 2025, "GreekSL-2025"),
    (203, 2024, "SuperLig-TR-2024"),  (203, 2025, "SuperLig-TR-2025"),
    (207, 2025, "CzechLiga-2025"),
    (218, 2024, "Belgian-2024"),      (218, 2025, "Belgian-2025"),
    (244, 2025, "Veikkausliiga-2025"),(244, 2026, "Veikkausliiga-2026"),
    # ── 欧洲杯赛（单场模式）────────────────────────────────────────────
    (2,   2024, "UCL-2024"),          (2,   2025, "UCL-2025"),
    (3,   2024, "UEL-2024"),          (3,   2025, "UEL-2025"),
    # ── 美洲 ──────────────────────────────────────────────────────────
    (128, 2024, "ArgPrimera-2024"),   (128, 2025, "ArgPrimera-2025"),
    (233, 2025, "MLS-2025"),
    (253, 2025, "BrazilianA-2025"),   (253, 2026, "BrazilianA-2026"),
    (71,  2024, "BrazilianB-2024"),   (71,  2025, "BrazilianB-2025"),
    (262, 2024, "LigaMX-2024"),       (262, 2025, "LigaMX-2025"),
    # ── 亚洲 ──────────────────────────────────────────────────────────
    (239, 2024, "JLeague-2024"),      (239, 2025, "JLeague-2025"),
    (240, 2024, "J2League-2024"),     (240, 2025, "J2League-2025"),
    (235, 2025, "KLeague-2025"),
    (236, 2024, "KLeague2-2024"),     (236, 2025, "KLeague2-2025"),
    (169, 2024, "CSL-2024"),          (169, 2025, "CSL-2025"),
    # ── 大洋洲 ────────────────────────────────────────────────────────
    (188, 2024, "ALeague-2024"),      (188, 2025, "ALeague-2025"),
    # ── 中东 ──────────────────────────────────────────────────────────
    (265, 2024, "SaudiPL-2024"),      (265, 2025, "SaudiPL-2025"),  (265, 2026, "SaudiPL-2026"),
    # ── 中亚（乌兹别克超）────────────────────────────────────────────
    (369, 2023, "UzbekSL-2023"),      (369, 2024, "UzbekSL-2024"),  (369, 2025, "UzbekSL-2025"),
]
FAST_LEAGUES = [(244, 2025, "Veikkausliiga-2025"), (39, 2024, "PL-2024"), (78, 2024, "BL-2024")]

WARMUP_ROUNDS = 6   # 每队前 N 场不参与回测（数据太少）
API_RATE = 0.4      # 每次 API 请求间隔（秒）

# ── 参数网格 ─────────────────────────────────────────────────────────────────
GRID = {
    "dc_rho":       [-0.05, -0.10, -0.13, -0.18, -0.22],
    "form_decay":   [0.65,  0.72,  0.80,  0.87,  0.93],
    "dc_weight":    [0.5,   0.6,   0.7,   0.8,   0.9],
}

# ── 辅助 ─────────────────────────────────────────────────────────────────────
def _api_get(endpoint, params, key, retries=3):
    url = f"https://v3.football.api-sports.io/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": key}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("errors"):
                raise RuntimeError(f"API error: {data['errors']}")
            return data.get("response", [])
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return []

def _load_env():
    env_path = BOT_DIR / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))

def _ts(date_str):
    if not date_str:
        return 0.0
    try:
        t = date_str.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(t).timestamp()
    except Exception:
        return 0.0


# ── 数据拉取与缓存 ────────────────────────────────────────────────────────────
def fetch_league_fixtures(league_id, season, key):
    """拉取某联赛某赛季所有 FT 比赛。"""
    print(f"  Fetching league={league_id} season={season} ...", flush=True)
    rows = _api_get("fixtures", {"league": league_id, "season": season, "status": "FT"}, key)
    time.sleep(API_RATE)
    result = []
    for r in rows:
        fx = r.get("fixture", {})
        tms = r.get("teams", {})
        gls = r.get("goals", {})
        lg  = r.get("league", {})
        result.append({
            "fixture_id": fx.get("id"),
            "date":       fx.get("date"),
            "ts":         _ts(fx.get("date")),
            "league_id":  lg.get("id"),
            "season":     lg.get("season"),
            "round":      lg.get("round"),
            "home_id":    tms.get("home", {}).get("id"),
            "home_name":  tms.get("home", {}).get("name"),
            "away_id":    tms.get("away", {}).get("id"),
            "away_name":  tms.get("away", {}).get("name"),
            "goals_home": gls.get("home"),
            "goals_away": gls.get("away"),
        })
    result.sort(key=lambda x: x["ts"])
    return result


def load_or_fetch_data(leagues, key, force_refresh=False):
    """从缓存加载或重新拉取所有联赛数据。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not force_refresh:
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"Loaded cache: {sum(len(v) for v in cache.values())} fixtures total")
        return cache
    cache = {}
    for lg, se, label in leagues:
        key_str = f"{lg}_{se}"
        rows = fetch_league_fixtures(lg, se, key)
        cache[key_str] = rows
        print(f"  {label}: {len(rows)} fixtures")
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"Cache saved: {sum(len(v) for v in cache.values())} fixtures total")
    return cache


# ── Point-in-Time Bundle 构建 ────────────────────────────────────────────────
def _norm(s):
    if not isinstance(s, str): return ""
    return "".join(c for c in s.casefold() if c.isalnum())


def build_pit_standings(all_fixtures, before_ts, league_id, season):
    """从截止时间前所有已结束比赛重建积分榜。"""
    teams = defaultdict(lambda: {
        "played": 0, "win": 0, "draw": 0, "lose": 0,
        "gf": 0, "ga": 0, "points": 0,
        "home_played": 0, "home_win": 0, "home_gf": 0, "home_ga": 0,
        "away_played": 0, "away_win": 0, "away_gf": 0, "away_ga": 0,
    })
    for fx in all_fixtures:
        if fx["ts"] >= before_ts: continue
        if fx["league_id"] != league_id or fx["season"] != season: continue
        gh = fx.get("goals_home"); ga = fx.get("goals_away")
        if gh is None or ga is None: continue
        hn = fx["home_name"]; an = fx["away_name"]
        # home team
        h = teams[hn]
        h["played"] += 1; h["home_played"] += 1
        h["gf"] += gh; h["ga"] += ga
        h["home_gf"] += gh; h["home_ga"] += ga
        if gh > ga:   h["win"] += 1;  h["home_win"] += 1;  h["points"] += 3
        elif gh == ga: h["draw"] += 1; h["points"] += 1
        else:          h["lose"] += 1
        # away team
        a = teams[an]
        a["played"] += 1; a["away_played"] += 1
        a["gf"] += ga; a["ga"] += gh
        a["away_gf"] += ga; a["away_ga"] += gh
        if ga > gh:   a["win"] += 1;  a["away_win"] += 1;  a["points"] += 3
        elif ga == gh: a["draw"] += 1; a["points"] += 1
        else:          a["lose"] += 1

    rows = []
    for name, s in sorted(teams.items(), key=lambda x: -x[1]["points"]):
        hp = max(s["home_played"], 1); ap = max(s["away_played"], 1)
        rows.append({
            "rank": 0,  # will fill below
            "team": name, "team_id": None,
            "points": s["points"], "goalsDiff": s["gf"] - s["ga"],
            "form": None,
            "all":  {"played": s["played"],       "win": s["win"], "draw": s["draw"], "lose": s["lose"],
                     "goals": {"for": s["gf"],     "against": s["ga"]}},
            "home": {"played": s["home_played"],   "win": s["home_win"],
                     "goals": {"for": s["home_gf"], "against": s["home_ga"]}},
            "away": {"played": s["away_played"],   "win": s["away_win"],
                     "goals": {"for": s["away_gf"], "against": s["away_ga"]}},
            "description": None,
        })
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def build_recent_fixtures(all_fixtures, team_name, before_ts, n=8):
    """从截止时间前取该队最近 n 场。格式与 compact_recent 一致。"""
    tgt = _norm(team_name)
    matches = []
    for fx in reversed(all_fixtures):  # 最新在前
        if fx["ts"] >= before_ts: continue
        hn = _norm(fx["home_name"]); an = _norm(fx["away_name"])
        if tgt not in (hn, an): continue
        gh = fx.get("goals_home"); ga = fx.get("goals_away")
        if gh is None or ga is None: continue
        matches.append({
            "date": fx["date"],
            "league": f"League-{fx['league_id']}",
            "home": fx["home_name"], "away": fx["away_name"],
            "goals": {"home": gh, "away": ga},
            "status": "FT",
        })
        if len(matches) >= n:
            break
    return matches


def build_pit_bundle(fx, all_league_fixtures, all_fixtures_flat):
    """为单场比赛构建 point-in-time bundle。"""
    before_ts = fx["ts"]
    league_id = fx["league_id"]
    season    = fx["season"]
    home_name = fx["home_name"]
    away_name = fx["away_name"]

    standings = build_pit_standings(all_fixtures_flat, before_ts, league_id, season)
    home_recent = build_recent_fixtures(all_fixtures_flat, home_name, before_ts)
    away_recent = build_recent_fixtures(all_fixtures_flat, away_name, before_ts)
    h2h_recent  = [
        {"date": m["date"], "league": m["league"],
         "home": m["home"], "away": m["away"],
         "goals": m["goals"], "status": "FT"}
        for m in (home_recent + away_recent)
        if _norm(m.get("home","")) in (_norm(home_name), _norm(away_name))
        and _norm(m.get("away","")) in (_norm(home_name), _norm(away_name))
    ][:8]

    # home/away statistics — 从积分榜找到对应队
    def stats_for(team_name):
        tgt = _norm(team_name)
        for row in standings:
            if _norm(row["team"]) == tgt:
                hp = max(row["home"]["played"], 1); ap = max(row["away"]["played"], 1)
                return {
                    "form": None, "fixtures": row["all"],
                    "goals": {
                        "for":     {"average": {
                                        "home":  str(round(row["home"]["goals"]["for"] / hp, 2)),
                                        "away":  str(round(row["away"]["goals"]["for"] / ap, 2)),
                                        "total": str(round(row["all"]["goals"]["for"] / max(row["all"]["played"],1), 2))
                                    }, "total": row["all"]["goals"]["for"]},
                        "against": {"average": {
                                        "home":  str(round(row["home"]["goals"]["against"] / hp, 2)),
                                        "away":  str(round(row["away"]["goals"]["against"] / ap, 2)),
                                        "total": str(round(row["all"]["goals"]["against"] / max(row["all"]["played"],1), 2))
                                    }, "total": row["all"]["goals"]["against"]},
                    },
                    "biggest": {}, "clean_sheet": {}, "failed_to_score": {}, "penalty": {},
                }
        return {}

    return {
        "fixture": {
            "fixture_id": fx["fixture_id"],
            "date": fx["date"], "league_id": league_id, "season": season,
            "home": home_name, "away": away_name,
            "home_id": fx["home_id"], "away_id": fx["away_id"],
        },
        "standings":       standings,
        "home_statistics": stats_for(home_name),
        "away_statistics": stats_for(away_name),
        "home_recent":     home_recent,
        "away_recent":     away_recent,
        "head_to_head":    h2h_recent,
        "injuries": [], "lineups": [], "odds_match_winner": [],
    }


# ── Brier Score + 参数评估 ─────────────────────────────────────────────────────
def brier_score(preds):
    """preds = [(p_home,p_draw,p_away, actual)], actual in ('home','draw','away')"""
    if not preds: return 1.0
    total = 0.0
    for ph, pd, pa, actual in preds:
        oh = 1.0 if actual == "home" else 0.0
        od = 1.0 if actual == "draw" else 0.0
        oa = 1.0 if actual == "away" else 0.0
        total += (ph - oh)**2 + (pd - od)**2 + (pa - oa)**2
    return total / len(preds)


def calibration_stats(preds):
    """按概率分箱，检查校准度（理想: 预测50%时实际命中~50%）"""
    bins = [(0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.0)]
    stats = {}
    for lo, hi in bins:
        label = f"{int(lo*100)}-{int(hi*100)}%"
        bucket_pred = bucket_hit = n = 0
        for ph, pd, pa, actual in preds:
            for p, result in [(ph, "home"), (pd, "draw"), (pa, "away")]:
                if lo <= p < hi:
                    n += 1; bucket_pred += p
                    if actual == result: bucket_hit += 1
        stats[label] = {"n": n,
                        "mean_pred": round(bucket_pred/n, 3) if n else None,
                        "actual_rate": round(bucket_hit/n, 3) if n else None}
    return stats


def run_predictions_with_params(all_bundles_results, sm_module, params):
    """用给定参数集对所有 bundle 跑预测，返回 predictions list。
    关键优化：回测期间 patch 掉 Elo 的磁盘 IO（每次调用不读写文件，只做内存计算），
    否则 8000场 × 125参数组 = 100万次磁盘 IO，耗时数小时。
    """
    # 临时替换模块参数
    orig_params = {k: getattr(sm_module, k.upper()) for k in params
                   if hasattr(sm_module, k.upper())}
    # patch 掉 Elo 文件 IO
    orig_load = sm_module._elo_load
    orig_save = sm_module._elo_save
    try:
        for k, v in params.items():
            key = k.upper()
            if hasattr(sm_module, key):
                setattr(sm_module, key, v)
        sm_module._elo_load = lambda: {}          # 返回空字典，不读文件
        sm_module._elo_save = lambda data: None   # 不写文件
        preds = []
        for bundle, actual in all_bundles_results:
            try:
                r = sm_module.compute_strength(bundle)
                if r.get("status") != "ok": continue
                bp = r["base_probs"]
                preds.append((bp["home"], bp["draw"], bp["away"], actual))
            except Exception:
                pass
        return preds
    finally:
        for k, v in orig_params.items():
            setattr(sm_module, k, v)
        sm_module._elo_load = orig_load
        sm_module._elo_save = orig_save


def grid_search(all_bundles_results, sm_module):
    """穷举参数网格，返回 Brier 最低的参数组合。"""
    from itertools import product
    keys = list(GRID.keys())
    vals = [GRID[k] for k in keys]
    best_score = float("inf"); best_params = {}; results = []
    total = 1
    for v in vals: total *= len(v)
    print(f"\nGrid search: {total} combinations on {len(all_bundles_results)} matches ...")
    done = 0
    for combo in product(*vals):
        params = dict(zip(keys, combo))
        preds = run_predictions_with_params(all_bundles_results, sm_module, params)
        score = brier_score(preds)
        results.append((score, params))
        if score < best_score:
            best_score = score; best_params = params
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{total} ... best so far {best_score:.4f}", flush=True)
    results.sort(key=lambda x: x[0])  # sort by Brier score ascending
    print(f"\nBest Brier: {best_score:.4f}  params: {best_params}")
    return best_params, best_score, results[:10]


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Football bot backtest + param tuning")
    ap.add_argument("--no-tune",      action="store_true", help="Only compute metrics, skip grid search")
    ap.add_argument("--fast",         action="store_true", help="Use only1 league for quick test")
    ap.add_argument("--refresh-cache",action="store_true", help="Re-fetch all data from API")
    args = ap.parse_args()

    _load_env()
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        print("ERROR: API_FOOTBALL_KEY not set", file=sys.stderr); sys.exit(1)
    if not HAS_REQUESTS:
        print("ERROR: requests not installed", file=sys.stderr); sys.exit(1)

    # 导入 strength_model（同目录）
    sys.path.insert(0, str(BOT_DIR))
    import strength_model as sm

    leagues = FAST_LEAGUES if args.fast else LEAGUES
    print(f"Backtesting {len(leagues)} league/season combos ...")

    # 1. 拉数据（有缓存就复用）
    cache = load_or_fetch_data(leagues, api_key, force_refresh=args.refresh_cache)

    # 2. 构建所有 bundle + 真实结果
    all_fixtures_flat = [fx for rows in cache.values() for fx in rows]
    # team → 已经出场几轮（用于 warmup 过滤）
    all_bundles_results = []
    team_games_seen: dict[str, int] = defaultdict(int)

    for lg_key, fixtures in cache.items():
        league_id = fixtures[0]["league_id"] if fixtures else None
        season    = fixtures[0]["season"] if fixtures else None
        print(f"\n{lg_key}: {len(fixtures)} matches, building bundles ...")
        skipped = used = 0
        for fx in fixtures:
            hn = fx["home_name"]; an = fx["away_name"]
            # warmup: 每队需要至少 WARMUP_ROUNDS 场经验
            if team_games_seen[f"{hn}_{league_id}_{season}"] < WARMUP_ROUNDS or \
               team_games_seen[f"{an}_{league_id}_{season}"] < WARMUP_ROUNDS:
                team_games_seen[f"{hn}_{league_id}_{season}"] += 1
                team_games_seen[f"{an}_{league_id}_{season}"] += 1
                skipped += 1; continue
            gh = fx.get("goals_home"); ga = fx.get("goals_away")
            if gh is None or ga is None: skipped += 1; continue
            if gh > ga: actual = "home"
            elif gh == ga: actual = "draw"
            else: actual = "away"
            bundle = build_pit_bundle(fx, cache[lg_key], all_fixtures_flat)
            all_bundles_results.append((bundle, actual))
            team_games_seen[f"{hn}_{league_id}_{season}"] += 1
            team_games_seen[f"{an}_{league_id}_{season}"] += 1
            used += 1
        print(f"  used={used} skipped={skipped}")

    print(f"\nTotal bundles: {len(all_bundles_results)}")
    if not all_bundles_results:
        print("No data to backtest."); sys.exit(0)

    # 3. 当前参数的 Brier score
    current_params = {k: getattr(sm, k.upper(), None) for k in GRID}
    current_preds  = run_predictions_with_params(all_bundles_results, sm, current_params)
    current_brier  = brier_score(current_preds)
    print(f"\nCurrent params: {current_params}")
    print(f"Current Brier: {current_brier:.4f} (random=0.333, perfect=0)")

    # 准确率（以最高概率方向为预测）
    correct = sum(1 for ph,pd,pa,actual in current_preds
                  if (ph>=pd and ph>=pa and actual=="home") or
                     (pd>ph and pd>=pa and actual=="draw") or
                     (pa>ph and pa>pd and actual=="away"))
    print(f"Top-1 accuracy: {correct}/{len(current_preds)} = {correct/len(current_preds):.1%}")

    # 校准报告
    calib = calibration_stats(current_preds)
    print("\nCalibration:")
    for bucket, s in calib.items():
        if s["n"]:
            print(f"  {bucket}: pred={s['mean_pred']} actual={s['actual_rate']} n={s['n']}")

    # 4. 网格搜索
    best_params = current_params
    if not args.no_tune:
        best_params, best_brier, top10 = grid_search(all_bundles_results, sm)
        improvement = current_brier - best_brier
        print(f"\nImprovement: {improvement:+.4f}")
        if improvement > 0.001:
            print(f"Saving best params -> {PARAMS_FILE}")
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            PARAMS_FILE.write_text(json.dumps({
                "dc_rho":     best_params.get("dc_rho",   sm.DC_RHO),
                "form_decay": best_params.get("form_decay", sm.FORM_DECAY),
                "dc_weight":  best_params.get("dc_weight",  0.70),
                "brier_score": round(best_brier, 5),
                "n_matches":   len(all_bundles_results),
                "updated": datetime.utcnow().date().isoformat(),
                "top10": [{"score": round(s,4), "params": p} for s,p,*_ in
                           [(x[0],x[1]) for x in top10]],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print("Improvement < 0.001, keeping current params")
    else:
        best_params = current_params

    # 5. 保存结果
    RESULTS_FILE.write_text(json.dumps({
        "current_brier": round(current_brier, 5),
        "n_matches":  len(all_bundles_results),
        "accuracy":   round(correct/len(current_preds), 4) if current_preds else 0,
        "calibration": calib,
        "best_params": best_params,
        "ran_at": datetime.utcnow().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
