#!/usr/bin/env python3
"""
mc_football.py — 足球赛前蒙特卡洛情景分析工具
基于 API-Football 原始数据，自动计算 V1~V6 变量，输出概率分布和置信区间。

使用方式：
  python mc_football.py --fixture-id 1234567 --api-key YOUR_KEY
  python mc_football.py --manual  # 手动输入模式
  python mc_football.py --json input.json  # 从 JSON 文件读取

输出：
  - 情景矩阵（5×3 表格）
  - 加权概率（主胜/平局/客胜）
  - 标准差和 90% 置信区间
  - 不确定性来源分解
  - 期望回报计算（需提供赔率）
"""

import argparse
import json
import math
import sys
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

BASE_URL = "https://v3.football.api-sports.io"

SCENARIO_WEIGHTS_DEFAULT = [0.12, 0.23, 0.30, 0.23, 0.12]

SCENARIO_LABELS = [
    "S1 主队全面占优",
    "S2 主队温和占优",
    "S3 基准均衡",
    "S4 客队温和占优",
    "S5 客队全面占优",
]

# 每个情景的变量偏移乘数（相对于基准的方向）
# 格式：(V1偏移, V2偏移, V3偏移, V4偏移, V5偏移)  正值=利好主队
SCENARIO_OFFSETS = [
    (+1.0, +1.0, +1.0, +1.0, +1.0),   # S1
    (+0.5, +0.5, +0.3, +0.4, +0.5),   # S2
    ( 0.0,  0.0,  0.0,  0.0,  0.0),   # S3 基准
    (-0.5, -0.5, -0.3, -0.4, -0.5),   # S4
    (-1.0, -1.0, -1.0, -0.5, -1.0),   # S5（V4主场效应不反转）
]

# 各变量对主胜概率调整的权重系数
VARIABLE_IMPACT = {
    "v1_roster":     0.20,   # 阵容完整度
    "v2_form":       0.18,   # 状态动量
    "v3_motivation": 0.12,   # 战意差值
    "v4_home":       0.08,   # 主场效应（已内置于基准，此处为额外波动）
    "v5_tactics":    0.07,   # 战术对位
}


# ─────────────────────────────────────────────
# 数据获取（API-Football）
# ─────────────────────────────────────────────

def api_get(endpoint: str, params: dict, api_key: str) -> Optional[dict]:
    if not HAS_REQUESTS:
        print("[警告] requests 未安装，跳过 API 调用", file=sys.stderr)
        return None
    headers = {
        "x-apisports-key": api_key,
        "Accept": "application/json",
    }
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", [])
    except Exception as e:
        print(f"[警告] API 调用失败 ({endpoint}): {e}", file=sys.stderr)
        return None


def fetch_fixture(fixture_id: int, api_key: str) -> Optional[dict]:
    results = api_get("fixtures", {"id": fixture_id}, api_key)
    return results[0] if results else None


def fetch_team_stats(team_id: int, league_id: int, season: int, api_key: str) -> Optional[dict]:
    results = api_get("teams/statistics", {
        "team": team_id,
        "league": league_id,
        "season": season,
    }, api_key)
    return results if results else None


def fetch_h2h(home_id: int, away_id: int, api_key: str, last: int = 10) -> list:
    results = api_get("fixtures/h2h", {
        "h2h": f"{home_id}-{away_id}",
        "last": last,
    }, api_key)
    return results or []


def fetch_injuries(fixture_id: int, api_key: str) -> list:
    results = api_get("injuries", {"fixture": fixture_id}, api_key)
    return results or []


def fetch_standings(league_id: int, season: int, api_key: str) -> list:
    results = api_get("standings", {"league": league_id, "season": season}, api_key)
    if results and isinstance(results, list) and results[0].get("league", {}).get("standings"):
        return results[0]["league"]["standings"][0]
    return []


# ─────────────────────────────────────────────
# 变量计算（V1~V6）
# ─────────────────────────────────────────────

def calc_v1_roster(injuries: list, team_id: int) -> dict:
    """V1：阵容完整度，基于伤停人数估算"""
    team_injuries = [i for i in injuries if i.get("team", {}).get("id") == team_id]
    key_injuries = sum(
        1 for i in team_injuries
        if i.get("player", {}).get("type") in ["injury", "suspension"]
        and i.get("player", {}).get("reason") not in ["Coach Decision"]
    )
    if key_injuries == 0:
        v1 = 1.00
    elif key_injuries == 1:
        v1 = 0.92
    elif key_injuries == 2:
        v1 = 0.85
    elif key_injuries <= 4:
        v1 = 0.75
    else:
        v1 = 0.60

    pessimistic = max(0.50, v1 - 0.20)
    optimistic  = min(1.00, v1 + 0.08)
    return {"base": v1, "pessimistic": pessimistic, "optimistic": optimistic,
            "key_absences": key_injuries, "source": "API-Football /injuries"}


def calc_v2_form(team_stats: Optional[dict], last_n: int = 5) -> dict:
    """V2：状态动量，基于近期联赛胜率与赛季均值对比"""
    if not team_stats:
        return {"base": 0.0, "pessimistic": -0.15, "optimistic": 0.15,
                "note": "无球队统计数据，使用中性值"}

    fixtures = team_stats.get("fixtures", {})
    total_played = fixtures.get("played", {}).get("total", 1) or 1
    total_wins   = fixtures.get("wins",   {}).get("total", 0)
    season_win_rate = total_wins / total_played

    # 用赛季胜率与0.4基准对比，计算状态偏差
    baseline = 0.40
    deviation = (season_win_rate - baseline) / baseline
    v2 = max(-0.20, min(0.20, deviation * 0.5))

    return {
        "base": round(v2, 3),
        "pessimistic": round(max(-0.20, v2 - 0.12), 3),
        "optimistic":  round(min( 0.20, v2 + 0.12), 3),
        "season_win_rate": round(season_win_rate, 3),
        "source": "API-Football /teams/statistics",
    }


def calc_v3_motivation(home_standing: Optional[dict], away_standing: Optional[dict]) -> dict:
    """V3：战意差值，基于积分榜位置推断"""
    if not home_standing or not away_standing:
        return {"base": 0.0, "pessimistic": -0.08, "optimistic": 0.08,
                "note": "无积分榜数据"}

    home_rank = home_standing.get("rank", 10)
    away_rank = away_standing.get("rank", 10)
    total_teams = 20  # 默认

    # 排名越低（数字越大）越危险 → 战意越高（保级）
    # 排名越高（数字越小）竞争越激烈 → 战意越高（争冠）
    home_pressure = max(0, (total_teams - home_rank) / total_teams * 0.5 +
                           (1 - home_rank / total_teams) * 0.5)
    away_pressure = max(0, (total_teams - away_rank) / total_teams * 0.5 +
                           (1 - away_rank / total_teams) * 0.5)

    v3 = round((home_pressure - away_pressure) * 0.30, 3)
    v3 = max(-0.10, min(0.15, v3))

    return {
        "base": v3,
        "pessimistic": round(max(-0.10, v3 - 0.06), 3),
        "optimistic":  round(min( 0.15, v3 + 0.06), 3),
        "home_rank": home_rank,
        "away_rank": away_rank,
        "source": "API-Football /standings",
    }


def calc_v4_home_effect(team_stats: Optional[dict]) -> dict:
    """V4：主场效应，基于历史主场胜率"""
    if not team_stats:
        return {"base": 0.07, "pessimistic": 0.03, "optimistic": 0.12,
                "note": "无数据，使用默认值"}

    fixtures = team_stats.get("fixtures", {})
    home_played = fixtures.get("played", {}).get("home", 1) or 1
    home_wins   = fixtures.get("wins",   {}).get("home", 0)
    home_win_rate = home_wins / home_played

    # 主场胜率转换为效应值
    v4 = max(0.03, min(0.12, home_win_rate * 0.20))

    return {
        "base": round(v4, 3),
        "pessimistic": round(max(0.03, v4 - 0.03), 3),
        "optimistic":  round(min(0.12, v4 + 0.04), 3),
        "home_win_rate": round(home_win_rate, 3),
        "source": "API-Football /teams/statistics (home)",
    }


def calc_v5_tactics(h2h: list, home_id: int) -> dict:
    """V5：战术对位，基于近期历史交锋胜负"""
    if not h2h:
        return {"base": 0.0, "pessimistic": -0.06, "optimistic": 0.06,
                "note": "无历史交锋数据"}

    recent = h2h[:6]
    home_wins = sum(1 for f in recent
                    if f.get("teams", {}).get("home", {}).get("id") == home_id
                    and f.get("teams", {}).get("home", {}).get("winner") is True)
    away_wins = sum(1 for f in recent
                    if f.get("teams", {}).get("away", {}).get("id") == home_id
                    and f.get("teams", {}).get("away", {}).get("winner") is True)
    total = len(recent)

    if total == 0:
        return {"base": 0.0, "pessimistic": -0.06, "optimistic": 0.06, "note": "交锋数为0"}

    win_rate = (home_wins + away_wins) / total
    v5 = round((win_rate - 0.4) * 0.25, 3)
    v5 = max(-0.08, min(0.08, v5))

    return {
        "base": v5,
        "pessimistic": round(max(-0.08, v5 - 0.04), 3),
        "optimistic":  round(min( 0.08, v5 + 0.04), 3),
        "h2h_count": total,
        "home_team_wins": home_wins + away_wins,
        "source": "API-Football /fixtures/h2h",
    }


# ─────────────────────────────────────────────
# 蒙特卡洛核心计算
# ─────────────────────────────────────────────

def build_scenario_probs(base_home: float, base_draw: float, base_away: float,
                         v1: dict, v2: dict, v3: dict, v4: dict, v5: dict,
                         weights: list) -> list:
    """
    构建5情景的概率矩阵。
    返回 list of dict: [{label, weight, p_home, p_draw, p_away, delta}, ...]
    """
    scenarios = []

    var_ranges = {
        "v1": (v1["pessimistic"], v1["base"], v1["optimistic"]),
        "v2": (v2["pessimistic"], v2["base"], v2["optimistic"]),
        "v3": (v3["pessimistic"], v3["base"], v3["optimistic"]),
        "v4": (v4["pessimistic"], v4["base"], v4["optimistic"]),
        "v5": (v5["pessimistic"], v5["base"], v5["optimistic"]),
    }

    for i, (label, offsets, weight) in enumerate(zip(SCENARIO_LABELS, SCENARIO_OFFSETS, weights)):
        # 计算每个变量在该情景下的值
        def interp(rng, mult):
            pess, base, opt = rng
            if mult >= 0:
                return base + mult * (opt - base)
            else:
                return base + mult * (base - pess)

        sv1 = interp(var_ranges["v1"], offsets[0])
        sv2 = interp(var_ranges["v2"], offsets[1])
        sv3 = interp(var_ranges["v3"], offsets[2])
        sv4 = interp(var_ranges["v4"], offsets[3])
        sv5 = interp(var_ranges["v5"], offsets[4])

        # 计算对主胜概率的总调整量
        v1_impact = (sv1 - v1["base"]) * VARIABLE_IMPACT["v1_roster"]
        v2_impact = sv2 * VARIABLE_IMPACT["v2_form"]
        v3_impact = sv3 * VARIABLE_IMPACT["v3_motivation"]
        v4_impact = (sv4 - v4["base"]) * VARIABLE_IMPACT["v4_home"]
        v5_impact = sv5 * VARIABLE_IMPACT["v5_tactics"]

        delta = v1_impact + v2_impact + v3_impact + v4_impact + v5_impact

        p_home = max(0.05, min(0.85, base_home + delta))
        # 调整量按比例分配给平局和客胜（60%来自客胜，40%来自平局）
        p_draw = max(0.05, min(0.50, base_draw - delta * 0.40))
        p_away = max(0.05, min(0.85, base_away - delta * 0.60))

        # 归一化
        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total

        scenarios.append({
            "label": label,
            "weight": weight,
            "p_home": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away": round(p_away, 4),
            "delta": round(delta, 4),
            "v1_used": round(sv1, 3),
            "v2_used": round(sv2, 3),
            "v3_used": round(sv3, 3),
        })

    return scenarios


def aggregate(scenarios: list) -> dict:
    """加权聚合概率 + 标准差 + 90% 置信区间"""
    p_home = sum(s["weight"] * s["p_home"] for s in scenarios)
    p_draw = sum(s["weight"] * s["p_draw"] for s in scenarios)
    p_away = sum(s["weight"] * s["p_away"] for s in scenarios)

    var_home = sum(s["weight"] * (s["p_home"] - p_home) ** 2 for s in scenarios)
    var_draw = sum(s["weight"] * (s["p_draw"] - p_draw) ** 2 for s in scenarios)
    var_away = sum(s["weight"] * (s["p_away"] - p_away) ** 2 for s in scenarios)

    sigma_home = math.sqrt(var_home)
    sigma_draw = math.sqrt(var_draw)
    sigma_away = math.sqrt(var_away)

    z90 = 1.645  # 90% 置信区间

    def uncertainty_level(sigma):
        if sigma < 0.04:
            return "低（情景收敛，判断稳健）"
        elif sigma < 0.08:
            return "中（正常范围）"
        else:
            return "高（情景分歧大，慎重推荐）"

    return {
        "p_home": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away": round(p_away, 4),
        "sigma_home": round(sigma_home, 4),
        "sigma_draw": round(sigma_draw, 4),
        "sigma_away": round(sigma_away, 4),
        "ci90_home": (round(max(0, p_home - z90 * sigma_home), 3),
                      round(min(1, p_home + z90 * sigma_home), 3)),
        "ci90_draw": (round(max(0, p_draw - z90 * sigma_draw), 3),
                      round(min(1, p_draw + z90 * sigma_draw), 3)),
        "ci90_away": (round(max(0, p_away - z90 * sigma_away), 3),
                      round(min(1, p_away + z90 * sigma_away), 3)),
        "uncertainty_home": uncertainty_level(sigma_home),
        "uncertainty_away": uncertainty_level(sigma_away),
    }


def calc_ev(odds_home: float, odds_draw: float, odds_away: float,
            p_home: float, p_draw: float, p_away: float) -> dict:
    """计算期望回报（Expected Value）"""
    ev_home = odds_home * p_home - 1.0
    ev_draw = odds_draw * p_draw - 1.0
    ev_away = odds_away * p_away - 1.0

    # 去水后隐含概率
    total_implied = 1/odds_home + 1/odds_draw + 1/odds_away
    overround = round((total_implied - 1) * 100, 2)
    implied_home = round((1/odds_home) / total_implied, 4)
    implied_draw = round((1/odds_draw) / total_implied, 4)
    implied_away = round((1/odds_away) / total_implied, 4)

    def verdict(ev):
        if ev > 0.15:  return "✅ 较好价值（>15%）"
        elif ev > 0.10: return "⚠️ 一定价值（>10%）"
        elif ev >= 0:   return "❌ 价值极低（<10%）"
        else:           return "❌ 无价值（负期望）"

    return {
        "ev_home": round(ev_home, 4),
        "ev_draw": round(ev_draw, 4),
        "ev_away": round(ev_away, 4),
        "verdict_home": verdict(ev_home),
        "verdict_draw": verdict(ev_draw),
        "verdict_away": verdict(ev_away),
        "overround_pct": overround,
        "implied_home": implied_home,
        "implied_draw": implied_draw,
        "implied_away": implied_away,
    }


# ─────────────────────────────────────────────
# 报告输出
# ─────────────────────────────────────────────

def print_report(home_name: str, away_name: str,
                 v1h: dict, v2h: dict, v3: dict, v4: dict, v5: dict,
                 v1a: dict, v2a: dict,
                 scenarios: list, agg: dict,
                 ev: Optional[dict] = None,
                 odds: Optional[tuple] = None):
    sep = "─" * 60

    print(f"\n{'='*60}")
    print(f"  ⚽ 蒙特卡洛情景分析报告")
    print(f"  {home_name} vs {away_name}")
    print(f"{'='*60}")

    # 变量摘要
    print(f"\n📊 变量评估（主队 / 客队）")
    print(sep)
    print(f"  V1 阵容完整度：{v1h['base']:.0%} [{v1h['pessimistic']:.0%}~{v1h['optimistic']:.0%}]"
          f"  /  客队：{v1a['base']:.0%} [{v1a['pessimistic']:.0%}~{v1a['optimistic']:.0%}]")
    print(f"  V1 主队缺阵人数：{v1h.get('key_absences','?')} 人  |  客队：{v1a.get('key_absences','?')} 人")
    print(f"  V2 状态动量：主队 {v2h['base']:+.3f} / 客队 {v2a['base']:+.3f}")
    print(f"  V3 战意差值：{v3['base']:+.3f}（主{v3.get('home_rank','?')}名 vs 客{v3.get('away_rank','?')}名）")
    print(f"  V4 主场效应：{v4['base']:.3f}（历史主场胜率 {v4.get('home_win_rate',0):.0%}）")
    print(f"  V5 战术对位：{v5['base']:+.3f}（近{v5.get('h2h_count','?')}场主队胜{v5.get('home_team_wins','?')}场）")

    # 情景矩阵
    print(f"\n🎲 情景矩阵")
    print(sep)
    print(f"  {'情景':<20} {'权重':>6} {'P(主胜)':>8} {'P(平局)':>8} {'P(客胜)':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for s in scenarios:
        print(f"  {s['label']:<20} {s['weight']:>5.0%}  "
              f"{s['p_home']:>7.1%}  {s['p_draw']:>7.1%}  {s['p_away']:>7.1%}")

    # 聚合结果
    print(f"\n🏆 加权聚合结果")
    print(sep)
    print(f"  {'':20} {'主胜':>8} {'平局':>8} {'客胜':>8}")
    print(f"  {'加权概率':<20} {agg['p_home']:>7.1%}  {agg['p_draw']:>7.1%}  {agg['p_away']:>7.1%}")
    print(f"  {'标准差 σ':<20} {agg['sigma_home']:>7.1%}  {agg['sigma_draw']:>7.1%}  {agg['sigma_away']:>7.1%}")
    ci_h = agg['ci90_home']
    ci_d = agg['ci90_draw']
    ci_a = agg['ci90_away']
    print(f"  {'90%置信区间':<20} [{ci_h[0]:.0%}~{ci_h[1]:.0%}]  [{ci_d[0]:.0%}~{ci_d[1]:.0%}]  [{ci_a[0]:.0%}~{ci_a[1]:.0%}]")
    print(f"\n  不确定性（主胜）：{agg['uncertainty_home']}")
    print(f"  不确定性（客胜）：{agg['uncertainty_away']}")

    # 赔率与期望回报
    if ev and odds:
        o_h, o_d, o_a = odds
        print(f"\n💰 赔率与期望回报")
        print(sep)
        print(f"  欧洲赔率：主胜 {o_h}  平局 {o_d}  客胜 {o_a}  "
              f"（超额返还率 +{ev['overround_pct']}%）")
        print(f"  市场隐含概率：主胜 {ev['implied_home']:.1%}  "
              f"平局 {ev['implied_draw']:.1%}  客胜 {ev['implied_away']:.1%}")
        print(f"\n  {'结果':<8} {'赔率':>6} {'我的概率':>9} {'市场概率':>9} {'期望回报':>9} 价值判断")
        print(f"  {'-'*8} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*16}")
        print(f"  {'主胜':<8} {o_h:>6}  {agg['p_home']:>8.1%}  {ev['implied_home']:>8.1%}  "
              f"{ev['ev_home']:>+8.1%}  {ev['verdict_home']}")
        print(f"  {'平局':<8} {o_d:>6}  {agg['p_draw']:>8.1%}  {ev['implied_draw']:>8.1%}  "
              f"{ev['ev_draw']:>+8.1%}  {ev['verdict_draw']}")
        print(f"  {'客胜':<8} {o_a:>6}  {agg['p_away']:>8.1%}  {ev['implied_away']:>8.1%}  "
              f"{ev['ev_away']:>+8.1%}  {ev['verdict_away']}")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────
# 手动输入模式
# ─────────────────────────────────────────────

def manual_input_mode():
    print("\n=== 手动输入模式 ===")
    print("（直接回车使用默认值）\n")

    home_name = input("主队名称: ").strip() or "主队"
    away_name = input("客队名称: ").strip() or "客队"

    def ask_float(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        return float(val) if val else default

    print("\n--- 主队变量 ---")
    v1h_base = ask_float("V1 主队阵容完整度 (0.5~1.0)", 0.85)
    v1h_pess = ask_float("V1 主队悲观值", max(0.50, v1h_base - 0.18))
    v1h_opt  = ask_float("V1 主队乐观值", min(1.00, v1h_base + 0.08))
    v1h = {"base": v1h_base, "pessimistic": v1h_pess, "optimistic": v1h_opt,
           "key_absences": "手动", "source": "手动输入"}

    v2h_base = ask_float("V2 主队状态动量 (-0.2~+0.2)", 0.0)
    v2h = {"base": v2h_base, "pessimistic": max(-0.20, v2h_base-0.10),
           "optimistic": min(0.20, v2h_base+0.10), "source": "手动输入"}

    print("\n--- 客队变量 ---")
    v1a_base = ask_float("V1 客队阵容完整度 (0.5~1.0)", 0.85)
    v1a_pess = ask_float("V1 客队悲观值", max(0.50, v1a_base - 0.18))
    v1a_opt  = ask_float("V1 客队乐观值", min(1.00, v1a_base + 0.08))
    v1a = {"base": v1a_base, "pessimistic": v1a_pess, "optimistic": v1a_opt,
           "key_absences": "手动", "source": "手动输入"}

    v2a_base = ask_float("V2 客队状态动量 (-0.2~+0.2)", 0.0)
    v2a = {"base": v2a_base, "pessimistic": max(-0.20, v2a_base-0.10),
           "optimistic": min(0.20, v2a_base+0.10), "source": "手动输入"}

    print("\n--- 共享变量 ---")
    v3_base = ask_float("V3 战意差值 主队优势 (-0.1~+0.15)", 0.0)
    v3 = {"base": v3_base, "pessimistic": max(-0.10, v3_base-0.05),
          "optimistic": min(0.15, v3_base+0.05), "home_rank": "?",
          "away_rank": "?", "source": "手动输入"}

    v4_base = ask_float("V4 主场效应 (0.03~0.12)", 0.07)
    v4 = {"base": v4_base, "pessimistic": max(0.03, v4_base-0.03),
          "optimistic": min(0.12, v4_base+0.04), "home_win_rate": v4_base/0.2,
          "source": "手动输入"}

    v5_base = ask_float("V5 战术对位 主队优势 (-0.08~+0.08)", 0.0)
    v5 = {"base": v5_base, "pessimistic": max(-0.08, v5_base-0.04),
          "optimistic": min(0.08, v5_base+0.04), "h2h_count": "?",
          "home_team_wins": "?", "source": "手动输入"}

    print("\n--- 基准概率（你的初步判断）---")
    base_home = ask_float("基准主胜概率 (0~1)", 0.40)
    base_draw = ask_float("基准平局概率 (0~1)", 0.28)
    base_away = 1.0 - base_home - base_draw
    print(f"  → 客胜基准概率自动计算为: {base_away:.2f}")

    print("\n--- 赔率（可选，直接回车跳过）---")
    odds_input = input("欧洲赔率 主胜/平局/客胜（如 2.10 3.40 3.60，空格分隔）: ").strip()
    odds = None
    ev = None
    if odds_input:
        try:
            parts = [float(x) for x in odds_input.split()]
            if len(parts) == 3:
                odds = tuple(parts)
        except ValueError:
            pass

    # 合并V1（取主客队较低值影响主胜）
    v1_combined = {
        "base": v1h["base"],
        "pessimistic": v1h["pessimistic"],
        "optimistic":  v1h["optimistic"],
    }

    scenarios = build_scenario_probs(
        base_home, base_draw, base_away,
        v1_combined, v2h, v3, v4, v5,
        SCENARIO_WEIGHTS_DEFAULT,
    )

    # 调整客队阵容影响（反向）
    v1a_impact = (v1a["base"] - 0.85) * (-VARIABLE_IMPACT["v1_roster"])
    for s in scenarios:
        s["p_home"] = round(max(0.05, min(0.85, s["p_home"] + v1a_impact * 0.5)), 4)
        s["p_away"] = round(max(0.05, min(0.85, s["p_away"] - v1a_impact * 0.5)), 4)
        total = s["p_home"] + s["p_draw"] + s["p_away"]
        s["p_home"] /= total
        s["p_draw"] /= total
        s["p_away"] /= total

    agg = aggregate(scenarios)

    if odds:
        ev = calc_ev(odds[0], odds[1], odds[2],
                     agg["p_home"], agg["p_draw"], agg["p_away"])

    print_report(home_name, away_name,
                 v1h, v2h, v3, v4, v5, v1a, v2a,
                 scenarios, agg, ev, odds)
    return agg


# ─────────────────────────────────────────────
# CLI 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="足球赛前蒙特卡洛情景分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python mc_football.py --manual
  python mc_football.py --fixture-id 1234567 --api-key YOUR_KEY
  python mc_football.py --fixture-id 1234567 --api-key YOUR_KEY --odds 2.10 3.40 3.60
  python mc_football.py --json match_data.json --odds 1.85 3.60 4.20
        """
    )
    parser.add_argument("--manual",        action="store_true",    help="手动输入模式")
    parser.add_argument("--fixture-id",    type=int,               help="API-Football 比赛ID")
    parser.add_argument("--api-key",       type=str,               help="API-Football API Key（或设置环境变量 API_FOOTBALL_KEY）")
    parser.add_argument("--json",          type=str,               help="从 JSON 文件读取输入数据")
    parser.add_argument("--odds",          type=float, nargs=3,    help="欧洲赔率 主胜 平局 客胜")
    parser.add_argument("--base-probs",    type=float, nargs=3,    help="基准概率 主胜 平局 客胜（默认 0.40 0.28 0.32）")
    parser.add_argument("--out-json",      type=str,               help="将结果输出到 JSON 文件")

    args = parser.parse_args()

    # 手动模式
    if args.manual:
        manual_input_mode()
        return

    # API 模式
    import os
    api_key = args.api_key or os.environ.get("API_FOOTBALL_KEY", "")

    if args.fixture_id:
        if not api_key:
            print("[错误] 需要提供 --api-key 或设置环境变量 API_FOOTBALL_KEY", file=sys.stderr)
            sys.exit(1)
        if not HAS_REQUESTS:
            print("[错误] 请先安装 requests: pip install requests", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] 获取 fixture {args.fixture_id} 数据...")
        fixture = fetch_fixture(args.fixture_id, api_key)
        if not fixture:
            print("[错误] 未找到该比赛，请检查 fixture_id", file=sys.stderr)
            sys.exit(1)

        home_id    = fixture["teams"]["home"]["id"]
        away_id    = fixture["teams"]["away"]["id"]
        home_name  = fixture["teams"]["home"]["name"]
        away_name  = fixture["teams"]["away"]["name"]
        league_id  = fixture["league"]["id"]
        season     = fixture["league"]["season"]

        print(f"[INFO] 比赛: {home_name} vs {away_name}")
        print(f"[INFO] 获取球队统计、H2H、伤停数据...")

        home_stats = fetch_team_stats(home_id, league_id, season, api_key)
        away_stats = fetch_team_stats(away_id, league_id, season, api_key)
        h2h        = fetch_h2h(home_id, away_id, api_key)
        injuries   = fetch_injuries(args.fixture_id, api_key)
        standings  = fetch_standings(league_id, season, api_key)

        home_standing = next((s for s in standings if s.get("team", {}).get("id") == home_id), None)
        away_standing = next((s for s in standings if s.get("team", {}).get("id") == away_id), None)

        v1h = calc_v1_roster(injuries, home_id)
        v1a = calc_v1_roster(injuries, away_id)
        v2h = calc_v2_form(home_stats)
        v2a = calc_v2_form(away_stats)
        v3  = calc_v3_motivation(home_standing, away_standing)
        v4  = calc_v4_home_effect(home_stats)
        v5  = calc_v5_tactics(h2h, home_id)

    elif args.json:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
        home_name = data.get("home_name", "主队")
        away_name = data.get("away_name", "客队")
        v1h = data["v1_home"]
        v1a = data["v1_away"]
        v2h = data["v2_home"]
        v2a = data["v2_away"]
        v3  = data["v3"]
        v4  = data["v4"]
        v5  = data["v5"]

    else:
        parser.print_help()
        return

    # 基准概率
    if args.base_probs:
        base_home, base_draw, base_away = args.base_probs
    else:
        base_home, base_draw, base_away = 0.40, 0.28, 0.32

    # 构建情景
    v1_combined = {
        "base":        v1h["base"],
        "pessimistic": v1h["pessimistic"],
        "optimistic":  v1h["optimistic"],
    }
    scenarios = build_scenario_probs(
        base_home, base_draw, base_away,
        v1_combined, v2h, v3, v4, v5,
        SCENARIO_WEIGHTS_DEFAULT,
    )

    # 加入客队阵容影响
    v1a_impact = (v1a["base"] - 0.85) * (-VARIABLE_IMPACT["v1_roster"])
    for s in scenarios:
        s["p_home"] = max(0.05, min(0.85, s["p_home"] + v1a_impact * 0.5))
        s["p_away"] = max(0.05, min(0.85, s["p_away"] - v1a_impact * 0.5))
        total = s["p_home"] + s["p_draw"] + s["p_away"]
        s["p_home"] = round(s["p_home"] / total, 4)
        s["p_draw"] = round(s["p_draw"] / total, 4)
        s["p_away"] = round(s["p_away"] / total, 4)

    agg = aggregate(scenarios)

    odds = None
    ev   = None
    if args.odds:
        odds = tuple(args.odds)
        ev = calc_ev(odds[0], odds[1], odds[2],
                     agg["p_home"], agg["p_draw"], agg["p_away"])

    print_report(home_name, away_name,
                 v1h, v2h, v3, v4, v5, v1a, v2a,
                 scenarios, agg, ev, odds)

    if args.out_json:
        result = {
            "home": home_name, "away": away_name,
            "variables": {"v1h": v1h, "v1a": v1a, "v2h": v2h, "v2a": v2a,
                          "v3": v3, "v4": v4, "v5": v5},
            "scenarios": scenarios,
            "aggregated": agg,
            "ev": ev,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果已保存到 {args.out_json}")


if __name__ == "__main__":
    main()
