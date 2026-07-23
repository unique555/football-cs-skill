#!/usr/bin/env python3
"""weekly_review.py — 每周自动复盘：拉实际结果 → 更新存档 → 校准报告 → 建议重调参

用法：
  ./venv/bin/python weekly_review.py          # 标准复盘
  ./venv/bin/python weekly_review.py --retune # 复盘后触发完整回测+调参
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BOT_DIR      = Path(__file__).parent
STATE_DIR    = BOT_DIR / "state"
PREDICTIONS  = STATE_DIR / "predictions.json"
REVIEWS_DIR  = STATE_DIR / "weekly_reviews"
PARAMS_FILE  = STATE_DIR / "model_params.json"

API_BASE = "https://v3.football.api-sports.io"
API_RATE = 0.4   # 每次请求间隔（秒）

# 样本量达到此数量后才建议重调参
RETUNE_THRESHOLD = 80


def _load_env():
    env_path = BOT_DIR / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def _api_get(endpoint, params, key):
    try:
        r = requests.get(f"{API_BASE}/{endpoint.lstrip('/')}",
                         headers={"x-apisports-key": key},
                         params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception as exc:
        print(f"  API error {endpoint}: {exc}", file=sys.stderr)
        return []


def load_predictions():
    if not PREDICTIONS.exists():
        return []
    entries = []
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def save_predictions(entries):
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def fetch_result(fixture_id, key):
    """从 API 获取已完成比赛结果。返回 'home'/'draw'/'away' 或 None。"""
    rows = _api_get("fixtures", {"id": fixture_id}, key)
    time.sleep(API_RATE)
    for r in rows:
        status = r.get("fixture", {}).get("status", {}).get("short", "")
        if status not in ("FT", "AET", "PEN"):
            return None  # 还未结束
        gls = r.get("goals", {})
        gh = gls.get("home"); ga = gls.get("away")
        if gh is None or ga is None:
            return None
        if gh > ga:   return "home"
        elif gh == ga: return "draw"
        else:          return "away"
    return None


def brier(ph, pd, pa, actual):
    oh = 1.0 if actual == "home" else 0.0
    od = 1.0 if actual == "draw" else 0.0
    oa = 1.0 if actual == "away" else 0.0
    return (ph - oh)**2 + (pd - od)**2 + (pa - oa)**2


def market_brier(entry, actual):
    mh = entry.get("market_home"); md = entry.get("market_draw"); ma = entry.get("market_away")
    if None in (mh, md, ma):
        return None
    return brier(mh, md, ma, actual)


def accuracy_report(resolved):
    """计算各维度准确率和 Brier 分数。"""
    if not resolved:
        return {}
    n = len(resolved)
    model_brier = market_brier_sum = market_n = 0.0
    model_correct = market_correct = 0
    result_counts = defaultdict(int)

    for entry in resolved:
        actual = entry["result"]
        result_counts[actual] += 1
        mp = entry.get("model_base_probs") or {}
        ph = mp.get("home", 1/3); pd = mp.get("draw", 1/3); pa = mp.get("away", 1/3)
        model_brier += brier(ph, pd, pa, actual)
        # top-1 accuracy
        pred = max(("home", ph), ("draw", pd), ("away", pa), key=lambda x: x[1])[0]
        if pred == actual: model_correct += 1
        # market comparison
        mb = market_brier(entry, actual)
        if mb is not None:
            market_brier_sum += mb; market_n += 1
            mpred = max(
                ("home",  entry.get("market_home",  0)),
                ("draw",  entry.get("market_draw",  0)),
                ("away",  entry.get("market_away",  0)),
                key=lambda x: x[1]
            )[0]
            if mpred == actual: market_correct += 1

    return {
        "n": n,
        "model_brier":   round(model_brier / n, 4),
        "model_accuracy": round(model_correct / n, 3),
        "market_brier":  round(market_brier_sum / market_n, 4) if market_n else None,
        "market_accuracy": round(market_correct / market_n, 3) if market_n else None,
        "result_dist": dict(result_counts),
        "baseline_brier": 0.333,  # random classifier
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retune", action="store_true", help="触发重调参（>80 样本时）")
    args = ap.parse_args()

    _load_env()
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        print("ERROR: API_FOOTBALL_KEY not set", file=sys.stderr); sys.exit(1)

    entries = load_predictions()
    print(f"Loaded {len(entries)} prediction records.")

    # 1. 对未解析的预测拉取实际结果
    updated = 0; still_pending = 0
    for entry in entries:
        if entry.get("result"):
            continue  # 已有结果
        fid = entry.get("fixture_id")
        if not fid:
            continue
        result = fetch_result(fid, key)
        if result:
            entry["result"] = result
            entry["model_correct"] = (
                result == max(
                    [("home",  (entry.get("model_base_probs") or {}).get("home",  0)),
                     ("draw",  (entry.get("model_base_probs") or {}).get("draw",  0)),
                     ("away",  (entry.get("model_base_probs") or {}).get("away",  0))],
                    key=lambda x: x[1]
                )[0]
            )
            updated += 1
            print(f"  fixture {fid}: result={result}")
        else:
            still_pending += 1

    if updated:
        save_predictions(entries)
        print(f"Updated {updated} predictions. Still pending: {still_pending}")

    # 2. 计算准确率报告
    resolved = [e for e in entries if e.get("result")]
    stats = accuracy_report(resolved)
    if not resolved:
        print("No resolved predictions to report."); return

    print(f"\n{'='*50}")
    print(f"WEEKLY REVIEW — {datetime.utcnow().date()}")
    print(f"{'='*50}")
    print(f"Resolved predictions : {stats['n']}")
    print(f"Model Brier score    : {stats['model_brier']} (random=0.333)")
    print(f"Model top-1 accuracy : {stats['model_accuracy']:.1%}")
    if stats.get("market_brier"):
        delta = stats["model_brier"] - stats["market_brier"]
        label = "BETTER" if delta < -0.005 else ("WORSE" if delta > 0.005 else "≈ same")
        print(f"Market Brier score   : {stats['market_brier']} → model vs market: {delta:+.4f} ({label})")
        print(f"Market top-1 accuracy: {stats['market_accuracy']:.1%}")
    print(f"Result distribution  : {stats['result_dist']}")
    print(f"Baseline (random)    : 0.333")

    # 3. 保存复盘报告
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    week = datetime.utcnow().strftime("%Y-W%W")
    report_file = REVIEWS_DIR / f"{week}.json"
    report_file.write_text(json.dumps({
        "week": week, "generated": datetime.utcnow().isoformat(), **stats
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {report_file}")

    # 4. 建议/触发调参
    n = stats["n"]
    if n >= RETUNE_THRESHOLD:
        print(f"\n{n} resolved predictions — enough for re-tuning.")
        if args.retune:
            print("Running backtest + param tuning ...")
            subprocess.run(
                [sys.executable, str(BOT_DIR / "backtest.py")],
                cwd=str(BOT_DIR), check=False
            )
        else:
            print("Run with --retune to trigger backtest + param tuning.")
    else:
        print(f"\n{n}/{RETUNE_THRESHOLD} resolved predictions. Need more data before retuning.")


if __name__ == "__main__":
    main()
