#!/usr/bin/env python3
"""竞彩数据中继 - 本地抓取并上传到服务器缓存

Usage:
  python jingcai_relay.py          # 抓取并上传到服务器
  python jingcai_relay.py --test   # 只抓取不上传（本地测试）

Windows Task Scheduler:
  08:50 和 16:50 CST 各运行一次，配合服务器 09:00 和 17:00 的 auto_push_loop
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone, timedelta

try:
    import requests
except ImportError:
    print("ERROR: requests not installed", file=sys.stderr)
    sys.exit(1)

# ── 竞彩 API ──────────────────────────────────────────────────────────────
JC_URL = (
    "https://webapi.sporttery.cn/gateway/uniform/football/"
    "getMatchCalculatorV1.qry?channel=web&poolCode=had,hhad,crs,ttg,hafu"
)
JC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.sporttery.cn/",
    "Accept": "application/json",
}

CST = timezone(timedelta(hours=8))

def fetch_jingcai() -> list[dict]:
    """抓取竞彩在售比赛"""
    try:
        r = requests.get(JC_URL, headers=JC_HEADERS, timeout=15)
        if r.status_code >= 400:
            print(f"ERROR: sporttery.cn returned HTTP {r.status_code}", file=sys.stderr)
            return []

        data = r.json()
        if not data.get("success"):
            print(f"ERROR: sporttery.cn returned success=false", file=sys.stderr)
            return []

        matches = []
        for day_group in data.get("value", {}).get("matchInfoList", []):
            for sub in day_group.get("subMatchList", []):
                if sub.get("matchStatus") == "Selling":
                    matches.append(sub)

        print(f"✓ 抓取成功：{len(matches)} 场在售比赛")
        return matches

    except Exception as exc:
        print(f"ERROR: fetch failed - {exc}", file=sys.stderr)
        return []

def upload_to_server(matches: list[dict]) -> bool:
    """通过 SSH 上传到服务器缓存文件"""
    try:
        # 使用已有的 ssh_helper.py 中的 SSH 连接逻辑
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        import ssh_helper

        # 构建缓存数据（加时间戳）
        cache_data = {
            "timestamp": time.time(),
            "datetime": datetime.now(tz=CST).isoformat(),
            "count": len(matches),
            "matches": matches,
        }

        cache_json = json.dumps(cache_data, ensure_ascii=False, indent=2)

        # 上传到服务器
        remote_path = "/opt/football-bot/state/jingcai_cache.json"
        client = ssh_helper.connect()
        try:
            # 使用 cat > file 方式写入
            stdin, stdout, stderr = client.exec_command(f"cat > '{remote_path}'")
            stdin.write(cache_json.encode("utf-8"))
            stdin.channel.shutdown_write()

            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                err = stderr.read().decode("utf-8", errors="replace")
                print(f"ERROR: upload failed (exit {exit_code}): {err}", file=sys.stderr)
                return False

            print(f"✓ 上传成功: {remote_path} ({len(cache_json)} bytes)")
            return True

        finally:
            client.close()

    except Exception as exc:
        print(f"ERROR: upload failed - {exc}", file=sys.stderr)
        return False

def main():
    ap = argparse.ArgumentParser(description="竞彩数据中继")
    ap.add_argument("--test", action="store_true", help="只抓取不上传")
    args = ap.parse_args()

    now = datetime.now(tz=CST)
    print(f"=== 竞彩中继 {now.strftime('%Y-%m-%d %H:%M:%S')} CST ===")

    matches = fetch_jingcai()
    if not matches:
        print("⚠️  未抓取到比赛数据")
        sys.exit(1)

    if args.test:
        print("\n[Test mode] 不上传，输出前3场：")
        for i, m in enumerate(matches[:3], 1):
            home = m.get("homeTeamAbbName", "?")
            away = m.get("awayTeamAbbName", "?")
            league = m.get("leagueAbbName", "?")
            num = m.get("matchNumStr", "?")
            print(f"  {i}. {num} {league} {home} vs {away}")
        return

    # 上传到服务器
    if upload_to_server(matches):
        print("✓ 中继完成")
    else:
        print("✗ 中继失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
