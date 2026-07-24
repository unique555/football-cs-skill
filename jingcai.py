#!/usr/bin/env python3
"""jingcai.py — 国内竞彩足球数据获取与推荐分析

工作 API：
  https://webapi.sporttery.cn/gateway/uniform/football/getMatchCalculatorV1.qry
  ?channel=web&poolCode=had,hhad,crs,ttg,hafu

五种投注类型：
  had   胜平负
  hhad  让球胜平负
  crs   比分
  ttg   进球数
  hafu  半全场
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger("football-bot.jingcai")

# 本地中继缓存文件（由大陆机器运行 jingcai_relay.py 上传）
# 有效期 2 小时；若新鲜则跳过对 sporttery.cn 的直接请求
_CACHE_PATH = Path("/opt/football-bot/state/jingcai_cache.json")
_CACHE_MAX_AGE = 7200  # 秒


def _load_cache() -> list[dict]:
    """读取本地缓存文件，若不存在或过期则返回空列表。"""
    try:
        if not _CACHE_PATH.is_file():
            return []
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        age = time.time() - data.get("timestamp", 0)
        if age > _CACHE_MAX_AGE:
            LOG.info("竞彩缓存已过期（%.0f 秒前）", age)
            return []
        matches = data.get("matches", [])
        LOG.info("竞彩从本地缓存读取：%d 场（%.0f 秒前上传）", len(matches), age)
        return matches
    except Exception as exc:
        LOG.warning("读取竞彩缓存失败: %s", exc)
        return []

_JC_URL = (
    "https://webapi.sporttery.cn/gateway/uniform/football/"
    "getMatchCalculatorV1.qry?channel=web&poolCode=had,hhad,crs,ttg,hafu"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.sporttery.cn/",
    "Accept": "application/json",
}
# 设置 JINGCAI_PROXY 环境变量可启用代理访问（服务器在境外时需要）
# 例如：JINGCAI_PROXY=http://127.0.0.1:7890
_PROXY = os.environ.get("JINGCAI_PROXY", "").strip() or None

# 设置 JINGCAI_RELAY_URL 可通过大陆 Serverless 函数中继访问（优先级高于直连和代理）
# 例如：JINGCAI_RELAY_URL=https://1457975165-9m72an2vb1.ap-guangzhou.tencentscf.com
_RELAY_URL = os.environ.get("JINGCAI_RELAY_URL", "").strip() or None

# ── 比分字段映射 ────────────────────────────────────────────────────────────
# crs 字段格式：s{主}s{客}，例如 s02s01 = 主队2:客队1
_CRS_DISPLAY = {
    "s00s00": "0:0", "s01s00": "1:0", "s02s00": "2:0", "s03s00": "3:0",
    "s04s00": "4:0", "s05s00": "5:0",
    "s01s01": "1:1", "s02s01": "2:1", "s03s01": "3:1", "s04s01": "4:1",
    "s05s01": "5:1",
    "s02s02": "2:2", "s03s02": "3:2", "s04s02": "4:2", "s05s02": "5:2",
    "s03s03": "3:3",
    "s00s01": "0:1", "s00s02": "0:2", "s00s03": "0:3", "s00s04": "0:4",
    "s00s05": "0:5",
    "s01s02": "1:2", "s01s03": "1:3", "s01s04": "1:4",
    "s02s03": "2:3", "s02s04": "2:4",
    "s1sh": "其他主胜", "s1sd": "其他平局", "s1sa": "其他客胜",
}

# 半全场代码 → 中文
_HAFU_LABEL = {
    "hh": "半主胜-全主胜", "hd": "半主胜-全平", "ha": "半主胜-全客胜",
    "dh": "半平-全主胜",   "dd": "半平-全平",   "da": "半平-全客胜",
    "ah": "半客胜-全主胜", "ad": "半客胜-全平", "aa": "半客胜-全客胜",
}

# 进球数代码 → 中文
_TTG_LABEL = {
    "s0": "0球", "s1": "1球", "s2": "2球", "s3": "3球",
    "s4": "4球", "s5": "5球", "s6": "6球", "s7": "7+球",
}


# ── 工具函数 ─────────────────────────────────────────────────────────────────
def _flt(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _ret_rate(*odds) -> float | None:
    """计算返还率 = 1 / sum(1/o)，忽略 None。"""
    vals = [_flt(o) for o in odds if _flt(o) is not None and _flt(o) > 0]
    if len(vals) < 2:
        return None
    return 1.0 / sum(1.0 / v for v in vals)


def _arrow(flag: str) -> str:
    """把 1/-1/0 转成 ↑↓= 符号。"""
    if flag == "1":
        return "↑"
    if flag == "-1":
        return "↓"
    return "="


# ── 主分析函数 ────────────────────────────────────────────────────────────────
def format_jingcai_match(sub: dict) -> str:
    """将一场竞彩比赛格式化为推荐分析文本。"""
    home = sub.get("homeTeamAbbName") or sub.get("homeTeamAllName", "主队")
    away = sub.get("awayTeamAbbName") or sub.get("awayTeamAllName", "客队")
    league = sub.get("leagueAbbName") or sub.get("leagueAllName", "")
    home_rank = sub.get("homeRank", "")
    away_rank = sub.get("awayRank", "")
    match_num = sub.get("matchNumStr", "")
    match_date = sub.get("matchDate", "")
    match_time = sub.get("matchTime", "")[:5] if sub.get("matchTime") else ""

    lines: list[str] = []
    lines.append(f"🏆 竞彩 {match_num} | {league}")
    lines.append(f"⚽ {home}{home_rank} vs {away}{away_rank}")
    lines.append(f"🕐 {match_date} {match_time} 北京时间")
    lines.append("")

    # ── 胜平负 ──────────────────────────────────────────────────────────────
    had = sub.get("had", {})
    if had:
        h, d, a = _flt(had.get("h")), _flt(had.get("d")), _flt(had.get("a"))
        rr = _ret_rate(h, d, a)
        rr_str = f"  返还率 {rr:.1%}" if rr else ""
        hf, df, af = _arrow(had.get("hf","0")), _arrow(had.get("df","0")), _arrow(had.get("af","0"))
        # 隐含概率
        if h and d and a and rr:
            ph = rr / h; pd = rr / d; pa = rr / a
            best = max((ph, "主胜"), (pd, "平局"), (pa, "客胜"))[1]
        else:
            best = "?"
        lines.append("📊 胜平负 (HAD)" + rr_str)
        lines.append(f"  主胜 {h or '-'}({hf})  平 {d or '-'}({df})  客胜 {a or '-'}({af})")
        if best != "?":
            lines.append(f"  💡 最大概率结果：{best}")

    # ── 让球胜平负 ───────────────────────────────────────────────────────────
    hhad = sub.get("hhad", {})
    if hhad:
        h, d, a = _flt(hhad.get("h")), _flt(hhad.get("d")), _flt(hhad.get("a"))
        gl = hhad.get("goalLine") or hhad.get("goalLineValue", "")
        hf, df, af = _arrow(hhad.get("hf","0")), _arrow(hhad.get("df","0")), _arrow(hhad.get("af","0"))
        rr = _ret_rate(h, d, a)
        rr_str = f"  返还率 {rr:.1%}" if rr else ""
        lines.append(f"\n📊 让球胜平负 HHAD (让{gl})" + rr_str)
        lines.append(f"  主胜 {h or '-'}({hf})  平 {d or '-'}({df})  客胜 {a or '-'}({af})")

    # ── 进球数 ───────────────────────────────────────────────────────────────
    ttg = sub.get("ttg", {})
    if ttg:
        vals = [(k, _flt(ttg.get(k)), _arrow(ttg.get(k+"f","0"))) for k in _TTG_LABEL if _flt(ttg.get(k))]
        vals.sort(key=lambda x: x[1])  # 赔率从低到高 = 概率从高到低
        lines.append("\n📊 进球数 (TTG)")
        for key, odds, arr in vals[:4]:  # 展示概率最高的4个
            lines.append(f"  {_TTG_LABEL[key]} {odds}({arr})")

    # ── 半全场 ───────────────────────────────────────────────────────────────
    hafu = sub.get("hafu", {})
    if hafu:
        vals = [(k, _flt(hafu.get(k)), _arrow(hafu.get(k+"f","0") if k+"f" in hafu else hafu.get(k[:2]+"f","0")))
                for k in _HAFU_LABEL if _flt(hafu.get(k))]
        vals.sort(key=lambda x: x[1])
        lines.append("\n📊 半全场 (HAFU)")
        for key, odds, arr in vals[:4]:
            lines.append(f"  {_HAFU_LABEL[key]} {odds}({arr})")

    # ── 比分 ─────────────────────────────────────────────────────────────────
    crs = sub.get("crs", {})
    if crs:
        vals = [(k, _flt(crs.get(k))) for k in _CRS_DISPLAY if _flt(crs.get(k))]
        vals.sort(key=lambda x: x[1])
        lines.append("\n📊 比分 (CRS)")
        for key, odds in vals[:5]:  # 概率最高的5个比分
            lines.append(f"  {_CRS_DISPLAY[key]} → {odds}")

    return "\n".join(lines)


async def fetch_jingcai(client) -> list[dict]:
    """获取今日竞彩比赛数据。client 为 httpx.AsyncClient。

    优先级：
    1. JINGCAI_RELAY_URL（腾讯云/阿里云 Serverless 函数中继，大陆IP直连）
    2. 本地中继缓存文件（由大陆机器运行 jingcai_relay.py 上传）
    3. 直连 API（需要大陆IP，境外服务器会被 geo-block）
    4. JINGCAI_PROXY 代理（备用）
    """
    # 1. 优先使用 Serverless 中继 URL（透明代理 sporttery.cn）
    if _RELAY_URL:
        try:
            r = await client.get(_RELAY_URL, timeout=25)
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                matches: list[dict] = []
                for day_group in data.get("value", {}).get("matchInfoList", []):
                    for sub in day_group.get("subMatchList", []):
                        if sub.get("matchStatus") == "Selling":
                            matches.append(sub)
                LOG.info("竞彩从中继URL获取：%d 场", len(matches))
                return matches
        except Exception as exc:
            LOG.warning("竞彩中继URL请求失败 (%s): %s", _RELAY_URL, exc)

    # 2. 尝试本地缓存文件
    cached = _load_cache()
    if cached:
        return cached

    # 3. 缓存不可用，尝试直连 API（境外服务器通常失败，记录日志即可）
    import httpx as _httpx
    try:
        if _PROXY:
            # 使用独立客户端走代理（境外服务器需要配置 JINGCAI_PROXY）
            async with _httpx.AsyncClient(proxy=_PROXY, follow_redirects=True) as proxy_client:
                r = await proxy_client.get(_JC_URL, headers=_HEADERS, timeout=20)
        else:
            r = await client.get(_JC_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        LOG.warning("竞彩 API 请求失败 (proxy=%s): %s", _PROXY or "none", exc)
        return []

    if not data.get("success"):
        LOG.warning("竞彩 API 返回失败: %s", data.get("errorMessage"))
        return []

    matches: list[dict] = []
    for day_group in data.get("value", {}).get("matchInfoList", []):
        for sub in day_group.get("subMatchList", []):
            if sub.get("matchStatus") == "Selling":
                matches.append(sub)
    return matches


def format_jingcai_list(matches: list[dict]) -> str:
    """格式化今日竞彩比赛列表（简要版，用于 /jingcai 命令）。"""
    if not matches:
        return "今日暂无竞彩在售比赛。"

    lines = [
        f"🎰 今日竞彩比赛（共 {len(matches)} 场）",
        "——————————————————",
    ]
    for i, sub in enumerate(matches, 1):
        home = sub.get("homeTeamAbbName", "主")
        away = sub.get("awayTeamAbbName", "客")
        league = sub.get("leagueAbbName", "")
        num = sub.get("matchNumStr", "")
        t = (sub.get("matchTime") or "")[:5]
        mid = sub.get("matchId", "")
        lines.append(f"{i}. {num} {league} {home} vs {away} {t}")
        lines.append(f"   👉 /jingcai {mid}")
    lines.append("——————————————————")
    lines.append("发送 /jingcai <序号或ID> 查看详细赔率推荐")
    return "\n".join(lines)


def format_jingcai_detail(match_id: str, matches: list[dict]) -> str:
    """根据 match_id 或序号返回详细分析。"""
    # 支持序号（1开始）或直接 match_id
    target = None
    if match_id.isdigit() and 1 <= int(match_id) <= len(matches):
        target = matches[int(match_id) - 1]
    else:
        for m in matches:
            if str(m.get("matchId")) == match_id:
                target = m
                break

    if target is None:
        return f"未找到编号或ID为 {match_id} 的比赛，请先发送 /jingcai 查看列表。"

    return format_jingcai_match(target)
