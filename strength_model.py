#!/usr/bin/env python3
"""strength_model.py v2 — 全面升级版独立赛前强度模型

v2 相比 v1 的核心改进：
  1. 真正的联赛级 DC：用完整积分榜计算联赛均值，规范化攻防强度
  2. 持久化 Elo：跨分析保存至 state/elo_ratings.json，积分榜位次冷启动
  3. 位置加权阵容：G/D/M/F 权重不同，前锋缺阵 >> 替补后卫缺阵
  4. 赛程强度 SOS：对手实力调整近期状态评分
  5. 亚洲盘 / 大小球 / 双方进球 / 最可能比分（全从 DC 比分矩阵派生）
  6. 联赛专属先验：从完整积分榜实时计算，不用固定数字
  7. 对手质量调整：近期胜负加权对手积分

纯标准库，无新依赖。对外只暴露 compute_strength(bundle) -> dict。
"""
from __future__ import annotations
import json, math, os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── 路径 ──────────────────────────────────────────────────────────────────────
_STATE_DIR = Path(os.environ.get("STATE_DIR", "/opt/football-bot/state"))
ELO_FILE    = _STATE_DIR / "elo_ratings.json"
PARAMS_FILE = _STATE_DIR / "model_params.json"

# ── 参数（运行时从 model_params.json 覆盖）────────────────────────────────────
DC_MAX_GOALS     = 10
DC_RHO           = -0.13
FALLBACK_HOME    = 1.45
FALLBACK_AWAY    = 1.15
LAMBDA_FLOOR     = 0.10
LAMBDA_CEIL      = 5.50
FORM_DECAY       = 0.80
SHRINK_PSEUDO    = 4.0
ELO_K            = 24.0
ELO_HOME_ADV     = 60.0
ELO_SEED_SPREAD  = 25.0
ELO_DEFAULT      = 1500.0
DC_WEIGHT        = 0.70   # DC vs Elo 融合比（DC 比例）
AH_LINES         = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
OU_LINES         = (1.5, 2.5, 3.5, 4.5)
POS_ATK = {"G": 0.03, "D": 0.04, "M": 0.06, "F": 0.13}
POS_DEF = {"G": 0.10, "D": 0.09, "M": 0.04, "F": 0.02}
MAX_ATK_PEN = 0.28
MAX_DEF_PEN = 0.22

def _apply_tuned_params() -> None:
    """从 state/model_params.json 加载回测调优后的参数，覆盖上面的默认值。"""
    global DC_RHO, FORM_DECAY, ELO_K, DC_WEIGHT
    try:
        if PARAMS_FILE.is_file():
            p = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
            if "dc_rho"     in p: DC_RHO     = float(p["dc_rho"])
            if "form_decay" in p: FORM_DECAY  = float(p["form_decay"])
            if "elo_k"      in p: ELO_K       = float(p["elo_k"])
            if "dc_weight"  in p: DC_WEIGHT   = float(p["dc_weight"])
    except Exception:
        pass  # 加载失败静默降级，用默认值

_apply_tuned_params()   # 模块导入时立即执行一次

# ── 辅助 ──────────────────────────────────────────────────────────────────────
def _f(v, d=None):
    if v is None or isinstance(v, bool): return d
    try:
        r = float(v); return r if math.isfinite(r) else d
    except: return d

def _i(v, d=0): r=_f(v); return int(r) if r is not None else d

def _n(s):
    if not isinstance(s, str): return ""
    return "".join(c for c in s.casefold() if c.isalnum())

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _dt(s):
    if not isinstance(s, str) or not s: return None
    t = s.strip().replace("Z", "+00:00")
    try: return datetime.fromisoformat(t)
    except:
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try: return datetime.strptime(t, fmt)
            except: pass
    return None

def _ts(dt):
    if dt is None: return float("-inf")
    try: return dt.timestamp()
    except: return float("-inf")

def _dget(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
    return cur

def _shrink(samples, prior, pseudo=SHRINK_PSEUDO):
    num, den = prior*pseudo, pseudo
    for v, w in samples:
        if v is not None and w > 0: num += v*w; den += w
    return num/den if den > 0 else prior

def _weights(n):
    if n <= 0: return []
    raw = [FORM_DECAY**i for i in range(n)]
    s = sum(raw)
    return [w*n/s for w in raw]

# ── 持久化 Elo ─────────────────────────────────────────────────────────────────
def _elo_load():
    try:
        if ELO_FILE.is_file(): return json.loads(ELO_FILE.read_text(encoding="utf-8"))
    except: pass
    return {}

def _elo_save(data):
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = ELO_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(ELO_FILE)
    except: pass

def _elo_exp(ra, rb): return 1.0 / (1.0 + 10.0**((rb-ra)/400.0))

def _elo_seed(standings):
    n = len(standings); seeds = {}
    for row in standings:
        rank = _f(row.get("rank")); tid = str(row.get("team_id") or "")
        if rank and tid: seeds[tid] = ELO_DEFAULT + (n/2.0 - rank) * ELO_SEED_SPREAD
    return seeds

def compute_elo_persistent(home_name, away_name, home_tid, away_tid,
                            home_matches, away_matches, h2h_home,
                            standings, league_key):
    """
    持久化 Elo：加载文件→用近期比赛更新两队评分→保存→返回评分差和胜率。
    home_tid/away_tid: 字符串 team_id（来自 fixture）。
    """
    store    = _elo_load()
    bucket   = store.get(league_key, {})
    seeds    = _elo_seed(standings)
    K        = ELO_K

    def get_r(tid):
        if tid and tid in bucket: return bucket[tid]["rating"]
        return seeds.get(str(tid), ELO_DEFAULT)

    def set_r(tid, r, extra=0):
        if not tid: return
        if tid not in bucket: bucket[tid] = {"rating": r, "games": 1+extra, "updated": datetime.utcnow().date().isoformat()}
        else:
            bucket[tid]["rating"] = r
            bucket[tid]["games"]  = bucket[tid].get("games", 0) + 1
            bucket[tid]["updated"] = datetime.utcnow().date().isoformat()

    def process(matches, subject_tid):
        for m in sorted(matches, key=lambda x: _ts(x.get("dt"))):
            opp_tid = m.get("opponent_tid", "")
            score   = 1.0 if m["result"]=="W" else (0.5 if m["result"]=="D" else 0.0)
            rs = get_r(subject_tid); ro = get_r(opp_tid) if opp_tid else ELO_DEFAULT
            adj = ELO_HOME_ADV if m["is_home"] else -ELO_HOME_ADV
            exp = _elo_exp(rs + adj, ro)
            set_r(subject_tid, rs + K*(score - exp))

    process(home_matches, home_tid)
    process(away_matches, away_tid)

    # H2H 直接修正
    for m in (h2h_home or [])[:6]:
        rs = get_r(home_tid); ro = get_r(away_tid)
        score = 1.0 if m["result"]=="W" else (0.5 if m["result"]=="D" else 0.0)
        adj = ELO_HOME_ADV if m["is_home"] else -ELO_HOME_ADV
        exp = _elo_exp(rs + adj, ro)
        delta = (K*0.5)*(score - exp)
        set_r(home_tid, rs + delta); set_r(away_tid, get_r(away_tid) - delta)

    store[league_key] = bucket
    _elo_save(store)

    rh = get_r(home_tid); ra = get_r(away_tid)
    p_home_win = _elo_exp(rh + ELO_HOME_ADV, ra)
    conf = _clamp((len(home_matches)+len(away_matches)+len(h2h_home or [])) / 14.0, 0.0, 1.0)
    return {"home_elo": round(rh,1), "away_elo": round(ra,1),
            "p_home_elo": round(p_home_win,4), "confidence": round(conf,3)}

# ── 联赛参数 ───────────────────────────────────────────────────────────────────
def league_params(standings):
    n = len(standings)
    if n < 2:
        return {"home_avg": FALLBACK_HOME, "away_avg": FALLBACK_AWAY, "n": n, "src": "fallback"}
    hg = hm = ag = am = 0
    for row in standings:
        h = row.get("home") or {}; a = row.get("away") or {}
        hm += _i(_dget(h,"played")); hg += _i(_dget(h,"goals","for"))
        am += _i(_dget(a,"played")); ag += _i(_dget(a,"goals","for"))
    ha = hg/hm if hm else FALLBACK_HOME
    aa = ag/am if am else FALLBACK_AWAY
    return {"home_avg": round(ha,4), "away_avg": round(aa,4), "n": n, "src": "standings"}

def attack_defense(standings, team_name, is_home, ha, aa):
    tgt = _n(team_name)
    for row in standings:
        if _n(row.get("team") or "") != tgt: continue
        vk = "home" if is_home else "away"
        v = row.get(vk) or {}
        vp = _f(_dget(v,"played")) or 0
        vgf = _f(_dget(v,"goals","for")) or 0
        vga = _f(_dget(v,"goals","against")) or 0
        avg = ha if is_home else aa
        atk = (vgf/vp/avg) if vp>0 and avg>0 else 1.0
        df  = (vga/vp/avg) if vp>0 and avg>0 else 1.0
        # 全赛季辅助
        al = row.get("all") or {}
        ap2 = _f(_dget(al,"played")) or 0
        ag2 = _f(_dget(al,"goals","for")) or 0
        agaf = _f(_dget(al,"goals","against")) or 0
        a_atk = (ag2/ap2/avg) if ap2>0 else 1.0
        a_df  = (agaf/ap2/avg) if ap2>0 else 1.0
        w = _clamp(vp/4.0, 0.0, 1.0)
        return {"atk": round(_clamp(w*atk+(1-w)*a_atk,0.3,3.0),4),
                "def": round(_clamp(w*df +(1-w)*a_df, 0.3,3.0),4),
                "vp": int(vp), "src": "standings"}
    return {"atk":1.0,"def":1.0,"vp":0,"src":"missing"}

# ── 近期比赛提取（带 opponent_tid）────────────────────────────────────────────
def extract_matches(records, team_name, standings):
    tgt = _n(team_name)
    if not tgt or not isinstance(records, list): return []
    id_map = {_n(r.get("team") or ""): str(r.get("team_id") or "")
              for r in (standings or []) if r.get("team_id")}
    out = []
    for rec in records:
        if not isinstance(rec, dict): continue
        if (rec.get("status") or "").upper() not in ("FT","AET","PEN",""): continue
        hn = _n(rec.get("home")); an = _n(rec.get("away"))
        g = rec.get("goals") or {}
        gh = _f(g.get("home")); ga = _f(g.get("away"))
        if gh is None or ga is None: continue
        if tgt == hn:   is_h, gf, gg, opp, on = True,  gh, ga, rec.get("away"), an
        elif tgt == an: is_h, gf, gg, opp, on = False, ga, gh, rec.get("home"), hn
        else: continue
        res = "W" if gf>gg else ("L" if gf<gg else "D")
        out.append({"dt": _dt(rec.get("date")), "is_home": is_h, "gf": gf, "ga": gg,
                    "opponent": opp, "opponent_norm": on,
                    "opponent_tid": id_map.get(on,""), "result": res})
    out.sort(key=lambda m: _ts(m.get("dt")), reverse=True)
    return out

# ── SOS ────────────────────────────────────────────────────────────────────────
def compute_sos(matches, standings):
    if not standings: return {"sos":1.0,"n":0}
    tp = sum(_f(r.get("points"),0) for r in standings)
    nt = len(standings)
    lavg = tp/nt if nt else 1.0
    pts = {_n(r.get("team") or ""): _f(r.get("points"), lavg) for r in standings}
    ws = _weights(len(matches)); total_w = num = 0.0
    for m, w in zip(matches, ws):
        num += pts.get(m.get("opponent_norm",""), lavg) * w; total_w += w
    if total_w <= 0: return {"sos":1.0,"n":0}
    sos = _clamp(num/total_w/max(lavg,0.01), 0.7, 1.4)
    return {"sos": round(sos,3), "avg_opp_pts": round(num/total_w,1),
            "league_avg_pts": round(lavg,1), "n": len(matches)}

# ── 进失球率估计 ───────────────────────────────────────────────────────────────
def goal_rates(matches, stats, is_home, sos, ha, aa):
    n = len(matches); ws = _weights(n)
    # 衰减近期（SOS 修正：硬赛程中同样成绩 ≈ 更强）
    sfx = max(sos, 0.7)
    rgf = [(m["gf"]/sfx, w) for m, w in zip(matches, ws)]
    rga = [(m["ga"]*sfx, w) for m, w in zip(matches, ws)]
    same = [(m, w) for m, w in zip(matches, ws) if m["is_home"] == is_home]
    rgf += [(m["gf"]/sfx, w*0.7) for m, w in same]
    rga += [(m["ga"]*sfx, w*0.7) for m, w in same]
    # 赛季统计
    gf_k = _n("for"); ga_k = _n("against")
    vk = "home" if is_home else "away"
    sgf = _f(_dget(stats,"goals","for","average",vk)) or _f(_dget(stats,"goals","for","average","total"))
    sga = _f(_dget(stats,"goals","against","average",vk)) or _f(_dget(stats,"goals","against","average","total"))
    sw = 3.0
    if sgf: rgf.append((sgf, sw))
    if sga: rga.append((sga, sw))
    pg = ha if is_home else aa; pga = aa if is_home else ha
    return {"gf": round(_clamp(_shrink(rgf,pg),LAMBDA_FLOOR,LAMBDA_CEIL),3),
            "ga": round(_clamp(_shrink(rga,pga),LAMBDA_FLOOR,LAMBDA_CEIL),3),
            "n": n, "sos": round(sfx,3)}

# ── 位置加权阵容量化 ────────────────────────────────────────────────────────────
def lineup_v2(injuries, lineups, team_name):
    tgt = _n(team_name)
    # 从 lineups 建 name→pos 映射（新版含 pos 字段）
    pos_map = {}
    if isinstance(lineups, list):
        for item in lineups:
            if not isinstance(item, dict) or _n(item.get("team")) != tgt: continue
            for e in (item.get("startXI") or []) + (item.get("substitutes") or []):
                if isinstance(e, dict):
                    nm = _n(e.get("name") or ""); pos = e.get("pos")
                    if nm: pos_map[nm] = (pos or "").upper()[:1] or None
    absences = []
    if isinstance(injuries, list):
        for item in injuries:
            if not isinstance(item, dict) or _n(item.get("team")) != tgt: continue
            reason = item.get("reason") if isinstance(item.get("reason"),str) else ""
            if reason.strip().lower() in ("coach decision","coach's decision","rest"): continue
            itype = item.get("type") if isinstance(item.get("type"),str) else ""
            if not (itype.lower().strip() in ("injured","injury","suspension","suspended") or reason):
                continue
            pname = item.get("player") if isinstance(item.get("player"),str) else ""
            pos = pos_map.get(_n(pname))
            absences.append({"name": pname, "pos": pos, "reason": reason or itype})

    atk_pen = sum(POS_ATK.get((a.get("pos") or "").upper()[:1] or "X", POS_ATK.get("M",0.06))
                  for a in absences)
    def_pen = sum(POS_DEF.get((a.get("pos") or "").upper()[:1] or "X", POS_DEF.get("M",0.04))
                  for a in absences)
    atk_mult = _clamp(1.0 - atk_pen, 1.0 - MAX_ATK_PEN, 1.0)
    def_mult = _clamp(1.0 - def_pen, 1.0 - MAX_DEF_PEN, 1.0)
    pub = False; xi_n = None
    if isinstance(lineups, list):
        for item in lineups:
            if isinstance(item, dict) and _n(item.get("team")) == tgt:
                pub = True; xi_n = len(item.get("startXI") or [])
    return {"absences": len(absences), "atk_mult": round(atk_mult,3),
            "def_mult": round(def_mult,3), "lineup_pub": pub, "xi": xi_n,
            "detail": [f"{a['name']}({a.get('pos') or '?'})" for a in absences[:6]]}

# ── Dixon-Coles 比分矩阵 + 全量指标 ─────────────────────────────────────────────
def _pois(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)

def _tau(i, j, lam, mu, rho):
    if i==0 and j==0: return 1.0 - lam*mu*rho
    if i==0 and j==1: return 1.0 + lam*rho
    if i==1 and j==0: return 1.0 + mu*rho
    if i==1 and j==1: return 1.0 - rho
    return 1.0

def dc_matrix_metrics(lam_h, lam_a, rho=DC_RHO, mg=DC_MAX_GOALS):
    lam_h = _clamp(lam_h, LAMBDA_FLOOR, LAMBDA_CEIL)
    lam_a = _clamp(lam_a, LAMBDA_FLOOR, LAMBDA_CEIL)
    n = mg + 1
    mat = [[0.0]*n for _ in range(n)]; tot = 0.0
    for i in range(n):
        for j in range(n):
            p = _pois(i,lam_h)*_pois(j,lam_a)*max(0.0,_tau(i,j,lam_h,lam_a,rho))
            mat[i][j] = p; tot += p
    if tot > 0:
        for i in range(n):
            for j in range(n): mat[i][j] /= tot
    ph = pd = pa = btts = 0.0
    over = {l: 0.0 for l in OU_LINES}
    scores = []
    for i in range(n):
        for j in range(n):
            p = mat[i][j]
            if i > j:  ph += p
            elif i==j: pd += p
            else:       pa += p
            if i>0 and j>0: btts += p
            for l in OU_LINES:
                if i+j > l: over[l] += p
            scores.append((p, i, j))
    # Asian Handicap
    ah = {}
    for line in AH_LINES:
        ahp = 0.0
        for i in range(n):
            for j in range(n):
                diff = i - j
                if abs(line - round(line)) < 0.01:  # 整数线
                    if diff > line:  ahp += mat[i][j]
                    elif diff == int(line): ahp += mat[i][j]*0.5
                else:  # 半球线
                    if diff > line: ahp += mat[i][j]
        ah[f"{line:+.1f}"] = round(ahp, 4)
    scores.sort(reverse=True)
    top5 = [{"score": f"{i}-{j}", "prob": round(p,4)} for p,i,j in scores[:5]]
    return {"p_home": round(ph,4), "p_draw": round(pd,4), "p_away": round(pa,4),
            "lambda_home": round(lam_h,3), "lambda_away": round(lam_a,3),
            "xg_total": round(lam_h+lam_a,3),
            "most_likely_score": top5[0]["score"] if top5 else "1-1",
            "top5_scores": top5,
            "over": {f"o{str(l).replace('.','_')}": round(v,4) for l,v in over.items()},
            "btts": round(btts,4),
            "ah": ah, "rho": rho}

# ── 盘口信号 ────────────────────────────────────────────────────────────────────
def odds_signal(odds_list):
    if not isinstance(odds_list, list) or not odds_list:
        return {"available": False}
    hs, ds, as_, ts = [], [], [], []
    for m in odds_list:
        if not isinstance(m, dict): continue
        nv = m.get("no_vig") or {}
        h = _f(nv.get("home")); d = _f(nv.get("draw")); a = _f(nv.get("away"))
        if None in (h,d,a): continue
        hs.append(h); ds.append(d); as_.append(a)
        dt2 = _dt(m.get("updated"))
        if dt2: ts.append(dt2)
    if not hs: return {"available": False}
    def sp(v): return round(max(v)-min(v), 4)
    cons = {"home": round(sum(hs)/len(hs),4),
            "draw": round(sum(ds)/len(ds),4),
            "away": round(sum(as_)/len(as_),4)}
    mx = max(sp(hs), sp(ds), sp(as_))
    dis = "high" if mx>0.06 else ("moderate" if mx>0.03 else "low")
    return {"available": True, "n_books": len(hs), "consensus": cons,
            "spread": {"home": sp(hs), "draw": sp(ds), "away": sp(as_)},
            "disagreement": dis,
            "latest": max(ts).isoformat() if ts else None}

# ── 主函数 ─────────────────────────────────────────────────────────────────────
def compute_strength(bundle: dict) -> dict:
    try:
        return _compute(bundle)
    except Exception as exc:
        import traceback
        return {"status": "unavailable", "error": type(exc).__name__,
                "trace": traceback.format_exc()[-800:]}

def _compute(bundle: dict) -> dict:
    fixture   = bundle.get("fixture") or {}
    home_name = fixture.get("home") or "home"
    away_name = fixture.get("away") or "away"
    home_tid  = str(fixture.get("home_id") or "")
    away_tid  = str(fixture.get("away_id") or "")
    league_id = fixture.get("league_id") or ""
    season    = fixture.get("season") or ""
    lg_key    = f"league_{league_id}_season_{season}"
    standings = bundle.get("standings") or []
    stats_h   = bundle.get("home_statistics") or {}
    stats_a   = bundle.get("away_statistics") or {}

    # 1. 联赛参数
    lp = league_params(standings)
    ha = lp["home_avg"]; aa = lp["away_avg"]

    # 2. 近期比赛提取
    home_m = extract_matches(bundle.get("home_recent"), home_name, standings)
    away_m = extract_matches(bundle.get("away_recent"), away_name, standings)
    h2h_m  = extract_matches(bundle.get("head_to_head"), home_name, standings)

    # 3. SOS
    sos_h = compute_sos(home_m, standings)
    sos_a = compute_sos(away_m, standings)

    # 4. 进失球率（近期衰减 + SOS + 赛季统计）
    gr_h = goal_rates(home_m, stats_h, True,  sos_h.get("sos",1.0), ha, aa)
    gr_a = goal_rates(away_m, stats_a, False, sos_a.get("sos",1.0), ha, aa)

    # 5. 积分榜攻防强度（联赛级归一化）
    ad_h = attack_defense(standings, home_name, True,  ha, aa)
    ad_a = attack_defense(standings, away_name, False, ha, aa)

    # 6. λ = 本队进攻强度 × 对手防守强度 × 联赛均值，再乘阵容/SOS
    lup_h = lineup_v2(bundle.get("injuries"), bundle.get("lineups"), home_name)
    lup_a = lineup_v2(bundle.get("injuries"), bundle.get("lineups"), away_name)

    # 混合两种 λ 估计：一种来自积分榜强度，一种来自近期进球率
    if ad_h.get("vp", 0) >= 4 and ad_a.get("vp", 0) >= 4:
        lam_h_lg = ad_h["atk"] * ad_a["def"] * ha
        lam_a_lg = ad_a["atk"] * ad_h["def"] * aa
        w_lg = min((ad_h["vp"] + ad_a["vp"]) / 16.0, 0.7)
    else:
        lam_h_lg = lam_a_lg = None; w_lg = 0.0

    lam_h_rr = gr_h["gf"] * gr_a["ga"] / max(ha, 0.01)
    lam_a_rr = gr_a["gf"] * gr_h["ga"] / max(aa, 0.01)

    if lam_h_lg is not None:
        lam_h = w_lg * lam_h_lg + (1 - w_lg) * lam_h_rr
        lam_a = w_lg * lam_a_lg + (1 - w_lg) * lam_a_rr
    else:
        lam_h, lam_a = lam_h_rr, lam_a_rr

    # 阵容调整
    lam_h *= lup_h["atk_mult"]
    lam_a *= lup_a["atk_mult"]
    # 对手防守受缺阵影响（对方前锋缺阵→守方容易）
    lam_h *= (2.0 - lup_a["def_mult"]) / 1.0  # def_mult<1 → 对手防守弱 → lam_h 升
    lam_a *= (2.0 - lup_h["def_mult"]) / 1.0

    lam_h = _clamp(lam_h, LAMBDA_FLOOR, LAMBDA_CEIL)
    lam_a = _clamp(lam_a, LAMBDA_FLOOR, LAMBDA_CEIL)

    # 7. DC 比分矩阵 + 全量指标
    dc = dc_matrix_metrics(lam_h, lam_a)

    # 8. 持久化 Elo
    elo = compute_elo_persistent(
        home_name, away_name, home_tid, away_tid,
        home_m, away_m, h2h_m, standings, lg_key)

    # 9. 融合：DC 70% + Elo 30%（按 Elo 置信度加权）
    elo_w = min(elo.get("confidence", 0) * (1.0 - DC_WEIGHT), 1.0 - DC_WEIGHT)
    dc_w  = 1.0 - elo_w
    p_h_e = elo.get("p_home_elo", dc["p_home"])
    nd = dc["p_draw"]
    # Elo 只调主/客倾向，保留 DC 的平局
    nh = dc["p_home"] + dc["p_away"]
    if nh > 0:
        dc_home_share = dc["p_home"] / nh
        blended_share = dc_home_share * dc_w + p_h_e * elo_w
        bp_h = nh * blended_share
        bp_a = nh * (1 - blended_share)
    else:
        bp_h, bp_a = dc["p_home"], dc["p_away"]
    s = bp_h + nd + bp_a or 1.0
    base_h = round(bp_h/s, 4); base_d = round(nd/s, 4); base_a = round(bp_a/s, 4)

    # 10. 盘口信号
    mkt = odds_signal(bundle.get("odds_match_winner"))

    # 11. 置信度
    dp = gr_h["n"] + gr_a["n"] + (lp["n"] if lp["src"]=="standings" else 0)
    conf = _clamp(0.25 + 0.04*min(dp,12) + (0.15 if lp["src"]=="standings" else 0), 0.2, 0.95)

    return {
        "status": "ok",
        "engine": "strength_model_v2",
        "base_probs": {"home": base_h, "draw": base_d, "away": base_a},
        "base_probs_list": [base_h, base_d, base_a],
        "confidence": round(conf, 3),
        "lambda": {"home": round(lam_h,3), "away": round(lam_a,3)},
        "league_params": lp,
        "dixon_coles": dc,
        "elo": elo,
        "goal_rates": {"home": gr_h, "away": gr_a},
        "attack_defense": {"home": ad_h, "away": ad_a},
        "sos": {"home": sos_h, "away": sos_a},
        "lineup": {"home": lup_h, "away": lup_a},
        "market_signal": mkt,
        "notes": (
            "v2: Full-league DC normalization + persistent Elo + position-weighted lineup "
            "+ SOS-adjusted form + AH/OU/BTTS from score matrix. "
            "base_probs are model-derived (market-independent)."
        ),
    }

# ── __main__ ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src:
        with open(src, encoding="utf-8") as f: data = json.load(f)
    else:
        data = json.loads(sys.stdin.read())
    print(json.dumps(compute_strength(data), ensure_ascii=False, indent=2))
