# ⚽ Football Bot — 足球赛前量化分析机器人

基于 API-Football 原始数据的 Telegram 赛前分析机器人，集成 Dixon-Coles 泊松预测模型、Elo 评分、蒙特卡洛情景分析与 LLM 多市场综合判断。

---

## 核心功能

| 功能 | 说明 |
|------|------|
| 独立量化预测 | Dixon-Coles + Elo + 六项增强，独立于市场赔率 |
| 多市场分析 | 亚洲盘 / 大小球 / 双方进球 / 胜平负 / 最可能比分 |
| 蒙特卡洛 | 5情景×90%置信区间，mc_football.py 自动调用 |
| LLM综合判断 | 量化+基本面+赔率+阵容，每市场独立结论 |
| 历史回测 | 8583场/19联赛，Brier Score 0.618 |
| 每周自动复盘 | 拉取实际结果，更新准确率，每月重调参数 |

---

## Telegram 指令

```
/analyze 曼城 vs 利物浦         分析比赛
/analyze 2026-07-25 曼城 vs 利物浦   带日期分析
/a 曼城 vs 利物浦                简写
/help                           所有指令
/status                         服务运行状态
/calibration                    模型回测校准数据
/predictions                    最近预测存档
/leagues                        已校准联赛列表
/model                          查看/切换 LLM 模型
/bind 授权码                     绑定此聊天
```

---

## 快速部署

```bash
git clone https://github.com/unique555/football-cs-skill.git
cd football-cs-skill
python -m venv venv && source venv/bin/activate
pip install httpx python-dotenv requests
cp .env.example .env   # 填入 API keys
python app.py
```

`.env` 必填项：

```env
API_FOOTBALL_KEY=
TELEGRAM_BOT_TOKEN=
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=
LLM_MODEL=
ALLOWED_TELEGRAM_CHAT_IDS=
SKILL_PATH=/path/to/SKILL.md
MC_TOOL_PATH=/path/to/tools/mc_football.py
```

---

## 项目结构

```
├── app.py              主程序（Telegram + 数据采集 + 分析管道）
├── strength_model.py   独立预测模型（Dixon-Coles + Elo + 六项增强）
├── backtest.py         历史回测 + 参数自动调优
├── weekly_review.py    每周自动复盘
├── SKILL.md            LLM 分析框架（多市场判断规则，1100+ 行）
└── tools/
    └── mc_football.py  蒙特卡洛情景分析工具
```

---

## 预测模型（strength_model.py）

**六项增强：**

1. **Dixon-Coles 泊松** — 攻/防强度 → 期望进球 λ → 比分矩阵 → H/D/A 概率
2. **Elo 评分** — 跨赛季持久化，从积分榜+近期比赛校准
3. **近期状态指数衰减** — 近5场权重>50%
4. **阵容量化** — 按位置（GK/DEF/MID/FWD）加权缺阵影响
5. **情景因素** — 主客场分裂/赛程密度/战意
6. **赔率线变动** — 多家书商去水对比，盘口一致性

**输出：**
- 独立概率（主/平/客）+ 期望进球 λ
- 最可能比分 Top3
- 亚洲盘/大小球/BTTS 概率
- 市场偏差（模型 vs 去水概率，单位 pp）

---

## 回测结果

| 指标 | 数值 |
|------|------|
| 场次 | **8,583 场** |
| 联赛覆盖 | 英超/西甲/意甲/德甲/法甲/葡超/荷甲/土超/比利时/挪威/瑞典/芬兰/阿根廷/巴西/MLS/J联赛/K联赛/中超/沙特联 |
| Brier Score | **0.618**（随机基准 0.333） |
| 准确率 | **48.6%** |

```
0-30%:  pred=0.222 actual=0.237  diff=-1.5pp  ✅
30-45%: pred=0.363 actual=0.354  diff=+0.9pp  ✅
45-55%: pred=0.497 actual=0.482  diff=+1.5pp  ✅
55-65%: pred=0.595 actual=0.551  diff=+4.4pp  ⚠️
65%+:   pred=0.713 actual=0.682  diff=+3.1pp  ⚠️
```

---

## 自动化

```
每周一 03:03  → weekly_review.py    拉结果+更新存档+准确率报告
月初周一 04:17 → backtest --retune  重跑回测，若改善则更新参数
```

---

本项目仅供个人学习和研究使用，不构成投注建议。
