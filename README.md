# Football CS — Telegram 足球赛前分析机器人

基于 API-Football + Exa/Tavily 网络搜索 + Dixon-Coles 独立模型的 Telegram 机器人。

## 功能

- **独立预测模型**：Dixon-Coles 泊松 + 持久化 Elo + 近期状态指数衰减 + 阵容量化 + 情景因素 + 盘口变动
- **多市场推荐**：亚洲盘 / 大小球 / BTTS / 胜平负
- **三重数据来源**：API-Football（统计）+ Exa（网络核验）+ Tavily（辅助搜索）
- **时间感知**：距开赛 ≤60 分钟自动切换至「确认首发」搜索策略
- **回测调参**：30+ 联赛 8000+ 历史比赛，自动网格搜索最优参数
- **北单每日爬虫**：自动追加历史比赛数据，定期触发重新调参

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | Telegram 机器人主程序 |
| `strength_model.py` | 独立强度预测模型 |
| `skill/SKILL.md` | LLM 分析提示词 |
| `skill/tools/mc_football.py` | 蒙特卡洛情景分析工具 |
| `backtest.py` | 历史回测 + 参数调优脚本 |
| `beidan_crawl.py` | 北单联赛每日数据爬取 |
| `football-bot.service` | systemd 服务配置 |

## 快速部署

```bash
cp .env.example .env  # 填入各项 API Key
pip install -r requirements.txt
systemctl enable --now football-bot.service
```

## 环境变量

见 `.env.example`

## 每日数据更新（cron）

```
0 2 * * * cd /opt/football-bot && ./venv/bin/python beidan_crawl.py >> /var/log/beidan_crawl.log 2>&1
```

## 机器人指令

| 指令 | 说明 |
|------|------|
| `/analyze 曼城 vs 利物浦` | 分析比赛 |
| `/status` | 服务状态 |
| `/calibration` | 回测校准数据 |
| `/predictions` | 最近预测存档 |
| `/leagues` | 已校准联赛列表 |
| `/help` | 完整指令列表 |
