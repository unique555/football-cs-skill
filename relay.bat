@echo off
:: 竞彩数据中继 — 本地抓取并上传到服务器
:: 配置为每天 08:50 和 16:50 运行（比服务器 09:00/17:00 推送提前10分钟）
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python jingcai_relay.py >> relay.log 2>&1
