"""Real-time Web Dashboard for AI FX Trading Bot.

Lightweight Flask app displaying:
- Live candlestick charts (auto-updating)
- Open positions with entry/SL/TP levels
- Adaptive risk status
- Trade history
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.indicators import TechnicalAnalyzer
from src.utils.config_loader import TradingConfig, create_broker

app = Flask(__name__)
config: TradingConfig | None = None
broker: Any = None
tech_analyzer: TechnicalAnalyzer | None = None

# リアルタイム価格（ストリーミングスレッドが更新）
_live_prices: dict[str, dict[str, Any]] = {}
_stream_stop = threading.Event()

# チャートデータキャッシュ（30秒）
_cache: dict[str, Any] = {}
_cache_time: float = 0
_CACHE_TTL = 30  # seconds

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI FX Trading Bot - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
        }
        .header {
            background: #12121a;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2a2a3a;
        }
        .header h1 { font-size: 18px; color: #fff; }
        .header .status {
            display: flex;
            gap: 16px;
            font-size: 13px;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 4px;
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        .bot-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            padding: 3px 10px;
            border-radius: 4px;
            font-weight: 600;
        }
        .bot-status.running {
            background: rgba(34,197,94,0.15);
            color: #22c55e;
        }
        .bot-status.stopped {
            background: rgba(239,68,68,0.15);
            color: #ef4444;
            animation: none;
        }
        .bot-status.unknown {
            background: rgba(100,100,100,0.15);
            color: #888;
        }
        .bot-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            display: inline-block;
        }
        .running .bot-dot { background: #22c55e; animation: pulse 2s infinite; }
        .stopped .bot-dot { background: #ef4444; }
        .unknown .bot-dot { background: #888; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1px;
            background: #1a1a2a;
        }
        .chart-panel {
            background: #0f0f18;
            padding: 12px;
        }
        .chart-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .chart-title .pair { font-weight: 700; font-size: 16px; }
        .chart-title .price { font-family: monospace; font-size: 15px; }
        .price-up { color: #22c55e; }
        .price-down { color: #ef4444; }
        .chart-container { width: 100%; height: 350px; }
        .info-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1px;
            background: #1a1a2a;
            margin-top: 1px;
        }
        .info-card {
            background: #0f0f18;
            padding: 16px;
        }
        .info-card h3 {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .info-card .value {
            font-size: 20px;
            font-weight: 700;
            font-family: monospace;
        }
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .positions-table th {
            text-align: left;
            padding: 8px;
            color: #888;
            font-weight: 500;
            border-bottom: 1px solid #2a2a3a;
        }
        .positions-table td {
            padding: 8px;
            border-bottom: 1px solid #1a1a2a;
            font-family: monospace;
        }
        .buy-badge {
            background: rgba(34,197,94,0.15);
            color: #22c55e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .sell-badge {
            background: rgba(239,68,68,0.15);
            color: #ef4444;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .regime-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .regime-trending { background: rgba(34,197,94,0.15); color: #22c55e; }
        .regime-ranging { background: rgba(234,179,8,0.15); color: #eab308; }
        .regime-volatile { background: rgba(239,68,68,0.15); color: #ef4444; }
        .regime-normal { background: rgba(59,130,246,0.15); color: #3b82f6; }
        .footer {
            text-align: center;
            padding: 8px;
            font-size: 11px;
            color: #555;
        }
        .ai-log-section {
            padding: 12px 16px;
            background: #0a0a0f;
        }
        .ai-log-section h3 {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .ai-log-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 10px;
        }
        .ai-log-card {
            background: #0f0f18;
            border: 1px solid #1a1a2a;
            border-radius: 6px;
            padding: 14px;
        }
        .ai-log-card .card-pair {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 8px;
            border-bottom: 2px solid #2a2a3a;
        }
        .ai-log-card .card-time {
            font-size: 11px;
            color: #555;
        }
        .card-section {
            margin-top: 10px;
        }
        .card-section-title {
            font-size: 10px;
            font-weight: 600;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
            padding-bottom: 3px;
            border-bottom: 1px solid #1a1a2a;
        }
        .ai-log-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            padding: 4px 0;
            border-bottom: 1px solid #1a1a2a;
            color: #bbb;
        }
        .ai-log-row:last-child { border-bottom: none; }
        .ai-log-row .label { color: #888; min-width: 100px; font-size: 11px; }
        .ai-log-row .val { font-family: monospace; text-align: right; }
        .signal-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 4px;
        }
        .tag-buy { background: rgba(34,197,94,0.15); color: #22c55e; }
        .tag-sell { background: rgba(239,68,68,0.15); color: #ef4444; }
        .tag-none { background: rgba(100,100,100,0.15); color: #888; }
        .tag-consensus { background: rgba(168,85,247,0.15); color: #a855f7; }
        .tag-ai { background: rgba(59,130,246,0.15); color: #3b82f6; }
        .tag-mtf { background: rgba(234,179,8,0.15); color: #eab308; }
        .signal-detail {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .signal-metric {
            font-size: 11px;
            color: #999;
        }
        .signal-metric strong {
            color: #ccc;
        }
        .reasoning-text {
            font-size: 11px;
            color: #777;
            margin-top: 4px;
            margin-bottom: 6px;
            line-height: 1.5;
            word-break: break-word;
            padding-left: 8px;
            border-left: 2px solid #1a1a2a;
        }
        .vote-agree { color: #22c55e; font-weight: 600; }
        .vote-disagree { color: #ef4444; font-weight: 600; }
        .vote-neutral { color: #888; }
        .regime-section {
            background: #0f0f18;
            border: 1px solid #1a1a2a;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 10px;
            font-size: 12px;
            color: #bbb;
        }
        .regime-section .regime-title {
            font-size: 13px;
            font-weight: 600;
            color: #ddd;
            margin-bottom: 4px;
        }
        .no-data-msg {
            color: #555;
            font-size: 13px;
            padding: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI FX Trading Bot</h1>
        <div class="status">
            <span><span class="status-dot"></span> Live</span>
            <span id="update-time">--</span>
            <span id="regime-badge" class="regime-badge regime-normal">--</span>
            <span id="bot-status-badge" class="bot-status unknown">
                <span class="bot-dot"></span> Bot: 確認中...
            </span>
        </div>
    </div>

    <div class="info-bar" id="info-bar">
        <div class="info-card">
            <h3>Balance</h3>
            <div class="value" id="balance">--</div>
        </div>
        <div class="info-card">
            <h3>Unrealized P&L</h3>
            <div class="value" id="unrealized-pnl">--</div>
        </div>
        <div class="info-card">
            <h3>Daily P&L</h3>
            <div class="value" id="daily-pnl">--</div>
        </div>
        <div class="info-card">
            <h3>Drawdown</h3>
            <div class="value" id="drawdown">--</div>
        </div>
        <div class="info-card">
            <h3>Adaptive Risk</h3>
            <div class="value" id="risk-mult">--</div>
        </div>
        <div class="info-card">
            <h3>Win Rate</h3>
            <div class="value" id="win-rate">--</div>
        </div>
    </div>

    <div id="loading" style="text-align:center;padding:40px;color:#888;font-size:16px;">
        Loading chart data... (may take ~20s due to API rate limits)
    </div>
    <div class="charts-grid" id="charts-grid"></div>

    <div style="padding: 12px; background: #0f0f18;">
        <h3 style="font-size: 14px; margin-bottom: 8px; color: #888;">Open Positions</h3>
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Pair</th><th>Direction</th><th>Units</th><th>Entry</th>
                    <th>Current</th><th>SL</th><th>TP</th><th>P&L (pips)</th><th>P&L (¥)</th><th>Pattern</th>
                    <th>保有時刻</th><th>経過</th>
                </tr>
            </thead>
            <tbody id="positions-body"></tbody>
        </table>
    </div>

    <div class="ai-log-section">
        <h3>AI 市況分析ログ</h3>
        <div id="ai-log-container"><div class="no-data-msg">分析データを読み込み中...</div></div>
    </div>

    <div class="footer">AI FX Trading Bot - AI Sennin Methodology | Auto-refresh: 30s</div>

    <script>
        const charts = {};
        const candleSeries = {};
        const slLines = {};
        const tpLines = {};
        const entryLines = {};

        function createChart(containerId, pair) {
            const container = document.getElementById(containerId);
            const chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 350,
                layout: {
                    background: { color: '#0f0f18' },
                    textColor: '#888',
                },
                grid: {
                    vertLines: { color: '#1a1a2a' },
                    horzLines: { color: '#1a1a2a' },
                },
                crosshair: { mode: 0 },
                timeScale: {
                    timeVisible: true,
                    secondsVisible: false,
                },
            });

            const series = chart.addCandlestickSeries({
                upColor: '#22c55e',
                downColor: '#ef4444',
                borderUpColor: '#22c55e',
                borderDownColor: '#ef4444',
                wickUpColor: '#22c55e',
                wickDownColor: '#ef4444',
            });

            charts[pair] = chart;
            candleSeries[pair] = series;

            window.addEventListener('resize', () => {
                chart.applyOptions({ width: container.clientWidth });
            });
        }

        async function fetchData() {
            try {
                const resp = await fetch('/api/data');
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('loading').textContent = 'API Error: ' + data.error;
                    document.getElementById('loading').style.color = '#ef4444';
                    return;
                }

                // Update info cards
                document.getElementById('update-time').textContent = data.timestamp;
                document.getElementById('balance').textContent = data.balance;
                document.getElementById('unrealized-pnl').textContent = data.unrealized_pnl;
                document.getElementById('unrealized-pnl').style.color =
                    data.unrealized_pnl_raw >= 0 ? '#22c55e' : '#ef4444';
                document.getElementById('daily-pnl').textContent = data.daily_pnl;
                document.getElementById('daily-pnl').style.color =
                    data.daily_pnl_raw >= 0 ? '#22c55e' : '#ef4444';
                document.getElementById('drawdown').textContent = data.drawdown;
                document.getElementById('risk-mult').textContent = data.risk_multiplier;
                document.getElementById('win-rate').textContent = data.win_rate;

                const regimeBadge = document.getElementById('regime-badge');
                regimeBadge.textContent = data.regime;
                regimeBadge.className = 'regime-badge regime-' + data.regime;

                // Update charts
                const grid = document.getElementById('charts-grid');
                for (const pair of data.pairs) {
                    const containerId = 'chart-' + pair.instrument;

                    if (!charts[pair.instrument]) {
                        const panel = document.createElement('div');
                        panel.className = 'chart-panel';
                        panel.innerHTML = `
                            <div class="chart-title">
                                <span class="pair">${pair.name}</span>
                                <span class="price" id="price-${pair.instrument}">--</span>
                            </div>
                            <div class="chart-container" id="${containerId}"></div>
                        `;
                        grid.appendChild(panel);
                        createChart(containerId, pair.instrument);
                    }

                    if (pair.candles && pair.candles.length > 0) {
                        try {
                            candleSeries[pair.instrument].setData(pair.candles);
                        } catch (chartErr) {
                            console.error('Chart error for ' + pair.instrument + ':', chartErr, 'First candle:', pair.candles[0]);
                        }
                        const last = pair.candles[pair.candles.length - 1];
                        const priceEl = document.getElementById('price-' + pair.instrument);
                        priceEl.textContent = last.close.toFixed(pair.instrument.includes('JPY') ? 3 : 5);
                        priceEl.className = 'price ' + (last.close >= last.open ? 'price-up' : 'price-down');
                    }

                    // Draw position lines (or remove if no position)
                    const series = candleSeries[pair.instrument];
                    if (pair.position) {
                        const pos = pair.position;

                        // Remove old lines before redrawing
                        if (entryLines[pair.instrument]) {
                            series.removePriceLine(entryLines[pair.instrument]);
                            delete entryLines[pair.instrument];
                        }
                        if (slLines[pair.instrument]) {
                            series.removePriceLine(slLines[pair.instrument]);
                            delete slLines[pair.instrument];
                        }
                        if (tpLines[pair.instrument]) {
                            series.removePriceLine(tpLines[pair.instrument]);
                            delete tpLines[pair.instrument];
                        }

                        const dec = pair.instrument.includes('JPY') ? 3 : 5;
                        entryLines[pair.instrument] = series.createPriceLine({
                            price: pos.entry, color: '#3b82f6',
                            lineWidth: 2, lineStyle: 0,
                            axisLabelVisible: true,
                            title: pos.direction + ' ' + pos.size.toFixed(2) + ' lots @ ' + pos.entry.toFixed(dec) + ' [' + pos.pattern + ']',
                        });
                        slLines[pair.instrument] = series.createPriceLine({
                            price: pos.sl, color: '#ef4444',
                            lineWidth: 2, lineStyle: 2,
                            axisLabelVisible: true,
                            title: 'SL ' + pos.sl.toFixed(dec),
                        });
                        tpLines[pair.instrument] = series.createPriceLine({
                            price: pos.tp, color: '#22c55e',
                            lineWidth: 2, lineStyle: 2,
                            axisLabelVisible: true,
                            title: 'TP ' + pos.tp.toFixed(dec),
                        });
                    } else {
                        // ポジションなし → 残存ラインを削除
                        if (entryLines[pair.instrument]) {
                            series.removePriceLine(entryLines[pair.instrument]);
                            delete entryLines[pair.instrument];
                        }
                        if (slLines[pair.instrument]) {
                            series.removePriceLine(slLines[pair.instrument]);
                            delete slLines[pair.instrument];
                        }
                        if (tpLines[pair.instrument]) {
                            series.removePriceLine(tpLines[pair.instrument]);
                            delete tpLines[pair.instrument];
                        }
                    }
                }

                // Update positions table
                const tbody = document.getElementById('positions-body');
                if (data.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:#555;">No open positions</td></tr>';
                } else {
                    const now = new Date();
                    tbody.innerHTML = data.positions.map(p => {
                        // 経過時間を計算
                        let elapsedStr = '-';
                        if (p.opened_at) {
                            const opened = new Date(p.opened_at.replace(' ', 'T'));
                            const diffMs = now - opened;
                            const diffMins = Math.floor(diffMs / 60000);
                            const h = Math.floor(diffMins / 60);
                            const m = diffMins % 60;
                            elapsedStr = h > 0 ? `${h}h ${m}m` : `${m}m`;
                        }
                        // 保有時刻は月/日 時:分 表示
                        const openedLabel = p.opened_at ? p.opened_at.slice(5, 16) : '-';
                        return `
                        <tr data-instrument="${p.instrument}">
                            <td>${p.pair}</td>
                            <td><span class="${p.direction === 'BUY' ? 'buy-badge' : 'sell-badge'}">${p.direction}</span></td>
                            <td>${p.lots}</td>
                            <td>${p.entry}</td>
                            <td>${p.current}</td>
                            <td style="color:#ef4444">${p.sl}</td>
                            <td style="color:#22c55e">${p.tp}</td>
                            <td class="pips-cell" style="color:${p.pnl_pips >= 0 ? '#22c55e' : '#ef4444'};font-weight:600">${p.pnl_pips >= 0 ? '+' : ''}${p.pnl_pips.toFixed(1)} pips</td>
                            <td class="pnl-cell" style="color:${p.pnl_raw >= 0 ? '#22c55e' : '#ef4444'};font-weight:600">${p.pnl}</td>
                            <td>${p.pattern}</td>
                            <td style="color:#888;font-size:0.85em">${openedLabel}</td>
                            <td style="color:#facc15;font-size:0.85em">${elapsedStr}</td>
                        </tr>`;
                    }).join('');
                }

                document.getElementById('loading').style.display = 'none';

            } catch (e) {
                console.error('Fetch error:', e);
                document.getElementById('loading').textContent = 'Error: ' + e.message;
                document.getElementById('loading').style.color = '#ef4444';
                document.getElementById('update-time').textContent = 'Error';
                document.getElementById('update-time').style.color = '#ef4444';
            }
        }

        // --- AI Analysis Log ---
        const DIR_LABELS = { 'BUY': '買い', 'SELL': '売り', 'NONE': '様子見' };
        const REGIME_JP = {
            'trending': 'トレンド相場',
            'ranging': 'レンジ相場',
            'volatile': '高ボラティリティ',
            'normal': '通常'
        };
        const VOL_JP = { 'low': '低', 'medium': '中', 'high': '高', 'very_high': '超高' };
        const TREND_JP = { 'weak': '弱', 'moderate': '中', 'strong': '強' };

        function dirTag(d) {
            const cls = d === 'BUY' ? 'tag-buy' : d === 'SELL' ? 'tag-sell' : 'tag-none';
            return `<span class="signal-tag ${cls}">${DIR_LABELS[d] || d}</span>`;
        }

        function buildAiLogHTML(data) {
            if (!data || !data.pairs) return '<div class="no-data-msg">データなし</div>';

            let html = '';

            // Regime section
            const r = data.regime || {};
            if (r.regime) {
                const regJp = REGIME_JP[r.regime] || r.regime;
                const volJp = VOL_JP[r.volatility_level] || r.volatility_level || '-';
                const trendJp = TREND_JP[r.trend_strength] || r.trend_strength || '-';
                const mult = r.risk_multiplier != null ? `×${Number(r.risk_multiplier).toFixed(2)}` : '-';
                html += `<div class="regime-section">
                    <div class="regime-title">相場レジーム: ${regJp}</div>
                    <div>ボラティリティ: ${volJp} ／ トレンド強度: ${trendJp} ／ リスク倍率: ${mult}</div>
                    ${r.reasoning ? `<div class="reasoning-text">${r.reasoning.substring(0, 200)}</div>` : ''}
                </div>`;
            }

            html += `<div class="ai-log-grid">`;
            const pairs = Object.values(data.pairs);
            if (pairs.length === 0) {
                html += '<div class="no-data-msg">ペアの分析結果なし</div>';
            }
            for (const pair of pairs) {
                const pairName = pair.pair_name || '-';
                const analyzedAt = pair.analyzed_at || '-';
                const gv = pair.groq_vote || {};
                const signals = pair.signals || [];

                // --- Build signals section ---
                let signalsHtml = '';
                if (signals.length === 0) {
                    signalsHtml = '<div class="ai-log-row"><span class="label">検出なし</span></div>';
                } else {
                    for (const s of signals) {
                        const tags = [];
                        if (s.consensus_confirmed) tags.push('<span class="signal-tag tag-consensus">コンセンサス確定</span>');
                        if (s.ai_confirmed) tags.push('<span class="signal-tag tag-ai">AI確認済</span>');
                        if (s.mtf_confirmed) tags.push('<span class="signal-tag tag-mtf">MTF一致</span>');
                        signalsHtml += `
                        <div class="ai-log-row">
                            <span class="label">${s.pattern || '-'}</span>
                            <span class="val">
                                <span class="signal-detail">
                                    ${dirTag(s.direction)}
                                    <span class="signal-metric">信頼度 <strong>${s.confidence}%</strong></span>
                                    <span class="signal-metric">R:R <strong>${s.rr_ratio}</strong></span>
                                    ${tags.join('')}
                                </span>
                            </span>
                        </div>`;
                        if (s.reasoning) {
                            signalsHtml += `<div class="reasoning-text">${s.reasoning.substring(0, 150)}</div>`;
                        }
                    }
                }

                // --- Build AI votes section ---
                const groqDir = gv.direction || 'NONE';
                const groqConf = gv.confidence != null ? `${Number(gv.confidence).toFixed(0)}%` : '-';
                const groqReason = gv.reasoning ? gv.reasoning.substring(0, 150) : '';

                const gemv = pair.gemini_vote || {};
                const gemDir = gemv.direction || 'N/A';
                const gemConf = gemv.confidence != null && gemv.confidence > 0 ? `${Number(gemv.confidence).toFixed(0)}%` : '-';
                const gemReason = gemv.reasoning ? gemv.reasoning.substring(0, 150) : '';

                // Check if both AIs agree
                const bothExist = groqDir !== 'NONE' && gemDir !== 'N/A';
                const aiAgree = bothExist && groqDir === gemDir;
                const consensusNote = bothExist
                    ? (aiAgree
                        ? `<span class="vote-agree">AI一致: ${groqDir === 'BUY' ? '買い' : '売り'}</span>`
                        : `<span class="vote-disagree">AI不一致</span>`)
                    : '';

                // --- Overall verdict ---
                const ov = pair.overall || {};
                const ovDir = ov.direction || 'NONE';
                const ovConf = ov.confidence || 0;
                const ovAgree = ov.agree_count || 0;
                const ovTotal = ov.total_votes || 3;
                const hasVerdict = ovDir !== 'NONE';

                let verdictHtml = '';
                if (hasVerdict) {
                    const verdictColor = ovDir === 'BUY' ? '#22c55e' : '#ef4444';
                    const verdictLabel = ovDir === 'BUY' ? '買い' : '売り';
                    verdictHtml = `
                    <div style="background:${ovDir === 'BUY' ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)'};border:1px solid ${verdictColor}33;border-radius:6px;padding:8px 12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px">総合判断</span>
                            <div style="font-size:20px;font-weight:700;color:${verdictColor}">${verdictLabel}</div>
                        </div>
                        <div style="text-align:right">
                            <div style="font-size:18px;font-weight:700;color:${verdictColor}">${ovConf}%</div>
                            <div style="font-size:10px;color:#888">${ovAgree}/${ovTotal} AI一致</div>
                        </div>
                    </div>`;
                } else {
                    verdictHtml = `
                    <div style="background:rgba(100,100,100,0.08);border:1px solid #33333366;border-radius:6px;padding:8px 12px;margin-bottom:10px;text-align:center">
                        <span style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px">総合判断</span>
                        <div style="font-size:16px;font-weight:600;color:#666">見送り（AI不一致）</div>
                    </div>`;
                }

                // --- Build vote detail rows ---
                const votes = pair.votes || [];
                const gs = pair.gemini_status || {};
                let votesHtml = '';
                const SOURCE_LABELS = {'Algorithm': 'パターン検出', 'Gemini': 'Gemini（チャート）', 'Groq': 'Groq（ニュース）'};
                for (const v of votes) {
                    const srcLabel = SOURCE_LABELS[v.source] || v.source;
                    const vDir = v.direction || 'NONE';
                    const vConf = v.confidence || 0;
                    const isNone = vDir === 'NONE' || vDir === 'N/A' || vDir === 'NEUTRAL';

                    // Gemini用の特別表示: スキップ理由を表示
                    let valHtml = '';
                    if (v.source === 'Gemini' && isNone && gs.skip_reason) {
                        valHtml = `<span style="color:#888;font-size:11px">${gs.skip_reason}</span>`;
                    } else if (v.source === 'Gemini' && !isNone && gs.cached) {
                        valHtml = dirTag(vDir) + ' ' + vConf + '% <span style="color:#666;font-size:10px">(cache)</span>';
                    } else if (isNone) {
                        valHtml = '<span style="color:#555">---</span>';
                    } else {
                        valHtml = dirTag(vDir) + ' ' + vConf + '%';
                    }

                    votesHtml += `
                    <div class="ai-log-row">
                        <span class="label">${srcLabel}</span>
                        <span class="val">${valHtml}</span>
                    </div>`;
                }
                // Gemini budget display
                if (gs.budget_used != null) {
                    votesHtml += `
                    <div class="ai-log-row" style="margin-top:2px">
                        <span class="label" style="font-size:10px;color:#555">Gemini残量</span>
                        <span class="val" style="font-size:10px;color:#555">${gs.budget_used}/${gs.budget_limit} 使用済</span>
                    </div>`;
                }

                html += `<div class="ai-log-card">
                    <div class="card-pair">
                        <span>${pairName}</span>
                        <span class="card-time">${analyzedAt}</span>
                    </div>

                    ${verdictHtml}

                    <div class="card-section">
                        <div class="card-section-title">3票投票の内訳</div>
                        ${votesHtml}
                        ${groqReason ? `<div class="reasoning-text">${groqReason}</div>` : ''}
                        ${gemReason ? `<div class="reasoning-text">${gemReason}</div>` : ''}
                    </div>

                    <div class="card-section">
                        <div class="card-section-title">パターン検出シグナル</div>
                        ${signalsHtml}
                    </div>
                </div>`;
            }
            html += '</div>';

            if (data.updated_at) {
                html += `<div style="font-size:11px;color:#444;margin-top:8px;text-align:right">最終更新: ${data.updated_at}</div>`;
            }
            return html;
        }

        function updateBotStatus(data) {
            const badge = document.getElementById('bot-status-badge');
            if (!data || !data.updated_at) {
                badge.className = 'bot-status unknown';
                badge.innerHTML = '<span class="bot-dot"></span> Bot: データなし';
                return;
            }
            const updated = new Date(data.updated_at.replace(' ', 'T'));
            const diffMin = (Date.now() - updated.getTime()) / 60000;
            // 最終更新から12分以内なら稼働中（5分サイクル + 余裕7分）
            if (diffMin <= 12) {
                badge.className = 'bot-status running';
                badge.innerHTML = `<span class="bot-dot"></span> Bot: 稼働中 (${data.updated_at.slice(11, 16)} 更新)`;
            } else {
                badge.className = 'bot-status stopped';
                const ago = Math.floor(diffMin);
                badge.innerHTML = `<span class="bot-dot"></span> Bot: 停止中 (${ago}分前が最終更新)`;
            }
        }

        async function fetchAiLog() {
            try {
                const resp = await fetch('/api/ai_log');
                const data = await resp.json();
                updateBotStatus(data);
                document.getElementById('ai-log-container').innerHTML = buildAiLogHTML(data);
            } catch (e) {
                document.getElementById('ai-log-container').innerHTML =
                    '<div class="no-data-msg">AI分析ログの取得に失敗しました</div>';
            }
        }

        // --- リアルタイム価格更新（2秒） ---
        async function fetchLivePrices() {
            try {
                const resp = await fetch('/api/prices');
                const prices = await resp.json();
                for (const [instrument, p] of Object.entries(prices)) {
                    const priceEl = document.getElementById('price-' + instrument);
                    if (!priceEl) continue;
                    const isJpy = instrument.includes('JPY');
                    const dec = isJpy ? 3 : 5;
                    const mid = p.mid || p.ask || 0;
                    const prev = parseFloat(priceEl.dataset.prev || mid);
                    priceEl.textContent = mid.toFixed(dec);
                    priceEl.className = 'price ' + (mid >= prev ? 'price-up' : 'price-down');
                    priceEl.dataset.prev = mid;

                }
            } catch (e) { /* ストリーム未接続時は無視 */ }
        }

        // Show loading state
        document.getElementById('update-time').textContent = 'Loading...';
        fetchData();
        fetchAiLog();
        setInterval(fetchData, 30000);
        setInterval(fetchAiLog, 60000);
        setInterval(fetchLivePrices, 2000);
        fetchLivePrices();
    </script>
</body>
</html>
"""


def init_app() -> Flask:
    """Initialize the dashboard app with trading config."""
    global config, broker, tech_analyzer

    config = TradingConfig()
    broker = create_broker(config.env)
    broker.connect()
    tech_analyzer = TechnicalAnalyzer(config.indicators)

    # OANDAならストリーミングスレッドを起動
    if hasattr(broker, "stream_prices"):
        _start_price_stream()

    return app


def _start_price_stream() -> None:
    """OANDAストリーミングAPIのバックグラウンドスレッドを起動する。"""
    def _on_price(instrument: str, bid: float, ask: float, ts: str) -> None:
        mid = round((bid + ask) / 2, 6)
        _live_prices[instrument] = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "time": ts,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }

    def _run() -> None:
        instruments = config.active_pairs if config else []
        if not instruments:
            return
        logger.info(f"Price stream starting: {instruments}")
        broker.stream_prices(instruments, _on_price, stop_event=_stream_stop)

    t = threading.Thread(target=_run, daemon=True, name="price-stream")
    t.start()
    logger.info("Price stream thread started")


@app.route("/")
def index() -> str:
    """Serve the dashboard HTML."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/data")
def api_data() -> Any:
    """API endpoint returning all dashboard data."""
    global config, broker, tech_analyzer, _cache, _cache_time

    if not broker or not config:
        return jsonify({"error": "Not initialized"}), 500

    # Return cached data if fresh (avoids rate limits)
    now = time.time()
    if _cache and (now - _cache_time) < _CACHE_TTL:
        _cache["timestamp"] = datetime.now().strftime("%H:%M:%S")
        _cache["cached"] = True
        return jsonify(_cache)

    try:
        logger.info("Dashboard: fetching fresh data...")
        broker.ensure_session()
        account = broker.get_account_info()
        balance = account.get("balance", 0)

        # Read trade log for open positions
        positions = _get_positions()

        # Build pair data with candles (with delay between requests for rate limiting)
        pairs_data = []
        for i, instrument in enumerate(config.active_pairs):
            pair_config = config.pair_configs.get(instrument, {})
            pair_name = pair_config.get("name", instrument)

            # Rate limit: wait between requests
            if i > 0:
                time.sleep(8)  # Twelve Data free: 8 req/min

            # Fetch candles
            logger.info(f"Dashboard: fetching {instrument}...")
            df = broker.get_historical_prices(instrument, "HOUR", num_points=100)
            candles = []
            if not df.empty:
                seen_times: set[int] = set()
                for idx, row in df.iterrows():
                    ts = int(idx.timestamp())
                    # Avoid duplicate timestamps (Lightweight Charts requires unique ascending times)
                    if ts in seen_times:
                        continue
                    seen_times.add(ts)
                    candles.append({
                        "time": ts,
                        "open": round(float(row["open"]), 5),
                        "high": round(float(row["high"]), 5),
                        "low": round(float(row["low"]), 5),
                        "close": round(float(row["close"]), 5),
                    })
                # Ensure ascending order
                candles.sort(key=lambda c: c["time"])
                logger.info(f"Dashboard: {instrument} -> {len(candles)} candles, "
                            f"range: {candles[0]['time']} ~ {candles[-1]['time']}")
            else:
                logger.warning(f"Dashboard: {instrument} -> no data")

            # Find position for this pair
            pos_data = None
            for p in positions:
                if p.get("instrument") == instrument:
                    pos_data = {
                        "entry": p.get("entry", 0),
                        "sl": p.get("sl", 0),
                        "tp": p.get("tp", 0),
                        "direction": p.get("direction", "BUY"),
                        "size": p.get("size", 0),
                        "pattern": p.get("pattern", ""),
                    }

            pairs_data.append({
                "instrument": instrument,
                "name": pair_name,
                "candles": candles,
                "position": pos_data,
            })

        # OANDA APIから取得した口座全体の未実現損益を使用
        total_unrealized = float(account.get("unrealized_pl", account.get("profit_loss", 0)))

        # Build positions list for table
        positions_table = []
        for p in positions:
            instrument = p.get("instrument", "")
            pair_config = config.pair_configs.get(instrument, {})
            current_price = 0
            for pd_item in pairs_data:
                if pd_item["instrument"] == instrument and pd_item["candles"]:
                    current_price = pd_item["candles"][-1]["close"]

            entry = p.get("entry", 0)
            direction = p.get("direction", "BUY")
            size = p.get("size", 0)
            decimals = broker._price_precision(instrument) if hasattr(broker, '_price_precision') else (3 if "JPY" in instrument else 5)
            pip_size_val = broker.get_pip_value(instrument) if hasattr(broker, 'get_pip_value') else (0.01 if "JPY" in instrument else 0.0001)

            # OANDAのunrealizedPLをそのまま使用（APIが権威データ）
            pnl_amount = float(p.get("unrealized_pl", 0))

            pnl_diff_for_pips = (current_price - entry) if direction == "BUY" else (entry - current_price)
            pnl_pips = pnl_diff_for_pips / pip_size_val if pip_size_val else 0

            positions_table.append({
                "pair": pair_config.get("name", instrument),
                "instrument": instrument,
                "direction": direction,
                "lots": f"{size:,.0f}",
                "entry": f"{entry:.{decimals}f}",
                "entry_raw": entry,
                "size_raw": size,
                "current": f"{current_price:.{decimals}f}",
                "sl": f"{p.get('sl', 0):.{decimals}f}",
                "tp": f"{p.get('tp', 0):.{decimals}f}",
                "pnl_pips": round(pnl_pips, 1),
                "pnl": f"¥{pnl_amount:+,.0f}",
                "pnl_raw": pnl_amount,
                "pattern": p.get("pattern", ""),
                "opened_at": p.get("opened_at", ""),
            })

        # Drawdown計算（NAVベース）
        nav = float(account.get("nav", balance))
        dd_pct = ((balance - nav) / balance * 100) if balance > 0 and nav < balance else 0.0

        result = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "balance": f"¥{balance:,.0f}",
            "unrealized_pnl": f"¥{total_unrealized:+,.0f}",
            "unrealized_pnl_raw": total_unrealized,
            "daily_pnl": f"¥{float(account.get('realized_pl', 0)):+,.0f}",
            "daily_pnl_raw": float(account.get("realized_pl", 0)),
            "drawdown": f"{dd_pct:.1f}%",
            "risk_multiplier": "×1.00",
            "win_rate": "N/A",
            "regime": "normal",
            "pairs": pairs_data,
            "positions": positions_table,
            "cached": False,
        }

        # Cache the result
        _cache = result
        _cache_time = time.time()

        logger.info("Dashboard: data ready")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Dashboard API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/divergence")
def api_divergence() -> Any:
    """Divergence check endpoint: forward vs backtest comparison."""
    try:
        from src.divergence_checker import run_check
        result = run_check()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Divergence check error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug")
def api_debug() -> Any:
    """Debug endpoint to check raw data."""
    if not broker or not config:
        return jsonify({"error": "Not initialized"})

    instrument = "USD_JPY"
    df = broker.get_historical_prices(instrument, "HOUR", num_points=5)
    raw = []
    if not df.empty:
        for idx, row in df.iterrows():
            raw.append({
                "datetime": str(idx),
                "timestamp": int(idx.timestamp()),
                "open": float(row["open"]),
                "close": float(row["close"]),
            })
    return jsonify({"instrument": instrument, "count": len(raw), "data": raw})


@app.route("/api/prices")
def api_prices() -> Any:
    """ストリーミングで受信したリアルタイム価格を返す。"""
    return jsonify(_live_prices)


@app.route("/api/ai_log")
def api_ai_log() -> Any:
    """Return AI analysis log written by TradingBot each cycle."""
    log_path = Path(__file__).parent.parent / "data" / "ai_analysis_log.json"
    if not log_path.exists():
        return jsonify({"pairs": {}, "regime": {}, "updated_at": None})
    try:
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"ai_log read error: {e}")
        return jsonify({"error": str(e)}), 500


def _get_positions() -> list[dict]:
    """Get open positions — OANDA is truth source, trade log is fallback.

    Strategy:
    1. Try OANDA API first (authoritative)
    2. Fall back to trade log parsing if OANDA unavailable
    3. Overlay SL/TP from SQLite for both sources
    """
    positions: dict[str, dict] = {}

    # 1. Try OANDA API first (truth source)
    oanda_ok = False
    try:
        if broker:
            broker_df = broker.get_open_positions()
            if hasattr(broker_df, 'empty') and not broker_df.empty:
                oanda_ok = True
                for _, row in broker_df.iterrows():
                    deal_id = row.get("dealId", "")
                    if not deal_id:
                        continue
                    instrument = row.get("epic", "")
                    positions[deal_id] = {
                        "deal_id": deal_id,
                        "instrument": instrument,
                        "direction": row.get("direction", "BUY"),
                        "entry": float(row.get("level", 0)),
                        "sl": float(row.get("stopLevel", 0) or 0),
                        "tp": float(row.get("limitLevel", 0) or 0),
                        "size": float(row.get("size", 0)),
                        "unrealized_pl": float(row.get("unrealizedPL", 0)),
                        "pattern": "",
                        "opened_at": str(row.get("openedAt", ""))[:16],
                    }
                logger.info(f"Positions from OANDA: {list(positions.keys())}")
    except Exception as e:
        logger.warning(f"OANDA position fetch failed, falling back to log: {e}")

    # 2. Fall back to trade log if OANDA returned nothing
    if not oanda_ok:
        trade_log = Path("logs/trades.log")
        if trade_log.exists():
            try:
                with open(trade_log) as f:
                    for line in f:
                        if " | " not in line:
                            continue
                        _, _, trade_part = line.partition(" | ")
                        trade_part = trade_part.strip()

                        if trade_part.startswith("OPEN|"):
                            parts = trade_part.split("|")
                            if len(parts) >= 7:
                                pair_name = parts[1]
                                direction = parts[2]
                                fields = {}
                                for p in parts[3:]:
                                    if "=" in p:
                                        k, v = p.split("=", 1)
                                        fields[k] = v

                                ts_raw = line.split(" | ")[0].strip()
                                opened_at = ts_raw[:16]
                                instrument = pair_name.replace("/", "_")
                                deal_id = fields.get("deal", instrument)

                                positions[deal_id] = {
                                    "deal_id": deal_id,
                                    "instrument": instrument,
                                    "direction": direction,
                                    "entry": float(fields.get("entry", 0)),
                                    "sl": float(fields.get("sl", 0)),
                                    "tp": float(fields.get("tp", 0)),
                                    "size": float(fields.get("size", 0)),
                                    "pattern": fields.get("pattern", ""),
                                    "opened_at": opened_at,
                                }

                        elif trade_part.startswith("CLOSE|"):
                            parts = trade_part.split("|")
                            fields = {}
                            for p in parts[2:]:
                                if "=" in p:
                                    k, v = p.split("=", 1)
                                    fields[k] = v
                            deal_id = fields.get("deal")
                            if deal_id:
                                positions.pop(deal_id, None)
                            elif len(parts) >= 2:
                                inst = parts[1].replace("/", "_")
                                for did in [d for d, p in positions.items() if p["instrument"] == inst]:
                                    positions.pop(did, None)

            except Exception as e:
                logger.error(f"Error reading trade log: {e}")
            logger.info(f"Positions from log: {list(positions.keys())}")

    # 3. Enrich with pattern info from trade log (for OANDA-sourced positions)
    if oanda_ok:
        trade_log = Path("logs/trades.log")
        if trade_log.exists():
            try:
                # Build deal_id -> pattern map from log
                log_patterns: dict[str, str] = {}
                with open(trade_log) as f:
                    for line in f:
                        if " | " not in line or "OPEN|" not in line:
                            continue
                        _, _, trade_part = line.partition(" | ")
                        parts = trade_part.strip().split("|")
                        fields = {}
                        for p in parts[3:]:
                            if "=" in p:
                                k, v = p.split("=", 1)
                                fields[k] = v
                        if "deal" in fields and "pattern" in fields:
                            log_patterns[fields["deal"]] = fields["pattern"]
                for deal_id, pos in positions.items():
                    if deal_id in log_patterns:
                        pos["pattern"] = log_patterns[deal_id]
            except Exception:
                pass

    # 4. Overlay SL/TP from SQLite (deal_idで正確に照合)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models.position_store import PositionStore
        store = PositionStore()
        db_positions = {p["deal_id"]: p for p in store.get_open_positions()}
        for deal_id, pos in positions.items():
            if deal_id in db_positions:
                db_pos = db_positions[deal_id]
                if db_pos.get("stop_loss"):
                    pos["sl"] = db_pos["stop_loss"]
                if db_pos.get("take_profit"):
                    pos["tp"] = db_pos["take_profit"]
    except Exception as e:
        logger.warning(f"Could not overlay SL/TP from SQLite: {e}")

    return list(positions.values())


def main() -> None:
    """Run the dashboard server."""
    import argparse

    def _can_bind(host: str, port: int) -> bool:
        """Check whether host:port can be bound by this process."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return True
            except OSError:
                return False

    def _select_port(host: str, preferred_port: int, max_offset: int = 20) -> int:
        """Return the preferred port or the next available one."""
        if _can_bind(host, preferred_port):
            return preferred_port

        for offset in range(1, max_offset + 1):
            candidate = preferred_port + offset
            if _can_bind(host, candidate):
                logger.warning(
                    "Port {} is already in use. Falling back to port {}.",
                    preferred_port,
                    candidate,
                )
                return candidate

        raise RuntimeError(
            f"No available port in range {preferred_port}-{preferred_port + max_offset}"
        )

    parser = argparse.ArgumentParser(description="AI FX Bot Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    try:
        selected_port = _select_port(args.host, args.port)
    except RuntimeError as exc:
        logger.error(
            "Failed to start dashboard: {}. Use --port to specify an available port.",
            exc,
        )
        raise SystemExit(1) from exc

    init_app()
    print(f"\n  Dashboard: http://localhost:{selected_port}\n")
    app.run(host=args.host, port=selected_port, debug=False)


if __name__ == "__main__":
    main()
