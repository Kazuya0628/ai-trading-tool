# AI FX Trading Tool 設計書（統合改訂版）

- **文書番号**: DESIGN-001
- **バージョン**: 1.3
- **作成日**: 2026-03-27
- **最終更新**: 2026-03-30
- **関連文書**: SPEC-001 v1.1（仕様書）

---

## 0. 本改訂版の位置づけ

本設計書は、以下の方針で構成されている。

- 実装に落とし込みやすいファイル構成（既存プロジェクトとの親和性を重視）
- `main.py` / `TradingBot` を起点とした分かりやすい制御構造
- ダッシュボードを見据えた `cycle_log` 設計
- API障害時の実務的な挙動整理
- 起動時の状態復元フロー

同時に、以下の設計原則を反映する。

- Algorithm主導・AI補助の責務分離
- Gemini予算管理の独立サービス化
- 3者投票 / 2者フォールバックの明文化
- OANDAを主系の真実源とする整合設計
- 通貨エクスポージャー制御・安全停止・Phase運用の反映
- バックテストとライブ条件の整合性確保

### 0.1 v1.3 変更概要

v1.3 では、v1.2 のアーキテクチャを維持したまま、実装前に必要な整合修正を追加した。

主な変更点は以下の通り。

1. ディレクトリ責務の曖昧さを補うため、既存ファイル配置の責務を明文化
2. Consensus閾値を固定値ではなく設定値 / 状態値として扱うことを明記
3. RiskManager に JPY口座向け pip value 換算責務を追加
4. ExecutionService にシグナル失効時間・ペア別スリッページ閾値を追加
5. PortfolioService に daily / weekly NAV anchor 概念を追加
6. GeminiBudgetService の優先順位ロジックを、初期実装と拡張方針に分けて定義
7. PositionStore の Phase 1 実装範囲と将来拡張範囲を明確化

---

## 1. システムアーキテクチャ

### 1.1 全体構成図

```text
                           ┌──────────────────────┐
                           │       main.py        │
                           │   CLI エントリー      │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │      TradingBot      │
                           │   実行オーケストレータ │
                           └───┬──────┬──────┬────┘
                               │      │      │
                 ┌─────────────┘      │      └──────────────────────┐
                 │                    │                             │
        ┌────────▼────────┐  ┌────────▼────────┐          ┌────────▼────────┐
        │ Scheduler        │  │ Portfolio/State │          │ Notification    │
        │ Layer0/1/2制御   │  │ NAV/DD/Exposure │          │ LINE/Email/etc  │
        └────────┬────────┘  └────────┬────────┘          └─────────────────┘
                 │                    │
        ┌────────▼────────┐  ┌────────▼────────┐
        │ MarketData       │  │ PositionStore   │
        │ OANDA/代替データ │  │ SQLite永続化     │
        └────────┬────────┘  └─────────────────┘
                 │
        ┌────────▼────────┐
        │ TechnicalAnalyzer│
        │ 指標計算         │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ StrategyEngine   │
        │ 分析統合         │
        └──┬──────────┬───┘
           │          │
   ┌───────▼──────┐  ┌▼────────────────┐
   │ PatternDetector│ │ AIOrchestrator  │
   │ アルゴ検出     │ │ Gemini/Groq統合 │
   └───────┬──────┘  └──────┬─────────┘
           │                │
           └──────┬─────────┘
                  ▼
          ┌───────────────┐
          │ ConsensusEngine│
          │ 投票統合       │
          └──────┬────────┘
                 ▼
          ┌───────────────┐
          │ RiskManager    │
          │ リスク検証     │
          └──────┬────────┘
                 ▼
          ┌───────────────┐
          │ ExecutionSvc   │
          │ 発注/クローズ   │
          └──────┬────────┘
                 ▼
          ┌───────────────┐
          │ OANDA Broker   │
          │ Adapter        │
          └───────────────┘
```

### 1.2 基本設計方針

1. **TradingBotは司令塔であり、業務ロジックを持ちすぎない**
2. **分析・投票・リスク・執行をサービス分離する**
3. **OANDAを執行と主系価格の真実源とする**
4. **新規エントリーはLayer 1（4H確定時）のみ**
5. **安全側に倒す（fail-safe）**
6. **フォワードテスト時の可観測性を最優先する**

---

## 2. ディレクトリ構成

既存構成との親和性を保ちつつ、責務分離を強化した構成とする。

```text
AiTradeTool/
├── main.py
├── config/
│   └── trading_config.yaml
├── src/
│   ├── trading_bot.py              # TradingBot（実行オーケストレータ）
│   ├── dashboard.py                # 既存Flaskダッシュボード
│   ├── app/
│   │   ├── runtime_state.py        # 実行時状態（RuntimeState）
│   │   └── scheduler.py            # Layer0/1/2 スケジューラ
│   ├── brokers/
│   │   ├── base_broker.py          # BrokerClient interface (ABC)
│   │   ├── oanda_client.py         # OANDA v20 client
│   │   └── twelvedata_client.py    # 補助データソース
│   ├── data/
│   │   ├── market_data.py          # マーケットデータ取得
│   │   ├── indicators.py           # TechnicalAnalyzer
│   │   ├── csv_loader.py           # CSV/yfinance loader
│   │   ├── sentiment_fetcher.py    # 補助センチメント
│   │   ├── fetchers/               # データ取得モジュール
│   │   └── processors/             # データ加工モジュール
│   ├── domain/
│   │   ├── enums.py                # RunMode, Phase, ConsensusMode 等
│   │   └── models.py               # ConsensusVote, SignalCandidate, OrderIntent 等
│   ├── models/
│   │   ├── risk_manager.py         # RiskManager, AdaptiveRiskController
│   │   ├── position_store.py       # PositionStore (SQLite)
│   │   └── backtest_engine.py      # バックテストエンジン
│   ├── strategies/
│   │   ├── pattern_detector.py     # 6パターン検出
│   │   ├── ai_analyzer.py          # Gemini adapter wrapper
│   │   ├── groq_reviewer.py        # Groq adapter wrapper
│   │   ├── consensus_engine.py     # 3者投票 / 2者フォールバック
│   │   ├── strategy_engine.py      # マルチ戦略統合
│   │   └── ma_cross.py             # MAクロス戦略
│   ├── services/
│   │   ├── gemini_budget_service.py  # Gemini日次/サイクル予算管理
│   │   ├── execution_service.py      # 発注・クローズ処理
│   │   ├── portfolio_service.py      # NAV/DD/エクスポージャー
│   │   └── reconciliation_service.py # OANDA⇔DB整合
│   ├── repositories/
│   │   └── __init__.py             # PositionStore re-export
│   ├── web/
│   │   ├── dashboard.py            # Flask App ファクトリ
│   │   └── routes.py               # API Blueprint
│   └── utils/
│       ├── config_loader.py        # 設定読み込み・ブローカーファクトリ
│       ├── logger.py               # ログ設定
│       └── notifier.py             # Email/Discord/LINE通知
├── deploy/
│   ├── install.sh                  # Linux systemd インストール
│   └── install_mac.sh              # macOS LaunchAgent インストール
├── tests/                          # テストスイート
├── data/
│   ├── positions.db
│   ├── historical/
│   └── ai_analysis_log.json
├── logs/
├── docs/
├── scripts/
│   └── bulk_backtest.py
├── requirements.txt
├── .env
└── .env.example
```

### 2.1 責務配置ルール

既存構成との互換性を維持するため、ファイル配置と責務の対応を以下のように明示する。

| パス | 主責務 | 補足 |
|------|--------|------|
| `src/trading_bot.py` | 実行オーケストレーション | 業務判断ロジックは持たない |
| `src/dashboard.py` | 既存ダッシュボード起動互換 / 互換エントリ | 旧構成との互換維持 |
| `src/web/dashboard.py` | Flask App ファクトリ | Web構築責務の本体 |
| `src/web/routes.py` | API Blueprint | ルーティング責務 |
| `src/models/risk_manager.py` | リスク制御実装 | 配置上は `models` だが責務上は service 相当 |
| `src/models/position_store.py` | SQLite永続化 | Phase 1 では repository 実装を兼ねる |
| `src/services/execution_service.py` | 発注・クローズ実行 | OANDAとの注文境界 |
| `src/services/portfolio_service.py` | NAV / DD / anchor / exposure 計算 | 口座状態の真実性維持 |
| `src/services/reconciliation_service.py` | OANDA ⇔ DB 整合 | 起動時・定期整合 |

#### 実装ルール

- 新規ビジネスロジックは `TradingBot` に追加しない
- SQL直書きは `PositionStore` に閉じ込める
- ダッシュボードのルーティング追加は `src/web/routes.py` に集約する
- 既存パスは維持するが、責務は本節を正とする

---

## 3. コンポーネント設計

## 3.1 TradingBot（実行オーケストレータ）

**ファイル**: `src/trading_bot.py`

### 責務

- Schedulerからの Layer 0 / 1 / 2 実行要求を受ける
- 必要なサービスを順に呼び出す
- 実行時状態を保持する（RuntimeState経由）
- ダッシュボード用 `cycle_log` を管理する
- モード遷移（PAPER / ANALYSIS_ONLY / SAFE_STOP）を行う

### 設計方針

TradingBot は**薄いオーケストレータ**とし、以下を直接持たない。

- パターン判定ロジック → PatternDetector
- コンセンサス判定ロジック → ConsensusEngine
- リスク計算ロジック → RiskManager
- Gemini予算管理 → GeminiBudgetService
- DB SQL実装 → PositionStore
- NAV/DD計算 → PortfolioService
- 発注処理 → ExecutionService

### 実行時状態

TradingBotは `RuntimeState`（`src/app/runtime_state.py`）を通じて以下の状態を管理する。

```python
@dataclass
class RuntimeState:
    running: bool = False
    run_mode: RunMode = RunMode.PAPER
    phase: Phase = Phase.PHASE_1

    last_confirmed_h4_slot: str = ""
    next_4h_time: str = ""

    cycle_log: dict[str, dict] = field(default_factory=dict)

    gemini_cache: dict[str, dict] = field(default_factory=dict)
    groq_vote_cache: dict[str, dict | None] = field(default_factory=dict)

    safe_stop_reason: str | None = None
```

### Layer別コールバック

```python
class TradingBot:
    def _layer0_monitoring(self):
        """5分ごと: 軽量分析 + ダッシュボード更新"""

    def _layer1_entry_analysis(self):
        """4H確定時: フル分析 + エントリー判定"""

    def _layer2_position_management(self):
        """5分ごと: ポジション監視 + トレーリングストップ"""

    def _daily_reset(self):
        """日次: Gemini予算リセット + リコンシリエーション"""
```

---

## 3.2 Scheduler

**ファイル**: `src/app/scheduler.py`

### 責務

- Layer 0 / Layer 1 / Layer 2 の定期実行
- 日次リセット、週次レビュー、リコンシリエーションの起動
- 重複実行防止

### ジョブ一覧

| ジョブ | 実行タイミング | 目的 |
|-------|---------------|------|
| `layer0_job` | 5分ごと + 10秒 | 軽量監視分析 |
| `layer1_job` | 4H確定後 + 15秒 | 正式なエントリー判定 |
| `layer2_job` | 5分ごと + 40秒 | 既存ポジション監視 |
| `daily_reset_job` | UTC 00:00 | Gemini日次枠等の更新 |
| `weekly_review_job` | 週初 | 閾値レビュー |
| `reconciliation_job` | 起動時 + 定期 | OANDAとDB整合確認 |

### 4H足確定時刻（UTC）

`H4_SLOTS = (0, 4, 8, 12, 16, 20)`

### 主要メソッド

```python
class Scheduler:
    def current_4h_slot(self) -> str: ...
    def next_4h_display(self) -> str: ...
    def is_new_4h_candle(self, last_confirmed: str) -> bool: ...
    def should_run_layer0(self) -> bool: ...
    def should_run_layer2(self) -> bool: ...
    def should_daily_reset(self) -> bool: ...
    def should_weekly_review(self) -> bool: ...
    def run_loop(self, callbacks) -> None: ...
```

---

## 3.3 MarketDataManager

**ファイル**: `src/data/market_data.py`

### 責務

- OANDAから主系価格取得
- 補助データソース取得
- 複数時間足の同期
- データ整合性判定

### インターフェース

```python
class MarketDataManager:
    def get_account_info(self) -> dict: ...
    def get_multi_timeframe_data(self, instrument: str) -> dict[str, pd.DataFrame]: ...
    def get_latest_h4_close_slot(self, instrument: str) -> str: ...
    def get_open_positions(self) -> list[dict]: ...
    def get_current_pricing(self, instruments: list[str]) -> dict: ...
```

### 設計ポイント

- H4確定判定は**システム時計だけでなく、OANDAの最新確定足時刻を正**とする
- 時計ベースの判定は補助用途
- データ欠損時はそのペアのみスキップ可
- 主系データが広範囲に不達の場合は `ANALYSIS_ONLY` へ遷移

---

## 3.4 TechnicalAnalyzer

**ファイル**: `src/data/indicators.py`

### 責務

- SMA, EMA, RSI, MACD, BB, ATR, ADX の算出
- PatternDetector / StrategyEngine に渡す指標付きデータを作る

### 出力

- 元OHLCV
- SMA20/50/100/200
- EMA9/21/55/200
- RSI14
- MACDライン / シグナル / ヒストグラム
- BB上限 / 中央 / 下限
- ATR14
- ADX14

---

## 3.5 PatternDetector

**ファイル**: `src/strategies/pattern_detector.py`

### 責務

- 6パターンの検出
- Pattern Quality Score の算出
- 無効化価格・ブレイク妥当性の抽出
- Algorithm票の生成元情報の提供

### 対象パターン

- ダブルボトム
- ダブルトップ
- 逆ヘッドアンドショルダーズ
- ヘッドアンドショルダーズ
- チャネルブレイクアウト
- 移動平均クロスオーバー

### 出力モデル

```python
@dataclass
class PatternSignalResult:
    pattern: str
    direction: str
    pattern_score: float
    entry_ref_price: float
    invalidation_price: float
    stop_loss_price: float
    take_profit_price: float
    structural_rr: float
    meta: dict
```

---

## 3.6 StrategyEngine（分析統合）

**ファイル**: `src/strategies/strategy_engine.py`

### 責務

- PatternDetector と TechnicalAnalyzer を統合
- トレンド方向判定
- MTF整合チェック
- `use_gemini` による Quick / Full 分析切替
- 最終 SignalCandidate の生成

### analyze() フロー

```text
analyze(instrument, mtf_data, use_gemini, groq_vote_cache)
  1. 日足トレンド判定
  2. 4Hパターン検出
  3. Pattern Quality Score計算
  4. H1環境確認
  5. structural_rr判定
  6. use_gemini=True の場合は AIOrchestrator へ候補を渡す
  7. SignalCandidate を返す
```

### Quick分析（Layer 0向け）

- Gemini未使用
- Groqはキャッシュ利用
- ダッシュボード表示・次回候補抽出向け

### Full分析（Layer 1向け）

- Gemini予算通過候補のみGemini実行
- Groq方向性票をリフレッシュ

---

## 3.7 AIOrchestrator / AIAnalyzer / GroqReviewer

**ファイル**:

- `src/strategies/ai_analyzer.py`
- `src/strategies/groq_reviewer.py`
- `src/services/gemini_budget_service.py`

### 責務

- Geminiの画像分析
- Groqの方向性・レジーム分析
- 予算枠内でGemini対象を選抜
- レスポンスを共通 Vote 型に正規化

### GeminiBudgetService

#### 責務

- 日次ソフト上限16 / ハード上限18管理
- 4Hサイクルあたり上限3件
- 候補選抜
- スキップ理由記録
- 将来拡張可能な優先順位ロジックの提供

#### 初期実装方針（Phase 1）

Phase 1 では、選抜ロジックは **`pattern_score` 降順** を基本とする。
これは実装の単純性と挙動の追跡容易性を優先したものである。

#### 将来拡張方針

Phase 2 以降または改良時には、以下の合成優先度スコアへ拡張可能とする。

```text
priority_score =
  pair_pattern_performance_weight
+ mtf_alignment_weight
+ volatility_weight
+ session_weight
+ fairness_weight
```

#### 主要メソッド

```python
class GeminiBudgetService:
    def can_call(self) -> bool: ...
    def consume(self) -> None: ...
    def start_new_cycle(self) -> None: ...
    def select_candidates(self, candidates: list, max_select: int) -> list: ...
    def get_status(self) -> dict: ...
```

### AI出力の正規化

```python
@dataclass
class ConsensusVote:
    source: str              # algorithm / gemini / groq
    direction: str           # BUY / SELL / NONE / NEUTRAL
    confidence: float        # 0-100
    rationale: dict
```

---

## 3.8 ConsensusEngine

**ファイル**: `src/strategies/consensus_engine.py`

### 責務

- Algorithm / Gemini / Groq の票を統合
- 標準3者モードと2者フォールバックを判定
- reject理由を明示する
- 閾値を固定値ではなく設定値 / 状態値として扱う

### 閾値管理方針

エントリー閾値はコード固定値ではなく、以下を正とする。

1. `trading_config.yaml` の初期設定値
2. `system_state` に保存される現在値
3. 週次レビューによる範囲内更新

#### 初期値

- `three_vote_entry_threshold = 55`
- `fallback_entry_threshold = 62`

#### 許容範囲

- 最低: `50`
- 最高: `65`

### decide_entry ロジック

#### 標準3者モード（THREE_VOTE）

- Algorithm が `BUY` または `SELL`（ゲートキーパー）
- 3者中2者以上が同方向
- 多数派に **Algorithm が含まれる**
- 合意票平均confidenceが `three_vote_entry_threshold` 以上

#### 2者フォールバック（FALLBACK_TWO_VOTE）

- Gemini未実行 / 上限到達 / 障害 / 予算スキップ
- Algorithm と Groq が同方向
- 平均confidence が `fallback_entry_threshold` 以上
- `is_fallback=True` として記録

### decide_close ロジック

- Phase 1では**ログのみ**（実際のクローズなし）
- Phase 2以降:
  - Gemini と Groq が保有方向と反対方向
  - 両者 confidence ≥ 80
  - Gemini鮮度20分以内
  - 条件成立時のみ `AI_CLOSE`

### 出力モデル

```python
@dataclass
class ConsensusResult:
    direction: str
    confidence: float
    votes: list[ConsensusVote]
    consensus_reached: bool
    agree_count: int
    total_votes: int
    mode: str                  # THREE_VOTE / FALLBACK_TWO_VOTE
    reject_reason: str | None
```

---

## 3.9 RiskManager

**ファイル**: `src/models/risk_manager.py`

### 責務

- ロット計算
- JPY口座向け pip value 換算
- 新規エントリー可否判定
- 日次/週次損失制限
- DD防御
- 連敗制御
- Phase別適用

### 主要クラス

```python
class RiskManager:              # メインリスク制御
class AdaptiveRiskController:   # 適応的リスク計算
class PerformanceTracker:       # 直近20トレード勝率追跡
class MarketRegime:             # レジーム状態
class RiskState:                # リスク状態スナップショット
class PositionSize:             # 計算済みポジションサイズ
```

### 追加責務: pip value 換算

口座通貨は **JPY** を前提とする。
RiskManager はポジションサイズ計算に必要な **pip value per unit in JPY** の換算責務を持つ。

#### 必須メソッド

```python
class RiskManager:
    def calculate_pip_value_per_unit(
        self,
        instrument: str,
        price_map: dict[str, float],
        account_currency: str = "JPY"
    ) -> float: ...

    def calculate_position_size(
        self,
        nav: float,
        risk_pct: float,
        stop_distance_pips: float,
        pip_value_per_unit: float,
        is_fallback: bool = False
    ) -> PositionSize: ...
```

### ポジションサイズ計算式

```text
units = floor(
    (NAV × applied_risk_pct)
    / (stop_distance_pips × pip_value_per_unit_jpy)
)
```

### フォールバック時のサイズ調整

2者フォールバック時は、通常算出サイズに対して **80%** を適用する。

```text
fallback_units = floor(normal_units × 0.8)
```

### 適応的リスク計算

```python
def calculate_risk_multiplier(self, current_dd, phase) -> float:
    multiplier = 1.0
    if phase >= Phase.PHASE_2:
        multiplier *= loss_streak_factor()
        multiplier *= drawdown_factor(current_dd)
    if phase >= Phase.PHASE_3:
        multiplier *= regime_factor()
        multiplier *= performance_factor()
    return clamp(multiplier, 0.25, 1.00)
```

### 動的ポジション上限

仕様書確定稿に合わせる。

```python
def dynamic_max_positions(self, current_dd, phase) -> int:
    # multiplier >= 0.8 → 6
    # multiplier >= 0.5 → 5
    # multiplier >= 0.3 → 4
    # multiplier <  0.3 → 3
```

### Phase適用

- Phase 1: 固定0.5%を実運用、適応的リスクはshadow計算（ログのみ出力）
- Phase 2: 連敗係数とDD係数を適用
- Phase 3: Regime / Performance係数も適用

### 拒否条件

- 1ペア上限ポジション違反
- 同時保有上限超過
- 通貨偏重超過（初期実装は件数ベース）
- DD / 日次 / 週次損失超過
- 5連敗停止中
- structural_rr不足
- シグナル失効
- スプレッド異常

### 通貨偏重制御の実装方針

初期実装では **件数ベースの通貨エクスポージャー制御** を採用する。

- `MAX_CURRENCY_EXPOSURE = 3`

将来拡張として、**相関バスケット最大想定損失 `2.0% NAV`** のチェック追加を可能とする。

---

## 3.10 BrokerClient 抽象化

**ファイル**: `src/brokers/base_broker.py`

### 共通インターフェース

```python
class BrokerClient(ABC):
    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def ensure_session(self) -> bool: ...
    @abstractmethod
    def get_account_info(self) -> dict: ...
    @abstractmethod
    def get_historical_prices(self, instrument, resolution, count) -> pd.DataFrame: ...
    @abstractmethod
    def get_market_info(self, instrument) -> dict: ...
    @abstractmethod
    def get_positions(self) -> list[dict]: ...
    @abstractmethod
    def get_open_positions(self) -> list[dict]: ...
    @abstractmethod
    def open_position(self, instrument, direction, size, stop, limit) -> dict: ...
    @abstractmethod
    def close_position(self, deal_id) -> dict: ...
    @abstractmethod
    def update_position(self, deal_id, stop, limit) -> dict: ...
```

### OANDA返却例

```python
get_account_info() -> {
    "balance": float,
    "available": float,
    "unrealized_pl": float,
    "nav": float,
    "currency": "JPY"
}

get_positions() -> [{
    "deal_id": str,
    "instrument": str,
    "direction": str,
    "size": float,
    "entry_price": float,
    "unrealized_pl": float
}]
```

### 注意

- `realized_pl` は OANDAの累積 `pl` を直接使わない
- 日次損益は自前台帳集計を正とする

---

## 3.11 ExecutionService

**ファイル**: `src/services/execution_service.py`

### 責務

- 発注前最終検証
- MARKET注文送信
- 約定結果保存
- クローズ処理
- スリッページ記録
- 通知発行

### シグナル有効期限

新規エントリーシグナルは **4H確定後15分以内** のみ有効とする。
これを超過したシグナルは失効扱いとし、追いかけ発注は行わない。

#### 必須設定値

- `signal_expiry_minutes = 15`

### 発注前チェック

- 実行モードが PAPER
- シグナル失効前
- 同一ペア未保有
- スプレッド許容内
- submit price が decision price から大きく乖離していない
- SAFE_STOP / ANALYSIS_ONLY ではない

### スリッページ記録

- decision price
- submit reference price
- fill price
- total slippage

### ペア別許容スリッページ

| ペア群 | 許容値 |
|--------|--------|
| EUR/USD, GBP/USD | 1.5 pips |
| USD/JPY, EUR/JPY, GBP/JPY, CHF_JPY | 2.0 pips |

### 補足

- OANDA注文時に可能であれば `priceBound` を使用する
- 許容スリッページ超過時は発注しない
- 失効後再送は禁止する

---

## 3.12 PositionStore / Repository

**ファイル**: `src/models/position_store.py`

### 方針

Phase 1 では、既存構成との整合性を優先し、単一 `positions` テーブルを使用する。
ただし、本テーブルのみでは将来的な検証・KPI・AI追跡に不足するため、拡張前提の設計とする。

### 現行テーブル

```sql
CREATE TABLE positions (
    position_id TEXT PRIMARY KEY,
    broker_trade_id TEXT,
    signal_id TEXT,
    instrument TEXT,
    direction TEXT,
    entry_price REAL,
    size REAL,
    stop_loss REAL,
    take_profit REAL,
    opened_at TEXT,
    closed_at TEXT,
    exit_price REAL,
    pnl REAL,
    exit_reason TEXT,
    pattern TEXT,
    confidence REAL,
    is_fallback INTEGER DEFAULT 0
);
```

### Phase 1 実装方針

Phase 1 では以下を `positions` テーブル中心に管理する。

- オープンポジション
- クローズ済みポジション
- 基本的な損益
- fallbackフラグ

### 将来拡張候補テーブル

- `signals` — シグナル記録
- `ai_votes` — AI投票記録
- `account_snapshots` — 口座スナップショット
- `api_usage_daily` — API使用量追跡
- `pattern_stats` — パターン別成績

### 補足

Phase 1 では現行テーブルで開始してよいが、
Phase 2 以降の検証強化時には `signals` / `ai_votes` / `account_snapshots` の追加を優先する。

---

## 3.13 PortfolioService / ReconciliationService

**ファイル**:

- `src/services/portfolio_service.py`
- `src/services/reconciliation_service.py`

### PortfolioService

```python
@dataclass
class PortfolioState:
    nav: float = 0.0
    balance: float = 0.0
    unrealized_pl: float = 0.0
    drawdown_pct: float = 0.0
    peak_nav: float = 0.0
    open_position_count: int = 0
    currency_exposure: dict[str, int] = field(default_factory=dict)

    # 日次・週次基準
    daily_nav_anchor: float = 0.0
    weekly_nav_anchor: float = 0.0
    daily_loss_pct: float = 0.0
    weekly_loss_pct: float = 0.0

    # 補助表示用
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
```

#### 設計方針

日次 / 週次損失は、実現損益だけでなく **NAV基準** で評価する。
そのため、`daily_nav_anchor` および `weekly_nav_anchor` を保持する。

#### 主要メソッド

```python
class PortfolioService:
    def update_from_broker(self, account_info: dict) -> None: ...
    def update_positions(self, positions: list[dict]) -> None: ...
    def record_trade_pnl(self, pnl: float) -> None: ...
    def check_currency_exposure(self, instrument: str) -> tuple[bool, str]: ...

    def roll_daily_anchor(self, nav: float) -> None: ...
    def roll_weekly_anchor(self, nav: float) -> None: ...
    def calculate_daily_loss_pct(self, nav: float) -> float: ...
    def calculate_weekly_loss_pct(self, nav: float) -> float: ...
```

#### 基準ルール

- `daily_nav_anchor`: UTC 00:00 の NAV
- `weekly_nav_anchor`: 週初 UTC の NAV
- `daily_loss_pct`: `(daily_nav_anchor - current_nav) / daily_nav_anchor × 100`
- `weekly_loss_pct`: `(weekly_nav_anchor - current_nav) / weekly_nav_anchor × 100`

### ReconciliationService

#### 起動時復元フロー

```text
1. SQLiteから open positions を読み込み
2. OANDAの open trades を取得
3. OANDAを真実源として差分比較
4. DBに不足があれば補完
5. DBにだけ存在してOANDAにないものはクローズ扱いで整合
6. signal_id / broker_trade_id の対応を検証
7. 不整合は trading.log に記録
```

#### 注意

OANDAを**真実源**とする。旧設計の「ブローカーの paper_positions に復元」は採用しない。

---

## 3.14 Webダッシュボード

**ファイル**:

- `src/dashboard.py` — 既存Flaskダッシュボード（メイン）
- `src/web/dashboard.py` — Flask App ファクトリ（API Blueprint登録）
- `src/web/routes.py` — API Blueprint

### アーキテクチャ

```text
Flask App
  ├─ GET /
  ├─ GET /api/summary
  ├─ GET /api/signals/recent
  ├─ GET /api/system/status
  ├─ GET /api/data              # 互換性維持の集約API
  └─ 30秒ポーリング
```

### `/api/data` 集約レスポンス（旧ダッシュボード互換）

```json
{
  "account": { "balance": 0, "unrealized_pl": 0, "daily_realized_pl": 0, "nav": 0 },
  "pairs": [],
  "cycle_log": {},
  "trade_history": [],
  "regime": {}
}
```

### `cycle_log` の保持内容

各通貨ペアごとに以下を保持する。

```python
_cycle_log[instrument] = {
    "algorithm_vote": {...},
    "gemini_vote": {...},
    "groq_vote": {...},
    "consensus": {...},
    "gemini_status": {
        "active": bool,
        "skip_reason": str | None,
        "budget_used": int,
        "budget_limit": int,
        "cached": bool,
        "next_4h_time": str
    },
    "risk_check": {
        "passed": bool,
        "reject_reason": str | None
    },
    "updated_at": str
}
```

---

## 4. ドメインモデル設計

**ファイル**: `src/domain/models.py`, `src/domain/enums.py`

### 4.1 Enums

```python
class RunMode(str, Enum):
    PAPER = "paper"
    BACKTEST = "backtest"
    ANALYSIS = "analysis"
    SAFE_STOP = "safe_stop"
    ANALYSIS_ONLY = "analysis_only"

class Phase(IntEnum):
    PHASE_1 = 1
    PHASE_2 = 2
    PHASE_3 = 3

class ConsensusMode(str, Enum):
    THREE_VOTE = "three_vote"
    FALLBACK_TWO_VOTE = "fallback_two_vote"

class VoteSource(str, Enum):
    ALGORITHM = "algorithm"
    GEMINI = "gemini"
    GROQ = "groq"

class GeminiSkipReason(str, Enum):
    NOT_H4_CLOSE = "not_h4_close"
    NO_PATTERN = "no_pattern"
    SCORE_TOO_LOW = "score_too_low"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NOT_SELECTED_BY_PRIORITY = "not_selected_by_priority"
    API_ERROR = "api_error"
```

### 4.2 ConsensusVote

```python
@dataclass
class ConsensusVote:
    source: str                # algorithm / gemini / groq
    direction: str             # BUY / SELL / NONE / NEUTRAL
    confidence: float          # 0-100
    rationale: dict
```

### 4.3 ConsensusResult

```python
@dataclass
class ConsensusResult:
    direction: str
    confidence: float
    votes: list[ConsensusVote]
    consensus_reached: bool
    agree_count: int
    total_votes: int
    mode: str                  # THREE_VOTE / FALLBACK_TWO_VOTE
    reject_reason: str | None
```

### 4.4 SignalCandidate

```python
@dataclass
class SignalCandidate:
    signal_id: str
    instrument: str
    direction: str
    pattern: str
    pattern_score: float
    entry_ref_price: float
    invalidation_price: float
    stop_loss_price: float
    take_profit_price: float
    structural_rr: float
    algo_confidence: float
    candle_close_time: str = ""
```

### 4.5 OrderIntent

```python
@dataclass
class OrderIntent:
    signal_id: str
    instrument: str
    direction: str
    units: int
    stop_loss: float
    take_profit: float
    risk_pct: float
    is_fallback: bool = False
    expires_at: str = ""
```

---

## 5. データフロー設計

## 5.1 Layer 0（5分ごと + 10秒）

```text
OANDA価格取得
  → 指標計算
  → PatternDetector
  → StrategyEngine(Quick)
  → Groqキャッシュ参照
  → cycle_log更新
  → ダッシュボード表示
```

### 特徴

- 新規エントリーなし
- Gemini未使用
- 監視・可視化用

---

## 5.2 Layer 1（4H確定 + 15秒）

```text
OANDA H4確定足取得
  → データ整合性確認
  → PatternDetector
  → SignalCandidate生成
  → GeminiBudgetServiceで選抜
  → Gemini/Groq分析
  → ConsensusEngine
  → RiskManager
  → ExecutionService
  → PositionStore保存
  → Notification
```

### 新規エントリーはここだけ許可

---

## 5.3 Layer 2（5分ごと + 40秒）

```text
OANDA open trades取得
  → PortfolioState更新
  → Trail更新
  → 価格ベースクローズ判定
  → Phase 2以降なら AIクローズ条件確認
  → クローズ処理
  → Snapshot保存
```

---

## 6. 4H足確定検出設計

### 6.1 基本方針

時刻スロット判定を採用するが、**最終判定はOANDAの最新確定H4足時刻**で行う。

### 6.2 スロット計算（補助）

```python
H4_SLOTS = (0, 4, 8, 12, 16, 20)
current_slot = f"{utc_now:%Y-%m-%d} {(utc_now.hour // 4) * 4:02d}:00"
```

### 6.3 正式判定

- OANDAから取得した最新確定H4足の `close_time` を `confirmed_slot` とする
- `confirmed_slot != _last_confirmed_h4_slot` の場合のみ Layer 1 を実行

### 利点

- サーバ時計ずれやAPI反映遅れに強い
- バックテスト時も同じ概念で再現できる

---

## 7. Gemini予算管理フロー

```text
4H足確定?
  ├─ No → Geminiスキップ (NOT_H4_CLOSE)
  └─ Yes
       ├→ Algorithmが候補検出?
       │   ├─ No → skip_reason="NO_PATTERN"
       │   └─ Yes
       │        ├→ PatternScore >= 60 ?
       │        │   ├─ No → skip_reason="SCORE_TOO_LOW"
       │        │   └─ Yes
       │        │        ├→ 4Hサイクル上限内? (3件/サイクル)
       │        │        ├→ 日次上限内? (ソフト16/ハード18)
       │        │        └→ 候補選抜
       │        │              ├─ Phase 1: pattern_score 降順
       │        │              └─ 将来拡張: performance / MTF / volatility / session / fairness
       │        └→ 採用候補のみ Gemini 実行
       └→ Groq方向性投票リフレッシュ
```

---

## 8. エラーハンドリング設計

### 8.1 API障害時の動作

| 障害 | 動作 |
|------|------|
| OANDA認証失敗（起動時） | PAPER起動不可。ANALYSIS_ONLY へ遷移 |
| OANDA価格取得一時失敗 | リトライ、失敗継続で ANALYSIS_ONLY |
| OANDA発注失敗 | 当該注文を失敗記録、再試行は限定的 |
| Gemini API 429 | 新規Gemini停止、2者フォールバック判定 |
| Groq API 429 | キャッシュ利用。キャッシュ不在なら新規エントリー禁止 |
| Gemini/Groq一時エラー | 当該ペアを skip、GeminiSkipReason記録 |
| SQLite書込失敗 | SAFE_STOP |
| Dashboard失敗 | 取引継続可、ログ記録 |

### 8.2 モード遷移

```text
PAPER
  ├→ ANALYSIS_ONLY   : OANDA主系価格不達
  ├→ SAFE_STOP       : DD超過 / DB障害 / 致命的不整合
ANALYSIS_ONLY
  ├→ PAPER           : OANDA復旧
  └→ SAFE_STOP       : 致命的障害
```

---

## 9. セキュリティ設計

### 9.1 シークレット管理

- 全APIキーは `.env` で管理
- `.env` は `.gitignore`
- `.env.example` のみ公開

### 9.2 保護対象

| シークレット | 保管場所 |
|-------------|---------|
| OANDA_API_TOKEN | .env |
| OANDA_ACCOUNT_ID | .env |
| GEMINI_API_KEY | .env |
| GROQ_API_KEY | .env |
| LINE_CHANNEL_ACCESS_TOKEN | .env |
| SMTP_PASSWORD | .env |
| DISCORD_WEBHOOK_URL | .env |

### 9.3 通信

- すべてHTTPS
- Bearer Token / API Key認証
- ダッシュボードはローカルホスト bind をデフォルトとする

---

## 10. テスト設計

### 10.1 現行テスト

| テストファイル | 対象 |
|--------------|------|
| `test_patterns.py` | 6パターン検出 |
| `test_risk_manager.py` | リスク管理・Phase制御 |
| `test_indicators.py` | テクニカル指標計算 |
| `test_backtest.py` | バックテスト整合性 |
| `test_ma_cross.py` | MAクロス戦略 |

### 10.2 追加推奨テスト

| テストファイル | 対象 |
|--------------|------|
| `test_consensus.py` | 3者投票 / 2者フォールバック |
| `test_budget_service.py` | Gemini予算制御 |
| `test_execution.py` | 発注・クローズ |
| `test_reconciliation.py` | 起動時整合 |
| `test_portfolio.py` | NAV/DD/エクスポージャー |
| `test_scheduler.py` | 4H検出・Layer制御 |

---

## 11. 実装上の重要注意点

1. `TradingBot` にロジックを寄せすぎないこと（薄いオーケストレータを維持）
2. OANDA `pl` を日次実現損益として扱わないこと（自前台帳 + NAV anchor が正）
3. 起動時は必ず OANDA と DB を照合すること（ReconciliationService）
4. Gemini価格レベルは参考に留めること
5. フォールバック発動理由を必ず保存すること（GeminiSkipReason）
6. 4H確定足の終値即約定をバックテストで再現しないこと
7. ポジション上限は `6/5/4/3`（旧版の `8/6/4/3` は採用しない）
8. Phase 1ではAIクローズを実行しないこと（ログのみ）
9. ロット計算では必ず `pip_value_per_unit_jpy` を経由すること
10. Consensus閾値はコード固定値ではなく設定値 / 状態値を使用すること
11. シグナル有効期限15分を厳守すること
12. ダッシュボードの互換ルートと新APIルートの責務を混同しないこと

---

## 12. 最終要約

本統合改訂版 v1.3 では、以下を統合した。

- **実務的な具体性**（ファイル構成、cycle_log、障害対応）
- **責務分離と安全設計**（サービス分離、Phase制御、モード遷移）
- **仕様書との整合性**（6/5/4/3上限、2者フォールバック、Gemini予算管理）
- **実装前の整合修正**（閾値の設定化、JPY換算、失効時間、anchor管理）

特に重要な最終方針:

1. `main.py → TradingBot` の構造は維持する
2. `TradingBot` は**薄いオーケストレータ**とする
3. 分析・AI・投票・リスク・執行を分離する
4. 4H足確定時のみ新規エントリーする
5. Geminiは予算管理下で厳選利用する（ソフト16/ハード18/サイクル3）
6. Gemini不在時は 2者フォールバックを明示運用する
7. OANDAを真実源として復元・整合する
8. ダッシュボード向け `cycle_log` を正式に採用する
9. Phase制御で段階的にリスク管理を有効化する
10. リスク計算は JPY口座ベースの pip value 換算を前提とする
11. 損失制限は daily / weekly NAV anchor を基準とする

以上を、AI FX Trading Tool の**統合改訂版設計書 v1.3**とする。

---

# 付録A: SQLite DDL（推奨完全版）

## A.1 方針

- SQLite を永続化ストアとして使用する
- タイムスタンプは **UTC ISO8601文字列** で保存する
- Phase 1 では `positions` を主利用してよいが、将来のKPI・AI追跡・バックテスト整合のため、**拡張テーブルも先に作成することを推奨**する
- OANDA が真実源であり、DBは **監査・復元・可観測性のための永続化ストア** と位置づける

## A.2 初期化方針

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
```

## A.3 テーブル定義

### A.3.1 positions

```sql
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    broker_trade_id TEXT,
    signal_id TEXT,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
    entry_price REAL NOT NULL,
    size REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    exit_price REAL,
    pnl REAL,
    exit_reason TEXT,
    pattern TEXT,
    confidence REAL,
    is_fallback INTEGER NOT NULL DEFAULT 0 CHECK(is_fallback IN (0,1))
);
```

```sql
CREATE INDEX IF NOT EXISTS idx_positions_instrument ON positions(instrument);
CREATE INDEX IF NOT EXISTS idx_positions_broker_trade_id ON positions(broker_trade_id);
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at);
CREATE INDEX IF NOT EXISTS idx_positions_closed_at ON positions(closed_at);
CREATE INDEX IF NOT EXISTS idx_positions_signal_id ON positions(signal_id);
```

### A.3.2 signals

```sql
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
    pattern TEXT NOT NULL,
    pattern_score REAL NOT NULL,
    entry_ref_price REAL NOT NULL,
    invalidation_price REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    structural_rr REAL NOT NULL,
    algo_confidence REAL NOT NULL,
    candle_close_time TEXT NOT NULL,
    consensus_mode TEXT,
    consensus_result TEXT,
    consensus_confidence REAL,
    reject_reason TEXT,
    is_fallback INTEGER NOT NULL DEFAULT 0 CHECK(is_fallback IN (0,1)),
    gemini_skip_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
```

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_signals_natural
ON signals(instrument, candle_close_time, direction, pattern);

CREATE INDEX IF NOT EXISTS idx_signals_instrument ON signals(instrument);
CREATE INDEX IF NOT EXISTS idx_signals_candle_close_time ON signals(candle_close_time);
CREATE INDEX IF NOT EXISTS idx_signals_pattern ON signals(pattern);
CREATE INDEX IF NOT EXISTS idx_signals_consensus_result ON signals(consensus_result);
```

### A.3.3 ai_votes

```sql
CREATE TABLE IF NOT EXISTS ai_votes (
    vote_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('algorithm', 'gemini', 'groq')),
    direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL', 'NONE', 'NEUTRAL')),
    confidence REAL NOT NULL,
    rationale_json TEXT,
    freshness_seconds INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(signal_id) REFERENCES signals(signal_id) ON DELETE CASCADE
);
```

```sql
CREATE INDEX IF NOT EXISTS idx_ai_votes_signal_id ON ai_votes(signal_id);
CREATE INDEX IF NOT EXISTS idx_ai_votes_source ON ai_votes(source);
CREATE INDEX IF NOT EXISTS idx_ai_votes_created_at ON ai_votes(created_at);
```

### A.3.4 orders

```sql
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    signal_id TEXT,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
    requested_units REAL NOT NULL,
    filled_units REAL,
    decision_price REAL,
    submit_reference_price REAL,
    fill_price REAL,
    stop_loss REAL,
    take_profit REAL,
    risk_pct REAL,
    total_slippage_pips REAL,
    order_status TEXT NOT NULL,
    broker_order_id TEXT,
    failure_reason TEXT,
    is_fallback INTEGER NOT NULL DEFAULT 0 CHECK(is_fallback IN (0,1)),
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(signal_id) ON DELETE SET NULL
);
```

```sql
CREATE INDEX IF NOT EXISTS idx_orders_signal_id ON orders(signal_id);
CREATE INDEX IF NOT EXISTS idx_orders_instrument ON orders(instrument);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
```

### A.3.5 trades

```sql
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    broker_trade_id TEXT,
    signal_id TEXT,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    size REAL NOT NULL,
    pnl REAL NOT NULL,
    realized_r REAL,
    hold_minutes INTEGER,
    close_reason TEXT,
    is_fallback INTEGER NOT NULL DEFAULT 0 CHECK(is_fallback IN (0,1)),
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(position_id) REFERENCES positions(position_id) ON DELETE CASCADE,
    FOREIGN KEY(signal_id) REFERENCES signals(signal_id) ON DELETE SET NULL
);
```

```sql
CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);
```

### A.3.6 account_snapshots

```sql
CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    captured_at TEXT NOT NULL,
    balance REAL NOT NULL,
    nav REAL NOT NULL,
    unrealized_pl REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    peak_nav REAL NOT NULL,
    daily_nav_anchor REAL NOT NULL,
    weekly_nav_anchor REAL NOT NULL,
    daily_loss_pct REAL NOT NULL,
    weekly_loss_pct REAL NOT NULL,
    open_position_count INTEGER NOT NULL,
    currency_exposure_json TEXT,
    run_mode TEXT,
    phase INTEGER
);
```

```sql
CREATE INDEX IF NOT EXISTS idx_account_snapshots_captured_at ON account_snapshots(captured_at);
```

### A.3.7 api_usage_daily

```sql
CREATE TABLE IF NOT EXISTS api_usage_daily (
    usage_date TEXT NOT NULL,
    provider TEXT NOT NULL CHECK(provider IN ('gemini', 'groq')),
    used_count INTEGER NOT NULL DEFAULT 0,
    soft_limit INTEGER,
    hard_limit INTEGER,
    cycle_limit INTEGER,
    reserve_used INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (usage_date, provider)
);
```

### A.3.8 system_state

```sql
CREATE TABLE IF NOT EXISTS system_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

初期投入推奨キー:

- `run_mode`
- `phase`
- `three_vote_entry_threshold`
- `fallback_entry_threshold`
- `safe_stop`
- `safe_stop_reason`
- `last_confirmed_h4_slot`
- `daily_nav_anchor`
- `weekly_nav_anchor`
- `peak_nav`

### A.3.9 pattern_stats

```sql
CREATE TABLE IF NOT EXISTS pattern_stats (
    stat_id TEXT PRIMARY KEY,
    instrument TEXT NOT NULL,
    pattern TEXT NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0,
    gross_profit REAL NOT NULL DEFAULT 0,
    gross_loss REAL NOT NULL DEFAULT 0,
    profit_factor REAL,
    updated_at TEXT NOT NULL
);
```

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_pattern_stats_instrument_pattern
ON pattern_stats(instrument, pattern);
```

### A.3.10 execution_events

```sql
CREATE TABLE IF NOT EXISTS execution_events (
    event_id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    category TEXT NOT NULL,
    instrument TEXT,
    signal_id TEXT,
    position_id TEXT,
    message TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT NOT NULL
);
```

```sql
CREATE INDEX IF NOT EXISTS idx_execution_events_created_at ON execution_events(created_at);
CREATE INDEX IF NOT EXISTS idx_execution_events_category ON execution_events(category);
CREATE INDEX IF NOT EXISTS idx_execution_events_signal_id ON execution_events(signal_id);
```

## A.4 Phase 1 最低利用対象

Phase 1 実装で最低限使用するテーブルは以下とする。

- `positions`
- `orders`
- `trades`
- `account_snapshots`
- `api_usage_daily`
- `system_state`
- `execution_events`

Phase 1 でも作成だけは推奨するが、利用を段階化してよいテーブルは以下。

- `signals`
- `ai_votes`
- `pattern_stats`

## A.5 監査・整合の原則

- OANDA の open trades が真実源
- DB は監査・復元・KPI・分析再現のために利用
- 再起動時は `positions` を鵜呑みにせず、必ず Reconciliation を通す

---

# 付録B: `trading_config.yaml` 設計版

## B.1 方針

- 秘密情報は `.env`
- 運用・戦略・閾値・対象ペア・スケジューラ条件は `trading_config.yaml`
- 閾値の初期値は YAML に置き、可変値は `system_state` で上書き可能とする

## B.2 YAMLサンプル

```yaml
app:
  name: "AI FX Trading Tool"
  mode: "paper"
  phase: 1
  timezone: "UTC"
  account_currency: "JPY"

broker:
  provider: "oanda"
  environment: "practice"
  account_id_env: "OANDA_ACCOUNT_ID"
  token_env: "OANDA_API_TOKEN"

data:
  primary_source: "oanda"
  backup_source: "twelvedata"
  warmup_bars:
    day: 300
    h4: 500
    h1: 500
    m5: 300
  allow_backup_analysis_only: true

pairs:
  active:
    - "USD_JPY"
    - "EUR_USD"
    - "GBP_JPY"
    - "EUR_JPY"
    - "GBP_USD"
    - "CHF_JPY"
  registered:
    - "AUD_USD"

timeframes:
  trend: "DAY"
  signal: "H4"
  entry_context: "H1"

scheduler:
  layer0_interval_minutes: 5
  layer0_offset_seconds: 10
  layer1_h4_offset_seconds: 15
  layer2_interval_minutes: 5
  layer2_offset_seconds: 40
  h4_slots_utc: [0, 4, 8, 12, 16, 20]

strategy:
  pattern_score_min: 60
  structural_rr_min: 1.2
  allowed_patterns:
    - "double_bottom"
    - "double_top"
    - "inverse_head_shoulders"
    - "head_shoulders"
    - "channel_breakout"
    - "ma_crossover"

indicators:
  sma: [20, 50, 100, 200]
  ema: [9, 21, 55, 200]
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  bb_sigma: 2
  atr_period: 14
  adx_period: 14
  adx_trend_threshold: 25

ai:
  gemini:
    enabled: true
    model: "gemini-2.5-flash"
    daily_soft_limit: 16
    daily_hard_limit: 18
    per_h4_cycle_limit: 3
    chart_width: 1200
    chart_height: 800
    ai_close_freshness_minutes: 20

  groq:
    enabled: true
    model: "llama-3.3-70b-versatile"
    regime_ttl_minutes: 60

consensus:
  three_vote_entry_threshold_initial: 55
  fallback_entry_threshold_initial: 62
  threshold_min: 50
  threshold_max: 65
  threshold_adjust_step: 5
  ai_close_min_confidence: 80

risk:
  base_risk_pct_phase1: 0.5
  min_risk_pct: 0.25
  max_risk_pct: 1.0
  stop_loss_atr_multiple: 2.0
  take_profit_rr: 2.0
  trailing_stop_atr_multiple: 1.5

  max_drawdown_pct: 10.0
  max_daily_loss_pct: 2.0
  max_weekly_loss_pct: 5.0
  max_consecutive_losses: 5
  cooldown_hours_after_max_losses: 24

  dynamic_position_limits:
    ge_0_8: 6
    ge_0_5: 5
    ge_0_3: 4
    lt_0_3: 3

  max_positions_per_pair: 1
  max_currency_exposure_count: 3

  adaptive:
    enabled_phase2: true
    enabled_phase3: true
    loss_scale_factor: 0.75
    drawdown_scales:
      lt_3: 1.0
      ge_3: 0.75
      ge_5: 0.50
      ge_7: 0.25
    performance_pf_scales:
      ge_1_4: 1.0
      ge_1_0: 0.85
      lt_1_0: 0.70
    regime_scales:
      trend: 1.0
      range: 0.75
      high_volatility: 0.50

execution:
  signal_expiry_minutes: 15
  spread_filter_multiplier: 1.5
  use_price_bound: true

  slippage_limits_pips:
    EUR_USD: 1.5
    GBP_USD: 1.5
    USD_JPY: 2.0
    EUR_JPY: 2.0
    GBP_JPY: 2.0
    CHF_JPY: 2.0

portfolio:
  use_nav_anchors: true
  anchor_roll_timezone: "UTC"

dashboard:
  enabled: true
  host: "127.0.0.1"
  port: 5000
  refresh_seconds: 30

notifications:
  line_enabled: true
  email_enabled: true
  discord_enabled: true
  notify_on_startup: true
  notify_on_entry: true
  notify_on_close: true
  notify_on_drawdown_alert: true
  notify_on_daily_summary: true
  notify_on_error: true

backtest:
  enabled: true
  initial_capital_jpy: 1000000
  in_sample_ratio: 0.7
  out_of_sample_ratio: 0.3
  slippage_model: "conservative"
  spread_model: "median_or_fallback"
  fill_model: "next_bar_or_next_m1"
  forbid_same_bar_fill: true
```

## B.3 `.env` に置くもの

以下は `trading_config.yaml` ではなく `.env` に置く。

- `OANDA_API_TOKEN`
- `OANDA_ACCOUNT_ID`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `LINE_CHANNEL_ACCESS_TOKEN`
- `SMTP_HOST`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `DISCORD_WEBHOOK_URL`

## B.4 実装ルール

- `config_loader.py` は YAML を読み込み、型安全な設定オブジェクトへ変換する
- `.env` の欠落は起動時に検知する
- 閾値は YAML 初期値を採用しつつ、`system_state` に現在値があればそちらを優先する
