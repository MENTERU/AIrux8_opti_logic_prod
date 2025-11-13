# エアコン最適化システム (Air Conditioning Optimization System)

## 概要 (Overview)

エアコンの設定温度とモードを最適化し、電力消費を最小化するシステムです。営業時間内の室温制約を考慮した期間最適化スケジュールを生成します。
(A system that optimizes air conditioning temperature settings and modes to minimize power consumption, generating period optimization schedules considering indoor temperature constraints during business hours.)

## セットアップ (Setup)

### 1. プライベート情報ファイルの作成 (Creating Private Information File)

システムを使用する前に、`config/private_information.py` ファイルを作成し、以下の変数を設定してください：
(Before using the system, create a `config/private_information.py` file and set the following variables:)

```python
# config/private_information.py
# gmailのメールアドレス (Gdriveから取得する場合。現在停止中)
ACCESS_INFORMATION = "name@menteru.jp" または(or)　ACCESS_INFORMATION = "name@gmail.com"
# visual crossing Weather API Key
WEATHER_API_KEY = "weather_api_key_here"
```

**注意 (Note)**: このファイル`config/private_information.py`は `.gitignore` に含まれているため、Git にコミットされません。各開発者が個別に作成する必要があります。
(This file `config/private_information.py` is included in `.gitignore` and will not be committed to Git. Each developer needs to create it individually.)

### 3. データフォルダの準備 (Data Folder Setup)

#### ローカルパスを使用する場合（推奨）(Using Local Path - Recommended)

ローカルパスを使用する場合は、データフォルダをプロジェクトのルートディレクトリに配置してください。
(When using local paths, place the data folder in the project root directory.)

**🗂️フォルダ構造 (Folder Structure):**

```bash
AIrux8_opti_logic/
├── main.py                  # メインエントリーポイント (Main Entry Point)
├── config/                  # 設定ファイル (Configuration Files)
├── processing/              # データ処理モジュール (Data Processing Modules)
├── optimization/            # 最適化モジュール (Optimization Modules)
├── analysis/                # 分析・可視化モジュール (Analysis & Visualization)
├── pyproject.toml           # プロジェクト設定 (Project Configuration)
├── uv.lock                  # 依存関係ロックファイル (Dependency Lock File)
└── data/                    # ← このフォルダをダウンロードして配置 (Download & Place This Folder)
    ├── 00_InputData/        # 生データ (Raw Data) 
    ├── 01_MasterData/       # マスターデータ (Master Data)
    ├── 02_PreprocessedData/ # 前処理済みデータ（自動生成）(Preprocessed Data - Auto Generated)
    ├── 03_Models/           # 学習済みモデル（自動生成）(Trained Models - Auto Generated)
    ├── 04_PlanningData/     # 計画データ（自動生成）(Planning Data - Auto Generated)
    └── 05_ValidationResults/# 検証結果（自動生成）(Validation Results - Auto Generated)
```

**データフォルダの取得方法 (How to Get Data Folder):**

1. Google Drive または共有ストレージから `data/` フォルダをダウンロード (Download `data/` folder from Google Drive or shared storage)
2. プロジェクトのルートディレクトリ（`AIrux8_opti_logic/`）に配置 (Place it in the project root directory)
3. フォルダ構造が上記の通りになっていることを確認 (Verify the folder structure matches above)

## 実行方法 (Execution Methods)

### 基本的な実行コマンド (Basic Execution Commands)

```bash
# フルパイプライン実行（前処理→学習→最適化）(Full Pipeline - Preprocessing→Training→Optimization)
uv run main.py

# 特定のストアで実行 (Run for specific store)
uv run main.py --store Clea

# 特定の期間で実行 (Run for specific period)
uv run main.py --start-date 2024-01-01 --end-date 2024-01-02
```

### 段階別実行フラグ (Step-by-Step Execution Flags)

```bash
# 前処理のみ実行 (Preprocessing only)
uv run main.py --preprocess-only

# 集約のみ実行 (Aggregation only)
uv run main.py --aggregate-only

# モデル学習のみ実行 (Model training only)
uv run main.py --train-only

# 最適化のみ実行（事前に学習済みモデルが必要）(Optimization only - requires pre-trained models)
uv run main.py --optimize-only

```

## 最適化アルゴリズム (Optimization Algorithm)

### 概要 (Overview)

本システムは**履歴パターンマッチング最適化**を採用し、過去の類似天候条件での最適なエアコン設定を学習して、将来の天気予報に基づいて最適な設定を決定します。
(This system adopts **historical pattern matching optimization**, learning optimal AC settings from past similar weather conditions to determine optimal settings based on future weather forecasts.)

### 前処理・集約 (Preprocessing & Aggregation)

**データ前処理 (Data Preprocessing):**
- 生データの正規化とクリーニング (Raw data normalization and cleaning)
- AC制御データと電力メーターデータの統合 (Integration of AC control and power meter data)
- 欠損値処理と異常値検出 (Missing value handling and outlier detection)

**データ集約 (Data Aggregation):**
- 制御エリア単位でのデータ集約 (Data aggregation by control area)
- 時間特徴量の追加（曜日、時刻、月、週末フラグ等）(Time feature addition - day of week, hour, month, weekend flags, etc.)
- ラグ特徴量とローリング統計の生成 (Lag features and rolling statistics generation)

### 最適化アルゴリズム (Optimization Algorithm)

#### 1. 基本概念 (Basic Concept)

**日単位履歴パターンマッチング (Day-level Historical Pattern Matching):**
- 天気予報データを日単位でグループ化し、過去の類似天候日の履歴データと比較 (Group weather forecast data by day and compare with historical data from similar weather days)
- 予報日の1日全体と類似した過去の日を検索し、その日の最適なAC設定パターンを適用 (Search for past days similar to the entire forecast day and apply optimal AC setting patterns from that day)
- 類似日選択後、その日の各時刻の設定パターン（最低電力消費パターン）を使用 (After selecting similar day, use setting patterns for each hour from that day (lowest power consumption patterns))

#### 2. 最適化フロー (Optimization Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                    最適化プロセス  (Optimization Process)                    │
└─────────────────────────────────────────────────────────────────┘

入力データ (Input Data):
├── 天気予報データ (Weather Forecast Data)
│   ├── 外気温 (Outdoor Temperature)
│   ├── 日射量 (Solar Radiation)
│   └── 日時 (Datetime)
├── 履歴データ (Historical Data)
│   ├── 過去の天候データ (Past Weather Data)
│   ├── AC設定履歴 (AC Setting History)
│   ├── 電力消費履歴 (Power Consumption History)
│   └── 室内温度履歴 (Indoor Temperature History)
└── マスターデータ (Master Data)
    ├── 営業時間設定 (Operating Hours) - オプション (Optional)
    ├── ゾーン設定 (Zone Settings)
    └── 運転区分マッピング (Operation Type Mapping) - オプション (Optional)
        └── 制御マスタシートから読み込み (Loaded from 制御マスタ sheet)

最適化モード (Optimization Modes):
├── 全日モード (Whole Day Mode) - hour_block_size=None
│   └── 完全な24時間の履歴日を選択 (Selects complete 24-hour historical days)
└── 時間ブロックモード (Hour Block Mode) - hour_block_size=N (N >= 2)
    └── 候補日から最適なN時間ブロックを選択 (Selects best N-hour blocks from candidate days)
        └── 各ブロックは異なる日から選択可能 (Each block can come from different days)

各ゾーン・各予報日に対して (For Each Zone & Forecast Day):
├── 0. 予報時間範囲フィルタリング (Forecast Hour Range Filtering) - オプション
│   └── forecast_hour_rangeが指定されている場合、指定範囲外の時間は最適化対象外
│       (If forecast_hour_range specified, hours outside range are excluded from optimization)
│
├── 1. 日単位類似度計算 (Day-level Similarity Calculation)
│   ├── 予報日の1日平均外気温・日射量を計算 (Calculate daily mean outdoor temp & solar radiation for forecast day)
│   ├── 履歴データから同一ゾーンの各日の平均外気温・日射量を計算 (Calculate daily mean values for each historical day in same zone)
│   ├── Z-score正規化による標準化 (Standardization using Z-score normalization)
│   │   ├── 履歴データ全体の平均・標準偏差を計算 (Calculate mean & std dev for all historical data)
│   │   └── 予報日と各履歴日のZ-scoreを計算 (Calculate Z-scores for forecast day and each historical day)
│   ├── 類似度スコア計算 (Similarity Score Calculation)
│   │   ├── 時刻に応じた天候重み付け (Hour-based Weather Weighting)
│   │   │   ├── 17:00-6:59: 外気温 1.0, 日射量 0.0 (Temperature 1.0, Solar 0.0)
│   │   │   └── 7:00-16:59: 外気温 0.7, 日射量 0.3 (Temperature 0.7, Solar 0.3)
│   │   ├── スコア = 重み付きZ-score差の合計 (Score = weighted sum of Z-score differences)
│   │   └── スコアが小さいほど類似度が高い (Lower score = higher similarity)
│   └── 上位20件の類似日を選択 (Select Top 20 Similar Days)
│
├── 1.5. 運転区分フィルタリング (Operation Type Filtering) - オプション
│   ├── 予報日の月から運転区分を取得 (Get operation type from forecast day's month)
│   ├── 候補日を運転区分でフィルタリング (Filter candidate days by operation type)
│   │   ├── COOL: COOL(1) と FAN(3) を許可 (Allows COOL(1) and FAN(3))
│   │   ├── HEAT: HEAT(2) と FAN(3) を許可 (Allows HEAT(2) and FAN(3))
│   │   └── FAN/OFF: 該当モードのみ許可 (Only exact mode allowed)
│   └── フィルタリング後も候補日がない場合は元の候補日を使用 (Falls back to original candidates if filtered list is empty)
│
├── 2. パターン選択 (Pattern Selection)
│   │
│   ├── 2A. 全日モード: 最適完全日の選択 (Whole Day Mode: Best Complete Day Selection)
│   │   ├── 2段階優先順位システム (Two-tier Priority System)
│   │   │   ├── 第1優先: 完全なデータがある日から最小電力の日を選択 (Priority 1: Select day with lowest power from complete days)
│   │   │   │   └── 完全な日 = 予報日の全時刻のデータが存在 (Complete day = all forecast hours available)
│   │   │   └── 第2優先: 完全な日がない場合、欠損時間が最も少ない日を選択 (Priority 2: If no complete days, select day with least missing hours)
│   │   │       └── 同じ欠損数の場合、最小電力の日を優先 (If same missing count, prioritize lowest power)
│   │   └── 選択された日のパターンを抽出 (Extract Patterns from Selected Day)
│   │       ├── 運転区分でフィルタリング (Filter by operation type if available)
│   │       ├── モード優先設定時: 目標運転モードを優先、次に最低電力 (If mode priority: prefer target mode, then lowest power)
│   │       ├── デフォルト: 各時刻ごとに最低電力消費パターンを1つ選択 (Default: Select one lowest power pattern for each hour)
│   │       └── 1日1パターン/時刻のデータセットを作成 (Create dataset with one pattern per hour)
│   │
│   └── 2B. 時間ブロックモード: 最適時間ブロックの選択 (Hour Block Mode: Best Hour Block Selection)
│       ├── 予報時間を連続するN時間ブロックにグループ化 (Group forecast hours into consecutive N-hour blocks)
│       ├── 各予報ブロックに対して (For Each Forecast Block)
│       │   ├── 候補日の同一時刻ブロックを抽出 (Extract same-hour blocks from candidate days)
│       │   │   └── 重要: 予報が[8,9,10]の場合、履歴も[8,9,10]を選択 (Critical: If forecast is [8,9,10], historical must also be [8,9,10])
│       │   ├── 運転区分でフィルタリング (Filter by operation type if available)
│       │   ├── ブロック距離計算 (Calculate Block Distance)
│       │   │   ├── 時刻に応じた天候重み付けを使用 (Use hour-based weather weighting)
│       │   │   ├── 予報ブロックと履歴ブロックの平均天候を比較 (Compare mean weather of forecast and historical blocks)
│       │   │   └── Z-score正規化による距離計算 (Distance calculation using Z-score normalization)
│       │   ├── 最適ブロック選択 (Select Best Block)
│       │   │   ├── 最低天候距離、次に最低電力 (Lowest weather distance, then lowest power)
│       │   │   └── モード優先設定時: 目標運転モードを優先 (If mode priority: prefer target operation mode)
│       │   └── 各時刻を直接マッピング (Direct Hour Mapping)
│       │       └── 予報時刻 → 同一履歴時刻 (Forecast hour → Same historical hour)
│       └── 全ブロックの結果を統合 (Combine Results from All Blocks)
│
├── 3. 時刻別パターン適用 (Hourly Pattern Application)
│   ├── 予報日の各時刻に対して (For Each Hour in Forecast Day)
│   ├── 選択されたパターンを使用 (Use Selected Pattern)
│   │   ├── 全日モード: 選択された完全日の同一時刻のパターン (Whole Day Mode: Same hour from selected complete day)
│   │   └── 時間ブロックモード: 選択されたブロックの同一時刻のパターン (Hour Block Mode: Same hour from selected block)
│   ├── AC設定を抽出 (Extract AC Settings)
│   │   ├── 設定温度 (Set Temperature)
│   │   ├── 運転モード (Operation Mode)
│   │   ├── ファン速度 (Fan Speed)
│   │   └── ON/OFF状態 (ON/OFF Status)
│   └── 履歴パターンから関連値を取得 (Extract Related Values from Historical Pattern)
│       ├── 履歴電力消費 (Historical Power Consumption)
│       │   └── 選択されたパターンのadjusted_powerを使用 (Use adjusted_power from selected pattern)
│       └── 履歴室内温度 (Historical Indoor Temperature)
│           └── 選択されたパターンのIndoor Temp.を使用 (Use Indoor Temp. from selected pattern)
│
└── 4. 結果出力 (Result Output)
    ├── 最適AC設定 (Optimal AC Settings)
    │   ├── 設定温度 (Set Temperature)
    │   ├── 運転モード (Operation Mode)
    │   ├── ファン速度 (Fan Speed)
    │   └── ON/OFF状態 (ON/OFF Status)
    ├── 履歴参照値 (Historical Reference Values)
    │   ├── 履歴室内温度 (Historical Indoor Temperature)
    │   │   └── マッチした履歴パターンから取得 (From matched historical pattern)
    │   └── 履歴電力消費 (Historical Power Consumption)
    │       └── マッチした履歴パターンから取得 (From matched historical pattern)
    └── メタデータ (Metadata)
        ├── 使用した履歴日の日付 (Historical Date Used)
        ├── 使用した履歴データの日時 (Historical Datetime Used)
        ├── 履歴日の外気温・日射量 (Historical Outdoor Temp & Solar Radiation)
        └── 予報日の外気温・日射量 (Forecast Outdoor Temp & Solar Radiation)
```

#### 4. 最適化戦略 (Optimization Strategy)

**日単位マッチング戦略 (Day-level Matching Strategy):**
- 予報日の1日全体と類似した過去の日を検索することで、日中の変動パターンも考慮 (By searching for past days similar to entire forecast day, also consider intraday variation patterns)
- 同一日のパターンを使用することで、時刻間の一貫性を保証 (Using patterns from same day ensures consistency across hours)
- 電力消費最小化を優先し、過去の実績から最適な設定を学習 (Prioritize power consumption minimization, learning optimal settings from past performance)

**完全性重視の選択 (Completeness-first Selection):**
- データの完全性を最優先し、欠損データの影響を最小化 (Prioritize data completeness to minimize impact of missing data)
- 完全なデータがない場合でも、可能な限り多くの時刻のデータを提供 (Even when complete data is unavailable, provide data for as many hours as possible)
- 段階的なフォールバックにより、常に最適な結果を提供 (Progressive fallback ensures optimal results are always provided)

**学習型最適化 (Learning-based Optimization):**
- 過去の実績データから学習し、類似条件での最適解を適用 (Learn from past performance data and apply optimal solutions for similar conditions)
- 日単位の類似性により、季節性や天候パターンを考慮 (Day-level similarity considers seasonality and weather patterns)
- 継続的な改善と適応により、システムの精度を向上 (Continuous improvement and adaptation improve system accuracy)


# GCP Settings 
## Run locally with Docker Compose
```
docker compose down && docker compose up --build
```

## Test locally via HTTP
```
curl -X POST http://localhost:8080/execute_optimization_pipeline
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "airux8-opti-logic-prod",
    "name": "00_InputData/Clea/01_PreprocessedData/features_processed_Clea.csv"
  }'
```

## Manual deploy to Cloud Run Service
Prereqs:
- gcloud CLI authenticated and project set
- Artifact Registry repository exists: `airux8-optimize-repo` in `asia-northeast1`
- Service account has required roles: `${PROJECT}.iam.gserviceaccount.com`

Authenticate and set project:
```bash
gcloud auth login
gcloud config set project airux8-opti-logic
```

Authenticate Docker to Artifact Registry:
```bash
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
```

Build and push image (Apple Silicon: target linux/amd64):
```bash
IMAGE="asia-northeast1-docker.pkg.dev/airux8-opti-logic/airux8-optimize-repo/svc-airux8-optimize:prod"
docker buildx build --platform linux/amd64 -t "$IMAGE" . --push
```

Deploy Cloud Run Service (uses FastAPI/uvicorn CMD):
```bash
gcloud run deploy svc-airux8-optimize-prod \
  --region=asia-northeast1 \
  --image="$IMAGE" \
  --service-account=svc-airux8-optimize@airux8-opti-logic.iam.gserviceaccount.com \
  --memory=2Gi \
  --cpu=1 \
  --timeout=900s \
  --max-instances=1 \
  --set-env-vars=STORAGE_BACKEND=gcs,PROJECT_ID=airux8-opti-logic,BUCKET_NAME=airux8-opti-logic-prod \
  --no-allow-unauthenticated
```

Create a trigger from GCS (if needed):
```bash
gcloud eventarc triggers create trigger-gcs-upload \
  --location=asia-northeast1 \
  --destination-run-service=svc-airux8-optimize-prod \
  --destination-run-region=asia-northeast1 \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=airux8-opti-logic-prod" \
  --service-account=svc-airux8-optimize@airux8-opti-logic.iam.gserviceaccount.com
```

Notes:
- Artifact image format: `REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG`.
- For local only (no buildx), you can `docker build -t "$IMAGE" . && docker push "$IMAGE"`, but prefer buildx to ensure linux/amd64.

Create Scheduler Job (if needed) / スケジューラージョブの作成 : 
```bash

gcloud scheduler jobs create http svc-airux8-optimize-prod \
  --schedule "0 1 * * *" \
  --uri "https://svc-airux8-optimize-prod-144706892563.asia-northeast1.run.app/execute_optimization_pipeline" \
  --http-method POST \
  --location asia-northeast1 \
  --time-zone "Asia/Tokyo" \
  --description "Run svc-airux8-optimize-prod daily" \
  --oidc-service-account-email svc-airux8-optimize@airux8-opti-logic.iam.gserviceaccount.com \
  --oidc-token-audience "https://svc-airux8-optimize-prod-144706892563.asia-northeast1.run.app/execute_optimization_pipeline" 

  --headers "Content-Type=application/json" \
  --message-body='{"bucket":"airux8-opti-logic-prod","name":"00_InputData/Clea/01_PreprocessedData/features_processed_Clea.csv"}'

```

To delete : 
```bash
gcloud scheduler jobs delete svc-airux8-optimize-prod --location asia-northeast1
```

Manual Execution / 手動実行
To manually trigger the scheduler job:
スケジューラージョブを手動で実行するには：
```bash
gcloud scheduler jobs run svc-airux8-optimize-prod --location=asia-northeast1
```
