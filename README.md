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

**履歴パターンマッチング (Historical Pattern Matching):**
- 天気予報データと過去の履歴データを比較 (Compare weather forecast data with historical data)
- 類似した天候条件（外気温、日射量）の履歴パターンを検索 (Search for historical patterns with similar weather conditions - outdoor temperature, solar radiation)
- 最適な設定パターンを学習・適用 (Learn and apply optimal setting patterns)

#### 2. 最適化フロー (Optimization Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                    最適化プロセス  (Optimization Process)                    │
└─────────────────────────────────────────────────────────────────┘

入力データ (Input Data):
├── 天気予報データ (Weather Forecast Data)
│   ├── 外気温 (Outdoor Temperature)
│   ├── 日射量 (Solar Radiation)
│   └── 湿度 (Humidity)
├── 履歴データ (Historical Data)
│   ├── 過去の天候データ (Past Weather Data)
│   ├── AC設定履歴 (AC Setting History)
│   └── 電力消費履歴 (Power Consumption History)
└── マスターデータ (Master Data)
    ├── 営業時間設定 (Operating Hours)
    ├── 快適温度範囲 (Comfort Temperature Range)
    └── ゾーン設定 (Zone Settings)

各時刻・各ゾーンに対して (For Each Time Point & Zone):
├── 1. 営業時間判定 (Operating Hours Check)
│   ├── 営業時間外 (Non-Business Hours) → OFFモード設定 (Set OFF Mode)
│   └── 営業時間内 (Business Hours) → 最適化実行 (Execute Optimization)
│
├── 2. 類似パターン検索 (Similar Pattern Search)
│   ├── 同一時刻の履歴データを抽出 (Extract Historical Data for Same Hour)
│   ├── 天候類似度計算 (Weather Similarity Calculation)
│   │   ├── 外気温差 ≤ ±0.5°C (Outdoor Temp Diff ≤ ±0.5°C)
│   │   └── Z-score正規化による類似度スコア (Z-score Normalized Similarity Score)
│   └── 上位10件の類似パターンを選択 (Select Top 10 Similar Patterns)
│
├── 3. 快適性フィルタリング (Comfort Filtering)
│   ├── 快適温度範囲内のパターンのみを保持 (Keep Only Patterns Within Comfort Range)
│   └── 季節・月別の快適温度範囲を適用 (Apply Seasonal/Monthly Comfort Range)
│
├── 4. スコアリング・選択 (Scoring & Selection)
│   ├── 電力スコア計算 (Power Score Calculation)
│   │   └── 電力消費量の正規化スコア (Normalized Power Consumption Score)
│   ├── 温度スコア計算 (Temperature Score Calculation)
│   │   └── 快適温度からの偏差スコア (Deviation from Comfort Temperature Score)
│   ├── 時間重み付け (Time-based Weighting)
│   │   ├── 朝 (Morning): 温度重視 (Temp: 80%, Power: 20%)
│   │   ├── 午後 (Afternoon): バランス (Temp: 50%, Power: 50%)
│   │   └── 夕方 (Evening): 電力重視 (Temp: 30%, Power: 70%)
│   └── 最適パターン選択 (Optimal Pattern Selection)
│       └── 最小複合スコアのパターンを選択 (Select Pattern with Minimum Combined Score)
│
└── 5. 結果出力 (Result Output)
    ├── 最適AC設定 (Optimal AC Settings)
    │   ├── 設定温度 (Set Temperature)
    │   ├── 運転モード (Operation Mode)
    │   └── ファン速度 (Fan Speed)
    ├── 予測結果 (Prediction Results)
    │   ├── 予測室温 (Predicted Indoor Temperature)
    │   └── 予測電力消費 (Predicted Power Consumption)
    └── メタデータ (Metadata)
        ├── 類似度スコア (Similarity Score)
        ├── 複合スコア (Combined Score)
        └── 使用した履歴データ (Used Historical Data)
```

#### 3. アルゴリズムの特徴 (Algorithm Features)

**天候類似度計算 (Weather Similarity Calculation):**
- 外気温の重み: 70% (Outdoor Temperature Weight: 70%)
- 日射量の重み: 30% (Solar Radiation Weight: 30%)
- Z-score正規化による標準化 (Standardization using Z-score normalization)
- 温度許容差: ±0.5°C (Temperature Tolerance: ±0.5°C)

**時間重み付けシステム (Time-based Weighting System):**
- 朝 (6:00-12:00): 快適性重視 (Comfort Priority)
- 午後 (12:00-18:00): バランス重視 (Balance Priority)
- 夕方 (18:00-24:00): 省エネ重視 (Energy Saving Priority)

**快適性制約 (Comfort Constraints):**
- 季節別快適温度範囲の適用 (Apply Seasonal Comfort Temperature Range)
- 快適範囲外のパターンは除外 (Exclude Patterns Outside Comfort Range)
- 快適性を最優先に保証 (Guarantee Comfort as Top Priority)

#### 4. 最適化戦略 (Optimization Strategy)

**多目的最適化 (Multi-objective Optimization):**
- 目的1: 電力消費最小化 (Objective 1: Minimize Power Consumption)
- 目的2: 快適性維持 (Objective 2: Maintain Comfort)
- 重み付けによるバランス調整 (Balance Adjustment through Weighting)

**学習型最適化 (Learning-based Optimization):**
- 過去の実績データから学習 (Learn from Past Performance Data)
- 類似条件での最適解を適用 (Apply Optimal Solutions for Similar Conditions)
- 継続的な改善と適応 (Continuous Improvement and Adaptation)


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
