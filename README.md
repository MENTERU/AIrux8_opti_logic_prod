# エアコン最適化システム

## 概要

エアコンの設定温度とモードを最適化し、電力消費を最小化するシステムです。営業時間内の室温制約を考慮した期間最適化スケジュールを生成します。

## セットアップ

### 1. プライベート情報ファイルの作成

システムを使用する前に、`config/private_information.py` ファイルを作成し、以下の変数を設定してください：

```python
# config/private_information.py
# gmailのメールアドレス (Gdriveから取得する場合。現在停止中)
ACCESS_INFORMATION = "name@menteru.jp" または　ACCESS_INFORMATION = "name@gmail.com"
# visual crossing Weather API Key
WEATHER_API_KEY = "weather_api_key_here"
```

**注意**: このファイルは `.gitignore` に含まれているため、Git にコミットされません。各開発者が個別に作成する必要があります。

### 2. 必要な API キーの取得

- **Weather API Key**: Visual Crossing Weather API のキーを取得
  - サイト: https://www.visualcrossing.com/weather-api
  - 無料プランでも利用可能

### 3. データフォルダの準備

#### ローカルパスを使用する場合（推奨）

ローカルパスを使用する場合は、データフォルダをプロジェクトのルートディレクトリに配置してください。

**🗂️フォルダ構造:**

```
AIrux8_opti_logic/
├── run_optimization.py
├── config/
├── processing/
├── training/
├── optimization/
├── planning/
├── visualization/
└── data/                    # ← このフォルダをダウンロードして配置
    ├── 00_InputData/        # 生データ
    ├── 01_MasterData/       # マスターデータ
    ├── 02_PreprocessedData/ # 前処理済みデータ（自動生成）
    ├── 03_Models/           # 学習済みモデル（自動生成）
    ├── 04_PlanningData/     # 計画データ（自動生成）
    └── 05_ValidationResults/# 検証結果（自動生成）
```

**データフォルダの取得方法:**

1. Google Drive または共有ストレージから `data/` フォルダをダウンロード
2. プロジェクトのルートディレクトリ（`AIrux8_opti_logic/`）に配置
3. フォルダ構造が上記の通りになっていることを確認

## 実行方法

### 基本的な実行コマンド

```bash
# フルパイプライン実行（前処理→学習→最適化）
uv run run_optimization.py

# 特定のストアで実行
uv run run_optimization.py --store Clea

# 特定の期間で実行
uv run run_optimization.py --start-date 2024-01-01 --end-date 2024-01-02
```

### 段階別実行フラグ

```bash
# 前処理のみ実行
uv run run_optimization.py --preprocess-only

# 集約のみ実行
uv run run_optimization.py --aggregate-only

# モデル学習のみ実行
uv run run_optimization.py --train-only

# 最適化のみ実行（事前に学習済みモデルが必要）
uv run run_optimization.py --optimize-only

# 可視化をスキップ
uv run run_optimization.py --skip-visualization
```

## 最適化アルゴリズム

### 期間最適化システム（PeriodOptimizer）

本システムは**期間最適化**を採用し、各時刻で独立して最適化を実行します。

#### 1. 基本方針

- **営業時間内**: 快適温度範囲内で電力消費を最小化
- **営業時間外**: 自動的にOFFモードに設定
- **制約条件**: 温度変化制約（±1°C/時）、モード遷移制約を適用

#### 2. 最適化フロー

```
各時刻に対して:
├── 営業時間判定
│   ├── 営業時間外 → OFFモード（22°C設定）
│   └── 営業時間内 → 最適化実行
│       ├── 候補生成（温度×モード×ファン速度）
│       ├── 制約適用
│       │   ├── 温度変化制約（前時刻±1°C以内）
│       │   └── モード遷移制約
│       │       ├── HEAT → OFF/FANのみ
│       │       └── その他 → OFF/FAN/COOL
│       ├── 予測実行
│       │   ├── 室温予測（温度モデル）
│       │   └── 電力予測（電力モデル）
│       └── 最適解選択
│           ├── 快適範囲内 → 電力最小
│           └── 快適範囲外 → 快適温度に最も近い
```

#### 3. 制約条件

**温度変化制約:**
- 前時刻の設定温度から±1°C以内の変更のみ許可
- 制約違反時は最も近い温度で再計算

**モード遷移制約:**
- HEATモード → OFF/FANのみ許可
- その他のモード → OFF/FAN/COOL許可
- 制約により急激なモード変更を防止

**電力予測制約:**
- 負の電力予測値（0未満）の組み合わせは除外
- 電力予測値が0以上の組み合わせのみを最適化候補として考慮
- 非現実的な負の電力消費を防止し、実用的な最適化を実現

#### 4. 予測モデル

**温度予測:**
- XGBoost回帰モデル
- 特徴量: 設定温度、前時刻室温、A/C状態、外気温、湿度、時間特徴量等

**電力予測:**
- ログ変換XGBoost回帰モデル（非負値保証）
- 特徴量: 温度特徴量 + 室内温度
- 予測値は自動的に非負値に変換

#### 5. 最適化戦略

**快適範囲内の場合:**
- 電力消費量が最小の組み合わせを選択
- 快適性を保ちながら省エネを実現

**快適範囲外の場合:**
- 快適温度範囲に最も近い温度を予測する組み合わせを選択
- 快適性を優先しつつ、可能な限り省エネを考慮

## システム構成

### 1. データ処理プロセス (`processing/`)

#### DataPreprocessor (`preprocessor.py`)
- 生データの読み込みと前処理
- AC制御データ、電力メーターデータの正規化
- 欠損値処理とデータクリーニング

#### AreaAggregator (`aggregator.py`)
- 制御エリア単位でのデータ集約
- 時間特徴量の追加（曜日、時刻、月、週末フラグ等）
- ラグ特徴量の生成

### 2. 機械学習プロセス (`training/`)

#### ModelBuilder (`model_builder.py`)
- 制御エリア別の予測モデル学習
- 温度予測モデル（XGBoost）
- 電力予測モデル（ログ変換XGBoost）
- マルチアウトプットモデル（温度+電力同時予測）

#### DataProcessor (`data_processor.py`)
- 特徴量エンジニアリング
- ラグ特徴量、ローリング特徴量の生成
- 時間特徴量の追加

### 3. 最適化プロセス (`optimization/`)

#### PeriodOptimizer (`period_optimizer.py`)
- 期間最適化の実行
- 制約条件の適用
- 並列処理による高速化

#### OptimizationFeatureBuilder (`feature_builder.py`)
- 最適化時の特徴量構築
- 履歴データの管理
- エンジニアリング特徴量の生成

### 4. 計画・出力プロセス (`planning/`)

#### Planner (`planner.py`)
- 最適化結果のスケジュール出力
- 制御区分別・室内機別スケジュール生成
- CSV形式での出力


## 出力結果

### CSV ファイル

- `data/04_PlanningData/{Store}/control_type_schedule_YYYYMMDD.csv` - 制御区分別スケジュール
- `data/04_PlanningData/{Store}/unit_schedule_YYYYMMDD.csv` - 室内機別スケジュール

### 検証結果

- `data/05_ValidationResults/{Store}/valid_results_{Zone}.csv` - 各ゾーンの予測結果
- 温度予測、電力予測の詳細データ

### 学習済みモデル

- `data/03_Models/{Store}/models_{Zone}.pkl` - 制御エリア別学習済みモデル


## トラブルシューティング

### よくある問題

1. **`private_information.py` が見つからない**
   - `config/private_information.py` ファイルを作成してください
   - 必要な変数を設定してください

2. **モデルが見つからない**
   - `--train-only` でモデルを学習してください
   - `data/03_Models/{Store}/` にモデルファイルが存在するか確認

3. **天気データが取得できない**
   - Visual Crossing Weather API のキーが正しく設定されているか確認
   - インターネット接続を確認

4. **データファイルが見つからない**
   - `data/` フォルダがプロジェクトのルートディレクトリに配置されているか確認
   - `data/00_InputData/{Store}/` に必要な CSV ファイルが存在するか確認