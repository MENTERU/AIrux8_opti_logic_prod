# func-check-clea-files

## 概要

この関数は、Cloud Function として HTTP リクエスト(Cloud Scheduler)で呼び出され、AIrux8 Optimize プロジェクトの計画データファイルの存在を確認する。

## 機能

- **unit_schedule ファイルチェック**: 当日の unit_schedule ファイル (`unit_schedule_YYYYMMDD_*.csv`) が `4_PlanningData/Clea/` フォルダに存在するかチェック
- **zone_schedule ファイルチェック**: 当日の zone_schedule ファイル (`zone_schedule_YYYYMMDD_*.csv`) が `4_PlanningData/Clea/` フォルダに存在するかチェック
- **Slack通知**: ファイルが存在しない場合、`gcp_アラート` チャンネルにアラートを送信

## 必要な設定

### 1. GCP Secret Manager
Slack Webhook URL を GCP Secret Manager に保存：

```bash
# シークレットを作成
echo "https://hooks.slack.com/services/YOUR/WEBHOOK/URL" | \
gcloud secrets create SLACK_WEBHOOK_URL \
  --data-file=- \
  --project=airux8-opti-logic

# サービスアカウントにアクセス権限を付与
gcloud secrets add-iam-policy-binding SLACK_WEBHOOK_URL \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=airux8-opti-logic
```

### 2. サービスアカウント権限

Cloud Function が以下のリソースにアクセスできるよう、サービスアカウントに権限を付与：

- **Storage Object Viewer**: GCS バケットの読み取り
- **Secret Manager Secret Accessor**: Slack webhook URL の取得

## デプロイ方法

### Cloud Function のデプロイ

```bash
gcloud functions deploy func-check-clea-files \
  --gen2 \
  --runtime=python311 \
  --region=asia-northeast1 \
  --source=. \
  --entry-point=check_clea_files \
  --trigger-http \
  --no-allow-unauthenticated \
  --service-account=YOUR_SERVICE_ACCOUNT@airux8-opti-logic.iam.gserviceaccount.com \
  --set-env-vars BUCKET_NAME=airux8-opti-logic-prod \
  --set-secrets SLACK_WEBHOOK_URL=SLACK_WEBHOOK_URL:latest \
  --project=airux8-opti-logic
```

### Cloud Scheduler の設定

毎日自動実行するために Cloud Scheduler ジョブを作成：

```bash
gcloud scheduler jobs create http func-check-clea-files-scheduler \
  --location=asia-northeast1 \
  --schedule="0 6 * * *" \
  --time-zone="Asia/Tokyo" \
  --uri="https://asia-northeast1-airux8-opti-logic.cloudfunctions.net/func-check-clea-files" \
  --http-method=POST \
  --oidc-service-account-email="YOUR_SERVICE_ACCOUNT@airux8-opti-logic.iam.gserviceaccount.com" \
  --oidc-token-audience="https://asia-northeast1-airux8-opti-logic.cloudfunctions.net/func-check-clea-files"
```

## 期待されるファイル名

### unit_schedule ファイル
- **パス**: `4_PlanningData/Clea/unit_schedule_YYYYMMDD_*.csv`
- **例**: `4_PlanningData/Clea/unit_schedule_20251204_20251207.csv` (2025-12-04にチェック)

### zone_schedule ファイル
- **パス**: `4_PlanningData/Clea/zone_schedule_YYYYMMDD_*.csv`
- **例**: `4_PlanningData/Clea/zone_schedule_20251204_20251207.csv` (2025-12-04にチェック)

**注意**: ファイル名の最初の日付（開始日）が当日の日付と一致する必要があります。

## Slack通知の形式

### ファイルが存在しない場合
```
🚨 [AIrux8 Optimize] Missing Planning Data Files

❌ Missing unit_schedule file
Expected: unit_schedule_20251204_*.csv

❌ Missing zone_schedule file
Expected: zone_schedule_20251204_*.csv

Folder: airux8-opti-logic-prod/4_PlanningData/Clea/
Time: 2025-12-04 06:00:00 JST
```

## レスポンス形式

### 成功時のレスポンス
```json
{
  "timestamp": "2025-12-04T06:00:00.000000+09:00",
  "bucket": "airux8-opti-logic-prod",
  "folder": "4_PlanningData/Clea/",
  "target_date": "20251204",
  "checks": {
    "unit_schedule": true,
    "zone_schedule": false
  },
  "all_files_present": false,
  "alerts_sent": true,
  "missing_files": [
    "❌ Missing zone_schedule file\nExpected: zone_schedule_20251204_*.csv"
  ]
}
```

### エラー時のレスポンス
```json
{
  "error": "Error during file existence check: [エラーメッセージ]"
}
```

## ローカルでのテスト方法

### 1. 関数をローカルで起動

```bash
# 依存関係をインストール
pip install -r requirements.txt

# 環境変数を設定
export BUCKET_NAME=airux8-opti-logic-prod
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# 関数を起動
functions-framework --target=check_clea_files --debug --port=8080
```

### 2. 関数をローカルで呼び出す

```bash
curl -X POST http://localhost:8080
```

## トラブルシューティング

### よくある問題

1. **Slack通知が送信されない**
   - GCP Secret Manager の `SLACK_WEBHOOK_URL` シークレットが正しく設定されているか確認
   - サービスアカウントに Secret Manager アクセス権限があるか確認
   - Webhook URL が正しいか確認

2. **ファイル存在チェックが失敗する**
   - サービスアカウントにGCS読み取り権限があるか確認
   - バケット名が正しいか確認
   - フォルダパス `4_PlanningData/Clea/` が正しいか確認

3. **Cloud Schedulerが関数を呼び出せない**
   - サービスアカウントに適切な権限が付与されているか確認
   - 関数のURLが正しいか確認
   - `--no-allow-unauthenticated` フラグを使用している場合、OIDC認証が正しく設定されているか確認

### ログの確認

```bash
# Cloud Function のログを確認
gcloud functions logs read func-check-clea-files \
  --region=asia-northeast1 \
  --limit=50 \
  --project=airux8-opti-logic

# Cloud Scheduler のログを確認
gcloud scheduler jobs describe func-check-clea-files-scheduler \
  --location=asia-northeast1 \
  --project=airux8-opti-logic
```

### 手動テスト

```bash
# Cloud Scheduler ジョブを手動で実行
gcloud scheduler jobs run func-check-clea-files-scheduler \
  --location=asia-northeast1 \
  --project=airux8-opti-logic

# または、直接 HTTP リクエストを送信（認証が必要）
gcloud functions call func-check-clea-files \
  --region=asia-northeast1 \
  --project=airux8-opti-logic
```

---

# func-check-clea-files (English Version)

## Overview

This function is called as a Cloud Function via HTTP request (Cloud Scheduler) to check the existence of planning data files for the AIrux8 Optimize project.

## Features

- **unit_schedule File Check**: Checks if today's unit_schedule file (`unit_schedule_YYYYMMDD_*.csv`) exists in the `4_PlanningData/Clea/` folder
- **zone_schedule File Check**: Checks if today's zone_schedule file (`zone_schedule_YYYYMMDD_*.csv`) exists in the `4_PlanningData/Clea/` folder
- **Slack Notifications**: Sends alerts to the `gcp_アラート` channel if files are missing

## Required Setup

### 1. GCP Secret Manager
Store the Slack Webhook URL in GCP Secret Manager:

```bash
# Create secret
echo "https://hooks.slack.com/services/YOUR/WEBHOOK/URL" | \
gcloud secrets create SLACK_WEBHOOK_URL \
  --data-file=- \
  --project=airux8-opti-logic

# Grant access to service account
gcloud secrets add-iam-policy-binding SLACK_WEBHOOK_URL \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=airux8-opti-logic
```

### 2. Service Account Permissions

Grant the following permissions to the service account:

- **Storage Object Viewer**: Read access to GCS bucket
- **Secret Manager Secret Accessor**: Access to Slack webhook URL

## Deployment

### Deploy Cloud Function

```bash
gcloud functions deploy func-check-clea-files \
  --gen2 \
  --runtime=python311 \
  --region=asia-northeast1 \
  --source=. \
  --entry-point=check_clea_files \
  --trigger-http \
  --no-allow-unauthenticated \
  --service-account=YOUR_SERVICE_ACCOUNT@airux8-opti-logic.iam.gserviceaccount.com \
  --set-env-vars BUCKET_NAME=airux8-opti-logic-prod \
  --set-secrets SLACK_WEBHOOK_URL=SLACK_WEBHOOK_URL:latest \
  --project=airux8-opti-logic
```

### Setup Cloud Scheduler

Create a Cloud Scheduler job for daily execution:

```bash
gcloud scheduler jobs create http func-check-clea-files-scheduler \
  --location=asia-northeast1 \
  --schedule="0 6 * * *" \
  --time-zone="Asia/Tokyo" \
  --uri="https://asia-northeast1-airux8-opti-logic.cloudfunctions.net/func-check-clea-files" \
  --http-method=POST \
  --oidc-service-account-email="YOUR_SERVICE_ACCOUNT@airux8-opti-logic.iam.gserviceaccount.com" \
  --oidc-token-audience="https://asia-northeast1-airux8-opti-logic.cloudfunctions.net/func-check-clea-files"
```

## Expected File Names

### unit_schedule File
- **Path**: `4_PlanningData/Clea/unit_schedule_YYYYMMDD_*.csv`
- **Example**: `4_PlanningData/Clea/unit_schedule_20251204_20251207.csv` (checked on 2025-12-04)

### zone_schedule File
- **Path**: `4_PlanningData/Clea/zone_schedule_YYYYMMDD_*.csv`
- **Example**: `4_PlanningData/Clea/zone_schedule_20251204_20251207.csv` (checked on 2025-12-04)

**Note**: The first date (start date) in the filename must match today's date.

## Slack Notification Format

### When Files Are Missing
```
🚨 [AIrux8 Optimize] Missing Planning Data Files

❌ Missing unit_schedule file
Expected: unit_schedule_20251204_*.csv

❌ Missing zone_schedule file
Expected: zone_schedule_20251204_*.csv

Folder: airux8-opti-logic-prod/4_PlanningData/Clea/
Time: 2025-12-04 06:00:00 JST
```

## Response Format

### Success Response
```json
{
  "timestamp": "2025-12-04T06:00:00.000000+09:00",
  "bucket": "airux8-opti-logic-prod",
  "folder": "4_PlanningData/Clea/",
  "target_date": "20251204",
  "checks": {
    "unit_schedule": true,
    "zone_schedule": false
  },
  "all_files_present": false,
  "alerts_sent": true,
  "missing_files": [
    "❌ Missing zone_schedule file\nExpected: zone_schedule_20251204_*.csv"
  ]
}
```

### Error Response
```json
{
  "error": "Error during file existence check: [error message]"
}
```

## Local Testing

### 1. Run Function Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export BUCKET_NAME=airux8-opti-logic-prod
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Start the function
functions-framework --target=check_clea_files --debug --port=8080
```

### 2. Test the Function

```bash
curl -X POST http://localhost:8080
```

## Troubleshooting

### Common Issues

1. **Slack notifications not sending**
   - Verify `SLACK_WEBHOOK_URL` secret is properly configured in GCP Secret Manager
   - Check if service account has Secret Manager access permissions
   - Verify webhook URL is correct

2. **File existence check failing**
   - Verify service account has GCS read permissions
   - Check if bucket name is correct
   - Verify folder path `4_PlanningData/Clea/` is correct

3. **Cloud Scheduler unable to call function**
   - Verify service account has appropriate permissions
   - Check if function URL is correct
   - If using `--no-allow-unauthenticated`, verify OIDC authentication is properly configured

### View Logs

```bash
# View Cloud Function logs
gcloud functions logs read func-check-clea-files \
  --region=asia-northeast1 \
  --limit=50 \
  --project=airux8-opti-logic

# View Cloud Scheduler logs
gcloud scheduler jobs describe func-check-clea-files-scheduler \
  --location=asia-northeast1 \
  --project=airux8-opti-logic
```

### Manual Testing

```bash
# Manually trigger Cloud Scheduler job
gcloud scheduler jobs run func-check-clea-files-scheduler \
  --location=asia-northeast1 \
  --project=airux8-opti-logic

# Or call function directly (requires authentication)
gcloud functions call func-check-clea-files \
  --region=asia-northeast1 \
  --project=airux8-opti-logic
```
