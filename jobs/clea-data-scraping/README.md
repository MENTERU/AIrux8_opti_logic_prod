# Job-Clea-Data-Scraping

This service provides a FastAPI-based web scraper for collecting A/C Power Meter and A/C制御 (A/C Control) data from the Airux8 web interface. The service uses Playwright for browser automation and integrates with Google Cloud Secret Manager for secure credential management.

## GCP Settings

### Run locally with Docker Compose

```bash
docker compose down && docker compose up --build
```

### Test locally

Run scraping directly:
```bash
docker compose run --rm job-clea-data-scraping uv run python main.py
```

Or if you want to run it interactively:
```bash
docker compose up
```

### Manual deploy to Cloud Run Job

Prereqs:
- gcloud CLI authenticated and project set
- Artifact Registry repository exists: `airux8-optimize-repo` in `asia-northeast1`
- Service account has required roles: `job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com`
- Secret Manager secret `AIRUX8_WEB_LOGIN_INFO` exists with `username` and `password` fields

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
IMAGE="asia-northeast1-docker.pkg.dev/airux8-opti-logic/airux8-optimize-repo/job-clea-data-scraping:prod"
docker buildx build --platform linux/amd64 -t "$IMAGE" . --push

# or this command: 
docker buildx build --builder=desktop-linux --platform linux/amd64 --no-cache --pull -t "asia-northeast1-docker.pkg.dev/airux8-opti-logic/airux8-optimize-repo/job-clea-data-scraping:prod" . --push
```

Deploy Cloud Run Job:
```bash
gcloud run jobs deploy job-clea-data-scraping-prod \
  --region=asia-northeast1 \
  --image="$IMAGE" \
  --service-account=job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com \
  --memory=2Gi \
  --cpu=1 \
  --task-timeout=900s \
  --max-retries=1 \
  --set-env-vars=PROJECT_ID=airux8-opti-logic,BUCKET_NAME=airux8-opti-logic-prod
```

Notes:
- Artifact image format: `REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG`.
- For local only (no buildx), you can `docker build -t "$IMAGE" . && docker push "$IMAGE"`, but prefer buildx to ensure linux/amd64.
- The job requires Playwright browsers to be installed, which is handled in the Dockerfile.
- Cloud Run Jobs execute the container's CMD (`main.py`) and exit when complete (unlike Services which stay running).
- For local development, `docker-compose.yaml` can override the CMD if you want to run a web server, but by default it runs the scraping job directly.

### Create Scheduler Job (if needed) / スケジューラージョブの作成

**CRITICAL:** Before creating the scheduler, you MUST grant the service account permission to invoke the job. This is required for authentication.

**Step 1: Grant IAM permission to invoke the Cloud Run Job**

```bash
gcloud run jobs add-iam-policy-binding job-clea-data-scraping-prod \
  --region=asia-northeast1 \
  --member="serviceAccount:job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

**Step 1b: Grant Cloud Scheduler permission to use the service account**

Cloud Scheduler needs permission to impersonate the service account. Get your project number first:

```bash
PROJECT_NUMBER=$(gcloud projects describe airux8-opti-logic --format="value(projectNumber)")
```

Then grant the permission:

```bash
gcloud iam service-accounts add-iam-policy-binding job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --project=airux8-opti-logic
```

**Verify the permissions were granted:**

```bash
# Verify Cloud Run Job IAM
gcloud run jobs get-iam-policy job-clea-data-scraping-prod --region=asia-northeast1

# Verify Service Account IAM
gcloud iam service-accounts get-iam-policy job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com
```

You should see:
- `job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com` with `roles/run.invoker` on the Cloud Run Job
- `service-{PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com` with `roles/iam.serviceAccountUser` on the service account

**Step 2: Create the scheduler job**

Get your project number first:
```bash
PROJECT_NUMBER=$(gcloud projects describe airux8-opti-logic --format="value(projectNumber)")
```

Create the scheduler job using OAuth authentication (recommended for Cloud Run Jobs):
```bash
gcloud scheduler jobs create http job-clea-data-scraping-prod \
  --location="asia-northeast1" \
  --schedule="30 0 * * *" \
  --time-zone="Asia/Tokyo" \
  --uri="https://asia-northeast1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_NUMBER}/jobs/job-clea-data-scraping-prod:run" \
  --http-method="POST" \
  --oauth-service-account-email="job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" \
  --description="Scheduler job that triggers job-clea-data-scraping-prod daily at 00:30" \
  --attempt-deadline=900s \
  --min-backoff=60s \
  --max-backoff=60s \
  --max-retry-duration=300s
```

**Note:** This uses OAuth authentication (`--oauth-service-account-email` and `--oauth-token-scope`) instead of OIDC, which is more reliable for Cloud Run Jobs API calls.

### To delete scheduler job

```bash
gcloud scheduler jobs delete job-clea-data-scraping-prod --location asia-northeast1
```

### Manual Execution / 手動実行

To manually trigger the scheduler job:
スケジューラージョブを手動で実行するには：
```bash
gcloud scheduler jobs run job-clea-data-scraping-prod --location=asia-northeast1
```

To manually execute the Cloud Run Job:
```bash
gcloud run jobs execute job-clea-data-scraping-prod --region=asia-northeast1
```

### Troubleshooting / トラブルシューティング

#### Error: `UNAUTHENTICATED` (401) when scheduler runs

If you see an error like:
```
status: "UNAUTHENTICATED"
debugInfo: "URL_ERROR-ERROR_AUTHENTICATION. Original HTTP response code number = 401"
```

This usually means one of two things:
- The service account doesn't have permission to invoke the Cloud Run Job, OR
- Cloud Scheduler doesn't have permission to impersonate the service account

Fix it by:

1. **Grant the IAM permission to invoke the Cloud Run Job** (if not already done):
```bash
gcloud run jobs add-iam-policy-binding job-clea-data-scraping-prod \
  --region=asia-northeast1 \
  --member="serviceAccount:job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

2. **Grant Cloud Scheduler permission to use the service account** (this is often the missing step):
```bash
PROJECT_NUMBER=$(gcloud projects describe airux8-opti-logic --format="value(projectNumber)")
gcloud iam service-accounts add-iam-policy-binding job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --project=airux8-opti-logic
```

3. **Verify both permissions exist**:
```bash
# Verify Cloud Run Job IAM
gcloud run jobs get-iam-policy job-clea-data-scraping-prod --region=asia-northeast1

# Verify Service Account IAM
gcloud iam service-accounts get-iam-policy job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com
```

4. **If using OIDC, update the scheduler job** to refresh its configuration. However, **OAuth is recommended** for Cloud Run Jobs:
```bash
# OAuth approach (recommended)
PROJECT_NUMBER=$(gcloud projects describe airux8-opti-logic --format="value(projectNumber)")
gcloud scheduler jobs update http job-clea-data-scraping-prod \
  --location=asia-northeast1 \
  --oauth-service-account-email=job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com \
  --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
```

5. **Wait 2-5 minutes** for IAM changes to propagate across all Google Cloud services.

6. **Test the scheduler again**:
```bash
gcloud scheduler jobs run job-clea-data-scraping-prod --location=asia-northeast1
```

7. **Check the logs** to verify it worked:
```bash
gcloud logging read "resource.type=cloud_scheduler_job AND resource.labels.job_id=job-clea-data-scraping-prod AND resource.labels.location=asia-northeast1" --limit=3 --format="table(timestamp,jsonPayload.status,jsonPayload.debugInfo)" --project=airux8-opti-logic
```

**Note:** If you redeploy the Cloud Run Job, the IAM bindings should persist, but you may need to wait a few minutes for everything to sync. If errors persist after waiting, try steps 1-4 again.

## Local Development

When running locally with Docker Compose, the job runs the scraping directly (same as Cloud Run Jobs). The `docker-compose.yaml` file can be modified to override the CMD if you need to run a web server for testing.

**Note**: Cloud Run Jobs execute the container's CMD directly and do not expose HTTP endpoints. The job runs the scraping logic and exits when complete.

## Configuration

The scraping parameters (store name, date range, data types) are currently hardcoded in `main.py`. To modify:
- Edit the `store_name`, `start_date`, `end_date`, and `data_types` variables in the `main()` function

## Dependencies

- Playwright: Browser automation for web scraping
- Google Cloud Secret Manager: Secure credential storage
- Google Cloud Storage: Data storage backend
