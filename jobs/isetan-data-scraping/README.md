# Job-Isetan-Data-Scraping

This service provides a FastAPI-based web scraper for collecting A/C Power Meter and A/C制御 (A/C Control) data from the Airux8 web interface. The service uses Playwright for browser automation and integrates with Google Cloud Secret Manager for secure credential management.

## GCP Settings

### Create Service Account and Grant Permissions

Create the service account:

```bash
gcloud iam service-accounts create job-isetan-data-scraping \
  --project=airux8-opti-logic \
  --description="IsetanMitsukoshi Data Scraping Service Account" \
  --display-name="job-isetan-data-scraping"
```

Grant required IAM roles:

```bash
# BigQuery Job User
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"

# BigQuery Data Editor
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# Cloud Run Service Invoker
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

# Cloud Run Admin
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Eventarc Event Receiver
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/eventarc.eventReceiver"

# Secret Manager Secret Accessor
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Storage Admin
gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

### Run locally with Docker Compose

```bash
docker compose down && docker compose up --build
```

### Test locally

Run scraping directly:
```bash
docker compose run --rm job-isetanmitsukoshi-data-scraping uv run python main.py
```

Or if you want to run it interactively:
```bash
docker compose up
```

### Manual deploy to Cloud Run Job

Prereqs:
- gcloud CLI authenticated and project set
- Artifact Registry repository exists: `airux8-optimize-repo` in `asia-northeast1`
- Service account has required roles: `job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com`
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
IMAGE="asia-northeast1-docker.pkg.dev/airux8-opti-logic/airux8-optimize-repo/job-isetanmitsukoshi-data-scraping:prod"
docker buildx build --platform linux/amd64 -t "$IMAGE" . --push

# or this command: 
docker buildx build --builder=desktop-linux --platform linux/amd64 --no-cache --pull -t "asia-northeast1-docker.pkg.dev/airux8-opti-logic/airux8-optimize-repo/job-isetanmitsukoshi-data-scraping:prod" . --push
```

Deploy Cloud Run Job:
```bash
gcloud run jobs deploy job-isetanmitsukoshi-data-scraping-prod \
  --region=asia-northeast1 \
  --image="$IMAGE" \
  --service-account=job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com \
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
gcloud run jobs add-iam-policy-binding job-isetanmitsukoshi-data-scraping-prod \
  --region=asia-northeast1 \
  --member="serviceAccount:job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

**Step 1b: Grant Cloud Scheduler permission to use the service account**

Cloud Scheduler needs permission to impersonate the service account. Get your project number first:

```bash
PROJECT_NUMBER=$(gcloud projects describe airux8-opti-logic --format="value(projectNumber)")
```

Then grant the permission:

```bash
gcloud iam service-accounts add-iam-policy-binding job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --project=airux8-opti-logic
```

**Verify the permissions were granted:**

```bash
# Verify Service Account IAM
gcloud iam service-accounts get-iam-policy job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com
```

You should see:
```bash
gcloud iam service-accounts get-iam-policy job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com
bindings:
- members:
  - serviceAccount:service-144706892563@gcp-sa-cloudscheduler.iam.gserviceaccount.com
  role: roles/iam.serviceAccountUser
etag: BwZE8Iuwaz4=
version: 1
```

**Step 2: Create the scheduler job**

Get your project number first:
```bash
PROJECT_NUMBER=$(gcloud projects describe airux8-opti-logic --format="value(projectNumber)")
```

Create the scheduler job using OAuth authentication (recommended for Cloud Run Jobs):
```bash
gcloud scheduler jobs create http job-isetanmitsukoshi-data-scraping-prod \
  --location="asia-northeast1" \
  --schedule="30 0 * * *" \
  --time-zone="Asia/Tokyo" \
  --uri="https://asia-northeast1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_NUMBER}/jobs/job-isetanmitsukoshi-data-scraping-prod:run" \
  --http-method="POST" \
  --oauth-service-account-email="job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com" \
  --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" \
  --description="Scheduler job that triggers job-isetanmitsukoshi-data-scraping-prod daily at 00:30" \
  --attempt-deadline=900s \
  --min-backoff=60s \
  --max-backoff=60s \
  --max-retry-duration=300s
```

**Note:** This uses OAuth authentication (`--oauth-service-account-email` and `--oauth-token-scope`) instead of OIDC, which is more reliable for Cloud Run Jobs API calls.

### To delete scheduler job

```bash
gcloud scheduler jobs delete job-isetanmitsukoshi-data-scraping-prod --location asia-northeast1
```

### Manual Execution / 手動実行

To manually trigger the scheduler job:
スケジューラージョブを手動で実行するには：
```bash
gcloud scheduler jobs run job-isetanmitsukoshi-data-scraping-prod --location=asia-northeast1
```

To manually execute the Cloud Run Job:
```bash
gcloud run jobs execute job-isetanmitsukoshi-data-scraping-prod --region=asia-northeast1
```
6. **Test the scheduler again**:
```bash
gcloud scheduler jobs run job-isetanmitsukoshi-data-scraping-prod --location=asia-northeast1
```

7. **Check the logs** to verify it worked:
```bash
gcloud logging read "resource.type=cloud_scheduler_job AND resource.labels.job_id=job-isetanmitsukoshi-data-scraping-prod AND resource.labels.location=asia-northeast1" --limit=3 --format="table(timestamp,jsonPayload.status,jsonPayload.debugInfo)" --project=airux8-opti-logic
```

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


## BigQuery Tables

The scraped data is stored in BigQuery dataset `IsetanMitsukoshi` in project `airux8-opti-logic`. Below are the table schemas and commands to create them.

### AC Control Table (`ac_control`)

**Schema fields:**
- `ac_name` (STRING): A/C unit identifier (e.g., "G-21", "G-24")
- `datetime` (TIMESTAMP): Timestamp of the measurement (timezone-aware)
- `outdoor_temp` (INTEGER): Outdoor temperature in degrees Celsius
- `indoor_temp` (FLOAT64): Indoor temperature in degrees Celsius
- `ac_set_temperature` (FLOAT64): A/C set temperature in degrees Celsius
- `ac_on_off` (STRING): A/C power state ("ON" or "OFF")
- `ac_mode` (STRING): A/C operating mode (e.g., "HEAT", "COOL")
- `ac_fan_speed` (STRING): A/C fan speed setting (nullable, can be empty)
- `naive_energy_level` (INTEGER): Naive energy level (0-100 scale)
- `airux_energy_level` (INTEGER): Airux energy level (0-100 scale)
- `outdoor_room_temp` (FLOAT64): Outdoor room temperature in degrees Celsius
- `outdoor_set_temp` (FLOAT64): Outdoor set temperature in degrees Celsius
- `room_set_temp` (FLOAT64): Room set temperature in degrees Celsius

**Create Datasets**
```bash
bq --location=asia-northeast1 mk --dataset airux8-opti-logic:Clea
bq --location=asia-northeast1 mk --dataset airux8-opti-logic:IsetanMitsukoshi
```

**Create table command:**
```bash
cd schema

bq mk --table \
  --project_id=airux8-opti-logic \
  --dataset_id=IsetanMitsukoshi \
  --description="AC control data scraped from Airux8 web interface" \
  --time_partitioning_field=Datetime \
  --time_partitioning_type=DAY \
  --clustering_fields=Datetime,AC_Name \
  ac_control_raw \
  ./ac_control_schema.json

```

### AC Power Meter Table (`ac_power_meter`)

**Schema fields:**
- `mesh_id` (INTEGER): Mesh network identifier
- `pm_addr_id` (INTEGER): Power meter address identifier
- `datetime` (TIMESTAMP): Timestamp of the measurement (timezone-aware)
- `phase_a` (INTEGER): Power reading for Phase A in watts
- `phase_b` (INTEGER): Power reading for Phase B in watts
- `phase_c` (INTEGER): Power reading for Phase C in watts

**Create table command:**
```bash
cd schema 

bq mk --table \
  --project_id=airux8-opti-logic \
  --dataset_id=IsetanMitsukoshi \
  --description="AC power meter data scraped from nodes" \
  --time_partitioning_field=datetime \
  --time_partitioning_type=DAY \
  --clustering_fields=Datetime,Mesh_ID \
  ac_power_meter_raw \
  ./ac_power_meter_schema.json


```

### Verify Tables Created

```bash
# List tables in IsetanMitsukoshi dataset
bq ls --project_id=airux8-opti-logic IsetanMitsukoshi

# Describe table schemas
bq show --schema --format=prettyjson --project_id=airux8-opti-logic IsetanMitsukoshi.ac_control
bq show --schema --format=prettyjson --project_id=airux8-opti-logic IsetanMitsukoshi.ac_power_meter
```