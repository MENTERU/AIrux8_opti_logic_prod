# HVAC Data Loading & Composition Pipeline (IDU / ODU)

This repository contains a Python pipeline to **load, unify, and preprocess raw telemetry CSV files** for Indoor Units (IDU) and Outdoor Units / Power Meters (ODU). The pipeline supports **local filesystem** and **Google Cloud Storage (GCS)** sources, producing cleaned CSV files ready for downstream ingestion or analysis.

---

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Configuration / Environment Variables](#configuration--environment-variables)
- [How to Run Locally](#how-to-run-locally)
- [Data Loading & Processing Steps](#data-loading--processing-steps)
- [Input / Output Conventions](#input--output-conventions)
- [Running on GCP / Cloud Run](#running-on-gcp--cloud-run)
- [GCP Architecture Diagram](#gcp-architecture-diagram)
- [Observability & Logging](#observability--logging)
- [Testing & Validation](#testing--validation)

---

## Features

- Load raw IDU/ODU CSV files from **local directories** or **GCS buckets**.
- Filter files by **filename date ranges**, including handling reversed date inputs.
- Normalize datetime columns to **UTC**, removing timezone information and flooring to the nearest minute.
- Compose and standardize IDU and ODU datasets with consistent column names.
- Compute total energy (`total_kwh`) for ODU data.
- Save processed CSV outputs to configurable paths (local or GCS).
- Detailed logging with **INFO/WARNING/ERROR** levels.

---

## Repository Layout

```
.
├─ main.py             # Main pipeline script
├─ config/
│  ├─ config.py                 # Local paths, constants, filename prefixes
│  └─ config_gcp.py             # GCP paths / environment variables
└─ requirements.txt             # Python dependencies
```

---

## Configuration / Environment Variables

| Variable | Description |
|----------|-------------|
| `DATA_SOURCE_TYPE` | `"LOCAL"` or `"REMOTE"` to switch between local filesystem and GCS |
| `LOCAL_INPUT_DATA_PATH` | Root path for local input CSV files |
| `LOCAL_LOADED_DATA_PATH` | Root path for saving processed local CSVs |
| `GCPEnv.INPUT_DATA_PATH` | Bucket prefix for raw input data on GCS |
| `GCPEnv.LOADED_DATA_PATH` | Bucket prefix for processed data on GCS |
| `GCPEnv.PROJECT_ID` | GCP project ID for remote storage |
| `GCPEnv.BUCKET_ID` | GCS bucket ID for remote storage |
| `GCPEnv.START_DATE` | Optional start date filter (YYYY-MM-DD HH:MM:SS) |
| `GCPEnv.END_DATE` | Optional end date filter (YYYY-MM-DD HH:MM:SS) |

---

## How to Run Locally

1. Create a Python environment and install dependencies:

```bash
uv sync
```

To dump uv packages into requirements.txt

```bash
uv pip compile pyproject.toml --output-file requirements.txt --no-deps
```

2. Prepare local input directories with sample CSV files under `LOCAL_INPUT_DATA_PATH`.

3. Export required environment variables:

```bash
export DATA_SOURCE_TYPE=LOCAL
export LOCAL_INPUT_DATA_PATH="data/00_InputData"
export LOCAL_LOADED_DATA_PATH="data/06_LoadedData"
export GCPEnv.START_DATE="2025-01-01 00:00:00"
export GCPEnv.END_DATE="2025-01-31 23:59:59"
```

4. Run the main pipeline:

```bash
uv run --env-file .env main.py 
```
or
```bash
docker compose up --build
```

---

## Data Loading & Processing Steps

1. **Scan Input Paths**  
   - Recursively list all files in local path or GCS prefix.  
   - Filter files by type (`/ac-control/` for IDU, `/ac-power-meter/` for ODU) and filename date range (`YYYY-MM-DD-YYYY-MM-DD.csv`).

2. **Load CSVs**  
   - Merge multiple CSVs per type into a single DataFrame.  
   - Handle errors with logging; skip faulty files.

3. **Datetime Normalization**  
   - Detect datetime-like columns automatically.  
   - Convert to UTC, remove timezone, and floor to nearest minute.

4. **Compose IDU Dataset**  
   - Standardize columns: `measured_at`, `idu_id`, `ac_set_temperature`, `indoor_temperature`, `ac_on_off`, `ac_mode`, `ac_fan_speed`.  
   - Convert `"OFF"` → `0`, `"ON"` → `1`.

5. **Compose ODU Dataset**  
   - Compute `total_kwh` from phase readings.  
   - Standardize columns: `measured_at`, `odu_id`, `total_kwh`.

6. **Save Outputs**  
   - Write processed IDU/ODU CSVs to `LOADED_DATA_PATH/{store}`.  
   - Supports local filesystem or GCS depending on `DATA_SOURCE_TYPE`.

---

## Input / Output Conventions

- **Input:** CSV files located under `INPUT_DATA_PATH/{store}/ac-control/` and `/ac-power-meter/`.  
- **Filename pattern:** `YYYY-MM-DD-YYYY-MM-DD.csv` for date filtering.  
- **Output:** Processed CSV files per store:
  - `idu_loaded.csv` — cleaned IDU dataset
  - `odu_loaded.csv` — cleaned ODU dataset

---

## Running on GCP / Cloud Run

- The pipeline can be **containerized** and deployed to Cloud Run as a job or service.  
- For remote execution, set `DATA_SOURCE_TYPE=REMOTE`, `PROJECT_ID`, and `BUCKET_ID`.  
- The pipeline uses `menteru_tools.gcp_service.Storage` to read/write CSVs in GCS.

### Example: Create Service Account and Grant Permissions

```bash
gcloud iam service-accounts create job-trass-data-loader     --project=airux8-opti-logic     --description="HVAC Data Loader Service Account"     --display-name="job-trass-data-loader"

gcloud projects add-iam-policy-binding airux8-opti-logic     --member="serviceAccount:job-trass-data-loader@airux8-opti-logic.iam.gserviceaccount.com"     --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding airux8-opti-logic \
  --member="serviceAccount:job-trass-data-loader@airux8-opti-logic.iam.gserviceaccount.com" \
  --role="roles/bigquery.admin"

```

---

## GCP Architecture Diagram

```
+--------------------+      +--------------------+
| Cloud Run Job      | ---> | GCS Bucket (Raw)   |
| (pipeline)         |      | CSVs per store     |
+--------------------+      +--------------------+
        |                              |
        v                              v
+--------------------+      +--------------------+
| Processed CSVs     | ---> | GCS Bucket (Loaded)|
| IDU / ODU          |      | per store          |
+--------------------+      +--------------------+
```

---

## Observability & Logging

- Logs at **INFO/WARNING/ERROR** levels.  
- Logs include file paths, store names, and summary shapes of loaded DataFrames.  
- Helps debug missing or faulty files.

---

## Testing & Validation

- Use sample data in `LOCAL_INPUT_DATA_PATH` for local testing.  
- Validate that:
  - Datetime columns are unified and floored to minutes.
  - IDU/ODU columns are properly renamed.
  - `total_kwh` is correctly computed.
- Check logs for file scanning, filtering, and load summaries.
