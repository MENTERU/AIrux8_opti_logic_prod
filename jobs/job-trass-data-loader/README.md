# Raw Data Loading & Composition Pipeline

## Overview

This repository provides a Python-based pipeline to load, unify, and preprocess raw AC control (IDU) and power meter (ODU) telemetry CSV files per facility. The pipeline supports both **local filesystem** and **Google Cloud Storage (GCS)** sources and outputs cleaned, standardized CSV files suitable for downstream preprocessing or ingestion into a central database.

> **Note:** Raw telemetry ingestion for each facility should be performed by a Cloud Run Job or other ingestion service. This pipeline focuses on loading, datetime normalization, merging multiple files, and basic data composition.

## Key Features

* Load raw IDU/ODU CSV files from local directories or GCS buckets.
* Supports filtering by filename date ranges, including reversed date inputs.
* Normalize datetime columns to UTC without timezone information.
* Compose and standardize IDU and ODU datasets with consistent column names.
* Compute total energy (`total_kwh`) for ODU data.
* Saves processed CSV outputs to configurable paths (local or GCS).

## Repository layout (relevant files)

```
.
├─ main_pipeline.py             # Main loading, composing, and saving script
├─ config/
│  ├─ config.py                 # Constants and local paths, filename prefixes
│  └─ config_gcp.py             # GCP paths / environment variables
├─ menteru_tools/               # Internal helpers (storage, GCS interface)
└─ requirements.txt
```

## Configuration / Environment variables

* `DATA_SOURCE_TYPE`: "LOCAL" or "REMOTE" (switch between local filesystem and GCS).
* `PROJECT_ID`, `BUCKET_ID`: GCP project and bucket for remote operations.
* `INPUT_DATA_PATH`, `LOADED_DATA_PATH`: paths for raw input and output processed data.
* `START_DATE`, `END_DATE`: optional date range filter for processing files.

## How to run locally

1. Create a Python environment and install dependencies:

   ```bash
   uv sync
   ```
2. Prepare local input directories and sample CSV files under `LOCAL_INPUT_DATA_PATH`.
3. Export required environment variables:

   ```bash
   export DATA_SOURCE_TYPE=LOCAL
   export INPUT_DATA_PATH="data/00_InputData"
   export LOADED_DATA_PATH="data/06_LoadedData"
   export START_DATE="2025-01-01"
   export END_DATE="2025-01-31"
   ```
4. Run the main pipeline:

   ```bash
   python main_pipeline.py
   ```

## Data Loading & Processing Steps

1. **Scan Input Paths:**

   * Recursively list all files in local path or GCS prefix.
   * Filter files based on type (`/ac-control/` for IDU, `/ac-power-meter/` for ODU) and filename date range.

2. **Load CSVs:**

   * Merge multiple CSVs per type into a single DataFrame.
   * Handle reading errors with logging and skip faulty files.

3. **Datetime Normalization:**

   * Detect datetime-like columns automatically.
   * Convert to UTC, remove timezone, floor to nearest minute.

4. **Compose IDU Dataset:**

   * Standardize column names: `measured_at`, `idu_id`, `ac_set_temperature`, `indoor_temperature`, `ac_on_off`, `ac_mode`, `ac_fan_speed`.

5. **Compose ODU Dataset:**

   * Compute `total_kwh` from phase readings.
   * Standardize column names: `measured_at`, `odu_id`, `total_kwh`.

6. **Save Outputs:**

   * Write processed IDU and ODU CSVs to `LOADED_DATA_PATH/{store}`.
   * Use local filesystem or GCS based on `DATA_SOURCE_TYPE`.

## Input / Output Conventions

* **Input:** CSV files under `INPUT_DATA_PATH/{store}/ac-control/` and `/ac-power-meter/`.
* **Filename pattern:** `YYYY-MM-DD-YYYY-MM-DD.csv` is used for date filtering.
* **Output:** Processed CSV files:

  * `idu_loaded.csv` — cleaned IDU dataset
  * `odu_loaded.csv` — cleaned ODU dataset

## Running in GCP / Cloud Run

* The pipeline can be containerized and deployed to Cloud Run as a **job** or service.
* For remote execution, ensure `PROJECT_ID` and `BUCKET_ID` are set and `DATA_SOURCE_TYPE=REMOTE`.
* The pipeline will use `menteru_tools.gcp_service.Storage` to read/write CSVs in GCS.

## Testing & Validation

* Test local runs with sample data under `LOCAL_INPUT_DATA_PATH`.
* Validate that datetime columns are unified, IDU/ODU columns are properly renamed, and total energy is computed correctly.
* Ensure logs provide detailed information on processed files and any skipped/errored files.

## Observability

* Uses Python logging with INFO/WARNING/ERROR levels.
* Logs include file paths, store names, and summary shapes of loaded DataFrames.

## GCP Deployment

   ```bash
   gcloud iam service-accounts create job-trass-data-loader \
      --project=airux8-opti-logic \
      --description="HVAC Data Loader Service Account" \
      --display-name="job-trass-data-loader"

   gcloud projects add-iam-policy-binding airux8-opti-logic \
      --member="serviceAccount:job-trass-data-loader@airux8-opti-logic.iam.gserviceaccount.com" \
      --role="roles/storage.objectAdmin"

      ```


