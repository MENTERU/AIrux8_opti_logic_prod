import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import pytz
from google.api_core import exceptions as api_exceptions
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError, NotFound

# JST timezone for converting timestamps
JST = pytz.timezone("Asia/Tokyo")

logger = logging.getLogger(__name__)

# Example nested schema (can be customized)
DEFAULT_SCHEMA = [
    bigquery.SchemaField("measured_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("out_temp", "FLOAT"),
    bigquery.SchemaField("out_hum", "FLOAT"),
    bigquery.SchemaField(
        "created_at",
        "TIMESTAMP",
        mode="NULLABLE",
        default_value_expression="CURRENT_TIMESTAMP()",
    ),
]


class BigQuery:
    """
    Unified BigQuery wrapper for querying, reading, writing, and upserting tables.

    Dataset and table can be passed dynamically per method call — no need for separate objects.

    Parameters
    ----------
    project_id : str
        Google Cloud project ID.
    client : Optional[bigquery.Client], default=None
        Pre-initialized BigQuery client. If None, a new one is created.
    """

    def __init__(self, project_id: str, client: Optional[bigquery.Client] = None):
        self.project_id = project_id
        self.client = client or bigquery.Client(project=project_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _table_id(self, dataset_id: str, table_name: str) -> str:
        if not dataset_id or not table_name:
            raise ValueError("Both `dataset_id` and `table_name` must be provided.")
        return f"{self.project_id}.{dataset_id}.{table_name}"

    def create_dataset_if_not_exists(
        self, dataset_id: str, location: str = "asia-northeast1"
    ):
        """Create a dataset if it doesn't exist."""
        dataset_ref = self.client.dataset(dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            self.client.create_dataset(dataset)
            logger.info(f"Created dataset '{dataset_id}' in location {location}")

    def create_table_if_not_exists(
        self,
        dataset_id: str,
        table_id: str,
        schema=DEFAULT_SCHEMA,
        partition_field: str = None,
        time_partitioning_type: str = "DAY",  # DAY, HOUR, MONTH
        clustering_fields: list[str] = None,
        table_description: str = None,
    ):
        """
        Create a BigQuery table if it doesn't exist, with optional partitioning and clustering.
        """
        full_table_id = self._table_id(dataset_id, table_id)

        try:
            self.client.get_table(full_table_id)
            logger.info(f"Table '{full_table_id}' already exists.")
            return
        except NotFound:
            table_ref = bigquery.Table(full_table_id, schema=schema)

            # Partitioning
            if partition_field:
                partition_type_enum = getattr(
                    bigquery.TimePartitioningType, time_partitioning_type.upper(), None
                )
                if partition_type_enum is None:
                    raise ValueError(
                        f"Invalid time_partitioning_type '{time_partitioning_type}'. Must be DAY, HOUR, MONTH."
                    )
                table_ref.time_partitioning = bigquery.TimePartitioning(
                    type_=partition_type_enum, field=partition_field
                )

            # Clustering
            if clustering_fields:
                table_ref.clustering_fields = clustering_fields

            # Description
            if table_description:
                table_ref.description = table_description

            self.client.create_table(table_ref)
            logger.info(
                f"Created table '{full_table_id}'"
                f"{' partitioned on ' + partition_field if partition_field else ''}"
                f"{' with clustering on ' + ', '.join(clustering_fields) if clustering_fields else ''}."
            )

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------
    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        param_types: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string with @parameter_name placeholders.
        params : Optional[Dict[str, Any]]
            Dictionary of parameter values.
        param_types : Optional[Dict[str, str]]
            Dictionary mapping parameter names to BigQuery types (e.g., "TIMESTAMP", "STRING").
            If not provided, defaults to "STRING" for all parameters.

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame.
        """
        try:
            job_config = bigquery.QueryJobConfig()
            if params:
                if param_types:
                    # Use provided parameter types
                    job_config.query_parameters = [
                        bigquery.ScalarQueryParameter(
                            k, param_types.get(k, "STRING"), v
                        )
                        for k, v in params.items()
                    ]
                else:
                    # Default to STRING type if param_types not provided
                    job_config.query_parameters = [
                        bigquery.ScalarQueryParameter(k, "STRING", v)
                        for k, v in params.items()
                    ]

            query_job = self.client.query(sql, job_config=job_config)
            result = query_job.result().to_dataframe()
            logger.debug("Executed query successfully: %s", sql[:100])
            return result

        except GoogleCloudError as e:
            logger.error("BigQuery query failed: %s", e, exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Read table
    # -------------------------------------------------------------------------
    def read_table(self, dataset_id: str, table_name: str) -> pd.DataFrame:
        """
        Load a BigQuery table into a pandas DataFrame.
        """
        try:
            table_id = self._table_id(dataset_id, table_name)
            df = self.client.list_rows(table_id).to_dataframe()
            logger.debug("Loaded table '%s' successfully.", table_id)
            return df

        except NotFound:
            logger.error("Table '%s' not found.", table_id)
            raise
        except GoogleCloudError as e:
            logger.error("Error reading table '%s': %s", table_id, e, exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Write DataFrame
    # -------------------------------------------------------------------------
    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        dataset_id: str,
        if_exists: str = "replace",
        schema: Optional[List[bigquery.SchemaField]] = None,
        fallback_to_streaming: bool = True,
    ) -> None:
        """
        Write a pandas DataFrame to a BigQuery table.
        """
        if if_exists not in {"replace", "append", "fail"}:
            raise ValueError("if_exists must be one of: 'replace', 'append', 'fail'")

        write_disposition = {
            "replace": bigquery.WriteDisposition.WRITE_TRUNCATE,
            "append": bigquery.WriteDisposition.WRITE_APPEND,
            "fail": bigquery.WriteDisposition.WRITE_EMPTY,
        }[if_exists]

        table_id = self._table_id(dataset_id, table_name)

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition, schema=schema
        )

        try:
            load_job = self.client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            load_job.result()
            logger.debug(
                "DataFrame written to '%s' successfully (%d rows).", table_id, len(df)
            )

        except api_exceptions.Forbidden as e:
            # When the service account lacks bigquery.jobs.create, fall back to
            # streaming inserts so ingestion can proceed with dataset/table-level roles.
            if fallback_to_streaming and "bigquery.jobs.create" in str(e):
                logger.warning(
                    "Falling back to streaming inserts for '%s' because the account "
                    "lacks bigquery.jobs.create: %s",
                    table_id,
                    e,
                )
                self._streaming_insert(df, dataset_id, table_name)
                return
            logger.error(
                "Failed to write DataFrame to '%s' due to permission error: %s",
                table_id,
                e,
                exc_info=True,
            )
            raise
        except GoogleCloudError as e:
            logger.error(
                "Failed to write DataFrame to '%s': %s", table_id, e, exc_info=True
            )
            raise

    def _streaming_insert(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        table_name: str,
        chunk_size: int = 500,
    ) -> None:
        """
        Insert rows via tabledata.insertAll to avoid load jobs (no jobs.create needed).
        """
        table_id = self._table_id(dataset_id, table_name)
        sanitized_df = df.where(pd.notnull(df), None)
        rows = sanitized_df.to_dict(orient="records")

        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            errors = self.client.insert_rows_json(table_id, chunk)
            if errors:
                logger.error(
                    "Streaming insert errors for '%s' (chunk starting at %d): %s",
                    table_id,
                    start,
                    errors,
                )
                raise RuntimeError(f"Streaming insert failed for {table_id}: {errors}")

        logger.debug(
            "DataFrame streamed to '%s' successfully (%d rows).",
            table_id,
            len(rows),
        )

    # -------------------------------------------------------------------------
    # Delete table
    # -------------------------------------------------------------------------
    def delete_table(
        self, dataset_id: str, table_name: str, not_found_ok: bool = True
    ) -> None:
        """
        Delete a BigQuery table.
        """
        table_id = self._table_id(dataset_id, table_name)
        try:
            self.client.delete_table(table_id, not_found_ok=not_found_ok)
            logger.debug("Deleted table '%s'.", table_id)

        except GoogleCloudError as e:
            logger.error("Failed to delete table '%s': %s", table_id, e, exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # List tables
    # -------------------------------------------------------------------------
    def list_tables(self, dataset_id: str) -> List[str]:
        """
        List all table names in a dataset.
        """
        try:
            tables = self.client.list_tables(dataset_id)
            table_names = [t.table_id for t in tables]
            logger.debug(
                "Found %d tables in dataset '%s'.", len(table_names), dataset_id
            )
            return table_names

        except NotFound:
            logger.error("Dataset '%s' not found.", dataset_id)
            raise
        except GoogleCloudError as e:
            logger.error(
                "Error listing tables in dataset '%s': %s", dataset_id, e, exc_info=True
            )
            raise

    # -------------------------------------------------------------------------
    # Existence check
    # -------------------------------------------------------------------------
    def table_exists(self, dataset_id: str, table_name: str) -> bool:
        """
        Check if a table exists in BigQuery.
        """
        try:
            table_id = self._table_id(dataset_id, table_name)
            self.client.get_table(table_id)
            return True
        except NotFound:
            return False
        except GoogleCloudError as e:
            logger.error("Error checking table '%s': %s", table_id, e, exc_info=True)
            raise

    def upsert_dataframe_by_keys(
        self,
        df: pd.DataFrame,
        table_name: str,
        dataset_id: str,
        unique_keys: List[str],
        schema: Optional[List[bigquery.SchemaField]] = None,
    ) -> None:
        """
        Replace matching rows and insert new ones using server-side delete+insert.

        Steps:
        1. Load the incoming DataFrame into a temporary table in the same dataset.
        2. Delete any target rows whose unique keys appear in the temp table (with
           partition pruning for efficiency).
        3. Insert all rows from the temp table into the target table.
        4. Drop the temporary table.

        This avoids client-side row-by-row operations and keeps the operation atomic at
        the table level. Uses partition pruning when Datetime column exists to optimize
        DELETE performance on partitioned tables.
        """
        if df.empty:
            return

        missing_keys = [key for key in unique_keys if key not in df.columns]
        if missing_keys:
            raise ValueError(
                f"Unique key columns not found in DataFrame: {missing_keys}"
            )

        target_table_id = self._table_id(dataset_id, table_name)
        temp_table_name = f"_tmp_{table_name}_{uuid4().hex}"
        temp_table_id = self._table_id(dataset_id, temp_table_name)

        # Use target table schema so types align (prevents TIMESTAMP/DATETIME mismatch)
        if schema is None:
            target_table = self.client.get_table(target_table_id)
            schema = target_table.schema

        # 1) Load DataFrame into a temporary table
        load_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=schema,
        )
        load_job = self.client.load_table_from_dataframe(
            df, temp_table_id, job_config=load_config
        )
        load_job.result()

        try:
            # 2) Delete matching rows from target
            # Only check the same day partition(s) since we're adding daily data
            # This makes the DELETE much more efficient on partitioned tables
            datetime_col = None
            for col in df.columns:
                if col.lower() in ["datetime", "measured_at", "timestamp"]:
                    datetime_col = col
                    break

            # Build WHERE clause with date partition filter for efficient pruning
            where_conditions = []
            if datetime_col and datetime_col in df.columns:
                # Get unique dates from DataFrame to filter by partition
                datetime_series = df[datetime_col]
                if not datetime_series.empty and pd.notna(datetime_series).any():
                    # Convert to datetime if needed and extract unique dates
                    if not pd.api.types.is_datetime64_any_dtype(datetime_series):
                        datetime_series = pd.to_datetime(datetime_series)

                    # Extract unique dates (day-level) for partition pruning
                    unique_dates = datetime_series.dt.date.unique()

                    if len(unique_dates) == 1:
                        # Single day: use DATE() function for partition pruning
                        date_str = unique_dates[0].strftime("%Y-%m-%d")
                        where_conditions.append(
                            f"DATE(`{datetime_col}`) = DATE('{date_str}')"
                        )
                    else:
                        # Multiple days: use IN clause with DATE() function
                        date_list = ", ".join(
                            [f"DATE('{d.strftime('%Y-%m-%d')}')" for d in unique_dates]
                        )
                        where_conditions.append(
                            f"DATE(`{datetime_col}`) IN ({date_list})"
                        )

            # Build join conditions for matching unique keys
            # Use EXISTS with explicit equality conditions to avoid LEFT SEMI JOIN issues
            key_condition = None
            if len(unique_keys) == 1:
                # Single key: simple IN clause
                key = unique_keys[0]
                key_condition = (
                    f"TARGET.`{key}` IN (SELECT `{key}` FROM `{temp_table_id}`)"
                )
            else:
                # Multiple keys: use EXISTS with explicit join conditions
                # Build equality conditions for each key matching target table to source
                join_conditions = " AND ".join(
                    [f"TARGET.`{key}` = SOURCE.`{key}`" for key in unique_keys]
                )
                key_condition = f"""EXISTS (
                    SELECT 1 FROM `{temp_table_id}` AS SOURCE
                    WHERE {join_conditions}
                )"""

            # Combine date filter with key condition
            if key_condition:
                where_conditions.append(key_condition)
            where_clause = " AND ".join(where_conditions)

            delete_sql = f"""
            DELETE FROM `{target_table_id}` AS TARGET
            WHERE {where_clause}
            """
            self.query(delete_sql)

            # 3) Insert all rows from temp into target
            columns = [f"`{col}`" for col in df.columns]
            column_list = ", ".join(columns)
            insert_sql = f"""
            INSERT INTO `{target_table_id}` ({column_list})
            SELECT {column_list} FROM `{temp_table_id}`
            """
            self.query(insert_sql)
        finally:
            # 4) Drop temp table regardless of success/failure to avoid table bloat
            try:
                self.client.delete_table(temp_table_id, not_found_ok=True)
            except GoogleCloudError as cleanup_error:
                logger.warning(
                    "Failed to drop temp table '%s': %s", temp_table_id, cleanup_error
                )
