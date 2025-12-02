import json
import os
import pickle
from io import BytesIO
from typing import Any, List, Optional

import pandas as pd
from google.cloud import storage as gcs
from google.oauth2 import service_account


class GCSClient:
    """Google Cloud Storage (GCS) client.

    This class is used to read and write data to GCS.
    """

    def __init__(
        self, project_id: str, bucket_id: str, credentials_path: Optional[str] = None
    ) -> None:
        self.project_id = project_id
        self.bucket_id = bucket_id

        # Initialize client with credentials if provided
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self._client = gcs.Client(project=self.project_id, credentials=credentials)
        else:
            # Use Application Default Credentials
            self._client = gcs.Client(project=self.project_id)

        self._bucket = self._client.bucket(self.bucket_id)

    def _blob(self, path: str):
        """Get a blob (file) from the bucket."""
        return self._bucket.blob(path)

    def read_bytes(self, path: str) -> bytes:
        """Read bytes from the bucket."""
        blob = self._blob(path)
        return blob.download_as_bytes()

    def write_bytes(
        self, data: bytes, path: str, content_type: str | None = None
    ) -> None:
        """Write bytes to the bucket."""
        blob = self._blob(path)
        if content_type:
            blob.upload_from_string(data, content_type=content_type)
        else:
            blob.upload_from_string(data)

    def read_csv(self, path: str) -> pd.DataFrame:
        """Read a CSV file from the bucket."""
        blob = self._blob(path)
        content = blob.download_as_bytes()
        return pd.read_csv(BytesIO(content), low_memory=False)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to the bucket as a CSV file."""
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self.write_bytes(csv_bytes, path, content_type="text/csv")

    def read_excel(self, path: str, **kwargs) -> pd.DataFrame:
        """Read an Excel file from the bucket."""
        content = self.read_bytes(path)
        return pd.read_excel(BytesIO(content), **kwargs)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """Write a DataFrame to the bucket as an Excel file."""
        bio = BytesIO()
        df.to_excel(bio, index=False, **kwargs)
        self.write_bytes(
            bio.getvalue(),
            path,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def read_json(self, path: str) -> Any:
        """Read a JSON file from the bucket."""
        content = self.read_bytes(path)
        return json.loads(content.decode("utf-8"))

    def write_json(self, obj: Any, path: str, indent: int | None = 2) -> None:
        """Write an object to the bucket as a JSON file."""
        payload = json.dumps(obj, ensure_ascii=False, indent=indent).encode("utf-8")
        self.write_bytes(payload, path, content_type="application/json")

    def read_pickle(self, path: str) -> Any:
        """Read a pickle file from the bucket."""
        content = self.read_bytes(path)
        return pickle.loads(content)

    def write_pickle(self, obj: Any, path: str) -> None:
        """Write an object to the bucket as a pickle file."""
        payload = pickle.dumps(obj)
        self.write_bytes(payload, path, content_type="application/octet-stream")

    def list(self, prefix: str) -> List[str]:
        """List all objects in the bucket with the given prefix."""
        blobs = self._client.list_blobs(self.bucket_id, prefix=prefix)
        return [b.name for b in blobs]
