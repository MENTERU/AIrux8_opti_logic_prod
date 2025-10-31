import json
import os
import pickle
from io import BytesIO
from typing import Any, List

import pandas as pd
from google.cloud import storage as gcs


class StorageClient:
    """Abstract storage interface for CSV/byte IO and object listing.

    Implementations must support reading/writing CSV and raw bytes and
    listing object paths with a prefix filter.
    """

    def read_bytes(self, path: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def write_bytes(
        self, data: bytes, path: str, content_type: str | None = None
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_csv(self, path: str) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def write_csv(self, df: pd.DataFrame, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_excel(self, path: str, **kwargs) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def write_excel(
        self, df: pd.DataFrame, path: str, **kwargs
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_json(self, path: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def write_json(
        self, obj: Any, path: str, indent: int | None = 2
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_pickle(self, path: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def write_pickle(self, obj: Any, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def list(self, prefix: str) -> List[str]:  # pragma: no cover
        raise NotImplementedError


class GCSClient(StorageClient):
    """Google Cloud Storage implementation of StorageClient.

    Uses a project and bucket to perform CSV/byte IO and list objects.
    """

    def __init__(self, project_id: str, bucket_id: str) -> None:
        self.project_id = project_id
        self.bucket_id = bucket_id
        self._client = gcs.Client(project=self.project_id)
        self._bucket = self._client.bucket(self.bucket_id)

    def _blob(self, path: str):
        return self._bucket.blob(path)

    def read_bytes(self, path: str) -> bytes:
        blob = self._blob(path)
        return blob.download_as_bytes()

    def write_bytes(
        self, data: bytes, path: str, content_type: str | None = None
    ) -> None:
        blob = self._blob(path)
        if content_type:
            blob.upload_from_string(data, content_type=content_type)
        else:
            blob.upload_from_string(data)

    def read_csv(self, path: str) -> pd.DataFrame:
        blob = self._blob(path)
        content = blob.download_as_bytes()
        return pd.read_csv(BytesIO(content), low_memory=False)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self.write_bytes(csv_bytes, path, content_type="text/csv")

    def read_excel(self, path: str, **kwargs) -> pd.DataFrame:
        content = self.read_bytes(path)
        return pd.read_excel(BytesIO(content), **kwargs)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        bio = BytesIO()
        df.to_excel(bio, index=False, **kwargs)
        self.write_bytes(
            bio.getvalue(),
            path,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def read_json(self, path: str) -> Any:
        content = self.read_bytes(path)
        return json.loads(content.decode("utf-8"))

    def write_json(self, obj: Any, path: str, indent: int | None = 2) -> None:
        payload = json.dumps(obj, ensure_ascii=False, indent=indent).encode("utf-8")
        self.write_bytes(payload, path, content_type="application/json")

    def read_pickle(self, path: str) -> Any:
        content = self.read_bytes(path)
        return pickle.loads(content)

    def write_pickle(self, obj: Any, path: str) -> None:
        payload = pickle.dumps(obj)
        self.write_bytes(payload, path, content_type="application/octet-stream")

    def list(self, prefix: str) -> List[str]:
        blobs = self._client.list_blobs(self.bucket_id, prefix=prefix)
        return [b.name for b in blobs]


class LocalStorageClient(StorageClient):
    """Filesystem-backed storage rooted at a specified directory.

    All paths are resolved relative to the provided root directory.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def _full_path(self, path: str) -> str:
        return os.path.join(self.root_dir, path)

    def read_bytes(self, path: str) -> bytes:
        full_path = self._full_path(path)
        with open(full_path, "rb") as f:
            return f.read()

    def write_bytes(
        self, data: bytes, path: str, content_type: str | None = None
    ) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data)

    def read_csv(self, path: str) -> pd.DataFrame:
        full_path = self._full_path(path)
        return pd.read_csv(full_path, low_memory=False)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False)

    # -------- Excel / JSON / Pickle helpers (local filesystem) --------
    def read_excel(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_excel(self._full_path(path), **kwargs)

    def write_excel(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_excel(full_path, index=False, **kwargs)

    def read_json(self, path: str) -> Any:
        full_path = self._full_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, obj: Any, path: str, indent: int | None = 2) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)

    def read_pickle(self, path: str) -> Any:
        full_path = self._full_path(path)
        with open(full_path, "rb") as f:
            return pickle.load(f)

    def write_pickle(self, obj: Any, path: str) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            pickle.dump(obj, f)

    def list(self, prefix: str) -> List[str]:
        base = self._full_path(prefix)
        if not os.path.exists(base):
            return []
        paths: List[str] = []
        for root, _, files in os.walk(base):
            for filename in files:
                rel = os.path.relpath(os.path.join(root, filename), self.root_dir)
                paths.append(rel)
        return paths


def get_storage_client() -> StorageClient:
    """Factory to get a storage client based on STORAGE_BACKEND.

    - When STORAGE_BACKEND == "gcs": returns GCSClient using config values
    - Otherwise: returns LocalStorageClient rooted at LOCAL_DATA_ROOT or ./data
    """
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "gcs":
        from config.config_gcp import GCPEnv

        return GCSClient(project_id=GCPEnv.PROJECT_ID, bucket_id=GCPEnv.BUCKET_NAME)
    root = os.getenv("LOCAL_DATA_ROOT", os.path.join(os.getcwd(), "data"))
    return LocalStorageClient(root_dir=root)
