import io
import json
import logging
import os
from typing import Optional, Union

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleDriveUploader:
    """
    Google Drive file uploader using service account credentials.

    Usage:
        # Initialize with service account JSON string
        uploader = GoogleDriveUploader(service_account_json=json_string, folder_id="your_folder_id")

        # Upload DataFrame as CSV
        file_id = uploader.upload_dataframe(df, "filename.csv")

        # Upload file from path
        file_id = uploader.upload_file("/path/to/file.csv")
    """

    def __init__(
        self, service_account_json: str, folder_id: str, encoding: str = "utf-8"
    ):
        """
        Initialize Google Drive uploader.

        Args:
            service_account_json: Service account JSON credentials as string
            folder_id: Google Drive folder ID where files will be uploaded
            encoding: File encoding (utf-8, cp932, etc.)
        """
        self.service_account_json = service_account_json
        self.folder_id = folder_id
        self.encoding = encoding
        self._service = None

        # Initialize the service
        self._initialize_service()

    def _initialize_service(self):
        """Initialize Google Drive service with service account credentials."""
        try:
            service_account_info = json.loads(self.service_account_json)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=["https://www.googleapis.com/auth/drive"]
            )
            self._service = build("drive", "v3", credentials=credentials)
            logger.info("Successfully initialized Google Drive service")
        except Exception as e:
            logger.error(f"Error initializing Google Drive service: {e}")
            raise

    def upload_file(
        self,
        file_input: Union[str, pd.DataFrame, bytes],
        filename: str,
        encoding: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> str:
        """
        Upload any type of file to Google Drive.

        Args:
            file_input: Can be:
                - str: Path to local file
                - pd.DataFrame: DataFrame to upload as CSV
                - bytes: Raw file content
            filename: Name for the uploaded file
            encoding: File encoding (for DataFrame/CSV files)
            mime_type: MIME type (auto-detected if not provided)

        Returns:
            file_id: Google Drive file ID
        """
        try:
            # Handle different input types
            if isinstance(file_input, str):
                # Local file path
                with open(file_input, "rb") as f:
                    content = f.read()
                if mime_type is None:
                    mime_type = self._get_mime_type(filename)

            elif isinstance(file_input, pd.DataFrame):
                # DataFrame - convert to CSV
                encoding = encoding or self.encoding
                csv_buffer = io.StringIO()
                file_input.to_csv(csv_buffer, index=False, encoding=encoding)
                content = csv_buffer.getvalue().encode(encoding)
                mime_type = "text/csv"

            elif isinstance(file_input, bytes):
                # Raw bytes
                content = file_input
                if mime_type is None:
                    mime_type = self._get_mime_type(filename)

            else:
                raise ValueError(f"Unsupported file_input type: {type(file_input)}")

            # Upload to Google Drive
            file_id = self._upload_to_drive(content, filename, mime_type)

            logger.info(f"Successfully uploaded {filename}. File ID: {file_id}")
            return file_id

        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def _upload_to_drive(self, content: bytes, filename: str, mime_type: str) -> str:
        """Internal method to upload content to Google Drive."""
        try:
            # Create media object
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type)

            # File metadata
            file_metadata = {
                "name": filename,
                "parents": [self.folder_id],
            }

            # Upload file
            file = (
                self._service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,  # Support for shared drives
                )
                .execute()
            )

            return file.get("id")

        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {e}")
            raise

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type based on file extension."""
        extension = os.path.splitext(filename)[1].lower()

        mime_types = {
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }

        return mime_types.get(extension, "application/octet-stream")

    def list_files(self, query: Optional[str] = None) -> list:
        """
        List files in the configured folder.

        Args:
            query: Optional search query

        Returns:
            List of file information
        """
        try:
            # Build query
            search_query = f"'{self.folder_id}' in parents"
            if query:
                search_query += f" and {query}"

            # Execute search
            results = (
                self._service.files()
                .list(
                    q=search_query,
                    fields="files(id, name, mimeType, createdTime, modifiedTime)",
                    supportsAllDrives=True,
                )
                .execute()
            )

            return results.get("files", [])

        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise


def upload_dataframe_to_drive(
    df: pd.DataFrame,
    filename: str,
    service_account_json: str,
    folder_id: str,
    encoding: str = "utf-8",
) -> str:
    """
    Convenience function to upload DataFrame to Google Drive.

    Args:
        df: DataFrame to upload
        filename: Name for the uploaded file
        service_account_json: Service account JSON credentials as string
        folder_id: Google Drive folder ID
        encoding: File encoding

    Returns:
        file_id: Google Drive file ID
    """
    uploader = GoogleDriveUploader(service_account_json, folder_id, encoding)
    return uploader.upload_file(df, filename, encoding)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config.utils import load_config

    # Load config
    config = load_config()

    # Load service account JSON
    service_account_file = config.get("local_paths").get("service_account_json")
    folder_id = config.get("local_paths").get("folder_id")
    encoding = config.get("local_paths").get("encoding")
    file_path = config.get("local_paths").get("file_path")

    # Read the service account JSON file
    with open(service_account_file, "r") as f:
        service_account_json = f.read()

    # Find CSV files
    files_to_upload_to_drive = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".csv"):
                files_to_upload_to_drive.append(os.path.join(root, file))

    # Upload each CSV file to Google Drive
    for file_path in files_to_upload_to_drive:
        try:
            # Read CSV file as DataFrame
            df = pd.read_csv(file_path)

            # Extract just the filename (not full path)
            filename = os.path.basename(file_path)

            # Upload DataFrame to Google Drive
            file_id = upload_dataframe_to_drive(
                df, filename, service_account_json, folder_id, encoding
            )
            print(f"Uploaded {filename} to Google Drive. File ID: {file_id}")

        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

    print("Finished uploading all CSV files to Google Drive")
