import logging
import pickle
from io import BytesIO

import pandas as pd
from google.cloud import storage as gcs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCSClient:
    def __init__(self, project_id, bucket_id):
        """GCSのラッパークラス
        Arguments:
            project_id {str} -- GoogleCloudPlatform Project ID
            bucket_id {str} -- GoogleCloudStorage Bucket ID
        """
        self._project_id = project_id
        self._bucket_id = bucket_id
        self._client = gcs.Client(project_id)
        self._bucket = self._client.get_bucket(self._bucket_id)

    def show_bucket_names(self):
        """バケット名の一覧を表示"""
        [bucket.name for bucket in self._client.list_buckets()]

    def show_file_names(self, folder_path=""):
        """指定されたフォルダ内の直接のファイル一覧を表示

        Args:
            folder_path (str): バケット内のフォルダのパス
        """
        blobs = self._client.list_blobs(self._bucket, prefix=folder_path, delimiter="/")
        file_names = [blob.name for blob in blobs]
        return file_names

    def upload_file(self, local_path, gcs_path):
        """GCSにローカルファイルをアップロード

        Arguments:
            local_path {str} -- local file path
            gcs_path {str} -- gcs file path
        """
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def download_csvfile_as_dataframe(self, gcs_csv_path):
        """GCSのファイルをpd.DataFrameとしてダウンロード

        Arguments:
            gcs_csv_path {str} -- gcs file path (only csv file)

        Returns:
            [pd.DataFrame] -- csv data as pd.DataFrame
        """
        blob = self._bucket.blob(gcs_csv_path)
        content = blob.download_as_string()
        df = pd.read_csv(BytesIO(content))
        return df

    def upload_file_as_dataframe(self, df, gcs_path, flg_index=False, flg_header=True):
        """GCSにpd.DataFrameをCSVとしてアップロード

        Arguments:
            df {pd.DataFrame} -- DataFrame for upload
            gcs_path {str} -- gcs file path

        Keyword Arguments:
            flg_index {bool} -- DataFrame index flg (default: {False})
            flg_header {bool} -- DataFrame header flg (default: {True})
        """
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_string(
            df.to_csv(index=flg_index, header=flg_header, sep=",", encoding="utf-8-sig")
        )

    def upload_file_as_excel(self, df, gcs_path):
        """GCSにpd.DataFrameをExcelとしてアップロード

        Arguments:
            df {pd.DataFrame} -- DataFrame for upload
            gcs_path {str} -- gcs file path
        """
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_string(df.to_excel(index=False, header=True, sep=",", encoding="utf-8-sig"))
        print(f"Successfully uploaded {gcs_path}")
        return True
    except Exception as e:
        print(f"Error uploading {gcs_path}: {e}")
        return False
    finally:
        print(f"Successfully uploaded {gcs_path}")
        return True

    def upload_pickle_file(self, pickle_file, gcs_path):
        """GCSにファイルをPickleとしてアップロード

        Arguments:
            pickle_file {object} -- object for upload
            gcs_path {str} -- gcs pickle file path
        """
        pickle_file_str = pickle.dumps(pickle_file)
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_string(pickle_file_str)
