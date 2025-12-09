import os
from datetime import datetime
from typing import Dict, List

import functions_framework
import pytz
import requests
from google.cloud import storage

# Environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "airux8-opti-logic-prod")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
MONITORED_FOLDERS = ["4_PlanningData/Clea/"]

# Timezone for consistent date handling
TOKYO_TZ = pytz.timezone("Asia/Tokyo")


def send_slack_alert(message: str, webhook_url: str) -> bool:
    """
    Send a Slack alert using webhook URL.

    Args:
        message: The message to send
        webhook_url: Slack webhook URL

    Returns:
        bool: True if successful, False otherwise
    """
    if not webhook_url:
        print("Slack webhook URL not configured")
        return False

    try:
        payload = {
            "text": message,
            "username": "AIrux8 File Checker",
            "icon_emoji": ":warning:",
        }

        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        print("Slack alert sent successfully")
        return True

    except Exception as e:
        print(f"Failed to send Slack alert: {e}")
        return False


def list_files_in_folder(bucket_name: str, folder_name: str) -> List[str]:
    """
    List all files inside the monitored folder.

    Args:
        bucket_name: GCS bucket name
        folder_name: Folder path

    Returns:
        List of file names
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=folder_name)
        file_list = [blob.name for blob in blobs if blob.name.endswith('.csv')]

        print(f"Found {len(file_list)} CSV files in {folder_name}")
        return file_list

    except Exception as e:
        print(f"Error listing files in {folder_name}: {e}")
        return []


def check_required_files(file_list: List[str], target_date: str) -> Dict[str, bool]:
    """
    Check if required schedule files exist for the target date.

    Args:
        file_list: List of file paths in the folder
        target_date: Date string in YYYYMMDD format (today's date)

    Returns:
        Dict with file types as keys and existence status as values
    """
    required_files = {
        "unit_schedule": False,
        "zone_schedule": False
    }

    for file_path in file_list:
        file_name = file_path.split('/')[-1]  # Get filename from path

        # Check for unit_schedule_YYYYMMDD_*.csv
        if file_name.startswith(f"unit_schedule_{target_date}_"):
            required_files["unit_schedule"] = True
            print(f"✅ Found unit_schedule file: {file_name}")

        # Check for zone_schedule_YYYYMMDD_*.csv
        elif file_name.startswith(f"zone_schedule_{target_date}_"):
            required_files["zone_schedule"] = True
            print(f"✅ Found zone_schedule file: {file_name}")

    return required_files


@functions_framework.http
def check_clea_files(request):
    """
    Cloud Function entry point.
    Checks for required schedule files in monitored folders and sends Slack alerts if missing.
    """
    try:
        print("[AIrux8 Optimize] Starting file existence check")

        # Get current date in Tokyo timezone
        now_tokyo = datetime.now(TOKYO_TZ)
        today_str = now_tokyo.strftime("%Y%m%d")
        current_time_str = now_tokyo.strftime("%Y-%m-%d %H:%M:%S JST")

        print(f"Checking for files with date: {today_str}")
        print(f"Folders: {MONITORED_FOLDERS}")
        print(f"Bucket: {BUCKET_NAME}")

        # Track results across all folders
        all_missing_files = []
        all_folder_results = {}

        # Loop through each monitored folder
        for folder in MONITORED_FOLDERS:
            print(f"\n--- Checking folder: {folder} ---")
            
            # List all files in the folder
            all_files = list_files_in_folder(BUCKET_NAME, folder)

            if not all_files:
                error_msg = f"⚠️ Could not list files in {folder}"
                print(error_msg)
                all_missing_files.append(f"❌ **Folder: {folder}**\n{error_msg}")
                all_folder_results[folder] = {"error": "Could not list files"}
                continue

            # Check for required files
            file_status = check_required_files(all_files, today_str)

            # Track missing files for this folder
            folder_missing = []

            if not file_status["unit_schedule"]:
                folder_missing.append(f"❌ **Missing unit_schedule file**\nExpected: `unit_schedule_{today_str}_*.csv`")
                print(f"❌ Missing: unit_schedule_{today_str}_*.csv")

            if not file_status["zone_schedule"]:
                folder_missing.append(f"❌ **Missing zone_schedule file**\nExpected: `zone_schedule_{today_str}_*.csv`")
                print(f"❌ Missing: zone_schedule_{today_str}_*.csv")

            # Add folder context to missing files
            if folder_missing:
                all_missing_files.append(f"📁 **Folder: {folder}**\n" + "\n".join(folder_missing))

            # Store folder results
            all_folder_results[folder] = {
                "unit_schedule": file_status["unit_schedule"],
                "zone_schedule": file_status["zone_schedule"],
                "all_present": all(file_status.values())
            }

        # Send Slack alert if any files are missing
        if all_missing_files and SLACK_WEBHOOK_URL:
            alert_header = f"🚨 **[AIrux8 Optimize] Missing Planning Data Files**\n\n"
            alert_body = "\n\n".join(all_missing_files)
            alert_footer = f"\n\nBucket: `{BUCKET_NAME}`\nTime: {current_time_str}"
            
            combined_message = alert_header + alert_body + alert_footer

            alert_sent = send_slack_alert(combined_message, SLACK_WEBHOOK_URL)

            if alert_sent:
                print("✅ Slack alert sent successfully")
            else:
                print("❌ Failed to send Slack alert")

        # Prepare response
        response_data = {
            "timestamp": now_tokyo.isoformat(),
            "bucket": BUCKET_NAME,
            "folders": MONITORED_FOLDERS,
            "target_date": today_str,
            "folder_results": all_folder_results,
            "all_files_present": all(result.get("all_present", False) for result in all_folder_results.values()),
            "alerts_sent": len(all_missing_files) > 0,
        }

        if all_missing_files:
            response_data["missing_files"] = all_missing_files
            print("⚠️ File check completed with missing files")
        else:
            print("✅ All required files are present in all folders")

        return response_data, 200

    except Exception as e:
        error_message = f"Error during file existence check: {str(e)}"
        print(f"❌ {error_message}")

        # Send error alert to Slack if webhook is configured
        if SLACK_WEBHOOK_URL:
            now_tokyo = datetime.now(TOKYO_TZ)
            current_time_str = now_tokyo.strftime("%Y-%m-%d %H:%M:%S JST")
            error_alert = f"🚨 **[AIrux8 Optimize] File Check Error**\n\n{error_message}\n\nTime: {current_time_str}"
            send_slack_alert(error_alert, SLACK_WEBHOOK_URL)

        return {"error": error_message}, 500
