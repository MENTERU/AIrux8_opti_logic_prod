#!/usr/bin/env python3
"""
Secret Manager Service
Centralized service for managing GCP Secret Manager operations.
"""

import json
import os
from typing import Dict, Optional

from google.cloud import secretmanager

from config.config_gcp import GCPEnv


class SecretManagerClient:
    """Centralized service for GCP Secret Manager operations"""

    def __init__(self, project_id: str = None):
        self.project_id = project_id or GCPEnv.PROJECT_ID
        self.client = secretmanager.SecretManagerServiceClient()

    def get_secret_as_str(self, secret_name: str) -> Optional[str]:
        """
        Retrieve a secret from GCP Secret Manager and return as string

        Args:
            secret_name: Name of the secret to retrieve

        Returns:
            String containing the secret data, or None if failed
        """
        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"

            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})

            # Return the secret payload as string
            return response.payload.data.decode("UTF-8")

        except Exception as e:
            print(f"❌ Error accessing secret '{secret_name}': {e}")
            return None

    def get_secret_as_dict(self, secret_name: str) -> Optional[Dict]:
        """
        Retrieve a secret from GCP Secret Manager and return as dictionary

        Args:
            secret_name: Name of the secret to retrieve

        Returns:
            Dictionary containing the secret data, or None if failed
        """
        try:
            # Get secret as string first
            secret_data = self.get_secret_as_str(secret_name)
            if secret_data is None:
                return None

            # Parse as JSON
            return json.loads(secret_data)

        except Exception as e:
            print(f"❌ Error parsing secret '{secret_name}' as JSON: {e}")
            return None

    def get_store_config(
        self, secret_name: str = "STORES_MASTER_CONFIG"
    ) -> Optional[Dict]:
        """
        Get store configuration from GCP Secret Manager

        Args:
            secret_name: Name of the secret containing store configuration

        Returns:
            Store configuration dictionary, or None if failed
        """
        print(f"Loading {secret_name} from GCP Secret Manager...")
        print(f"Project ID: {self.project_id}")
        print(f"Secret Name: {secret_name}")

        config = self.get_secret_as_dict(secret_name)

        if config:
            store_count = len(config.keys())
            print(
                f"✅ Successfully loaded configuration for {store_count} stores from GCP Secret"
            )
            return config
        else:
            print("❌ Failed to load configuration from GCP Secret Manager")
            return None


# Global instance for easy access
secret_manager_service = SecretManagerService()
