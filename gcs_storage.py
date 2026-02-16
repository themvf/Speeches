#!/usr/bin/env python3
"""
Google Cloud Storage module for SEC Speeches dataset.
Handles reading/writing the speeches JSON dataset to a GCS bucket.
"""

import json
from google.cloud import storage
from google.oauth2 import service_account


BLOB_NAME = "all_speeches.json"


class GCSStorage:
    """Read/write the speeches dataset to Google Cloud Storage."""

    def __init__(self, bucket_name: str, credentials_info: dict):
        creds = service_account.Credentials.from_service_account_info(credentials_info)
        client = storage.Client(credentials=creds, project=credentials_info.get("project_id"))
        self.bucket = client.bucket(bucket_name)

    def load_speeches(self) -> dict:
        """Download the speeches dataset from GCS."""
        blob = self.bucket.blob(BLOB_NAME)
        if not blob.exists():
            return {
                "extraction_summary": {
                    "total_speeches_attempted": 0,
                    "successful_extractions": 0,
                    "failed_extractions": 0,
                    "extraction_date": "",
                },
                "speeches": [],
            }
        data = blob.download_as_text(encoding="utf-8")
        return json.loads(data)

    def save_speeches(self, data: dict):
        """Upload the speeches dataset to GCS."""
        blob = self.bucket.blob(BLOB_NAME)
        blob.upload_from_string(
            json.dumps(data, indent=2, ensure_ascii=False),
            content_type="application/json",
        )

    def get_existing_urls(self) -> set:
        """Return the set of speech URLs already in the dataset."""
        data = self.load_speeches()
        return {
            s.get("metadata", {}).get("url", "")
            for s in data.get("speeches", [])
        }
