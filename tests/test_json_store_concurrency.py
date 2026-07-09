import json
from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

import run_financial_news_pipeline as pipeline


class FakeBlob:
    """Minimal stand-in for google.cloud.storage.Blob covering exists/download/upload."""

    def __init__(self, bucket, name):
        self._bucket = bucket
        self.name = name
        self.generation = None

    def exists(self):
        record = self._bucket.objects.get(self.name)
        self.generation = record["generation"] if record else None
        return record is not None

    def download_as_text(self, encoding="utf-8"):
        record = self._bucket.objects[self.name]
        self.generation = record["generation"]
        return record["text"]

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        record = self._bucket.objects.get(self.name)
        current_generation = record["generation"] if record else 0
        if if_generation_match is not None and if_generation_match != current_generation:
            raise PreconditionFailed("generation mismatch")
        new_generation = current_generation + 1
        self._bucket.objects[self.name] = {"text": data, "generation": new_generation}
        self.generation = new_generation


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)


class FakeStorage:
    def __init__(self):
        self.bucket = FakeBucket()


def _normalize(payload):
    if not isinstance(payload, dict):
        payload = {}
    return {"updated_at": str(payload.get("updated_at", "") or ""), "documents": payload.get("documents", [])}


@pytest.fixture(autouse=True)
def _reset_module_state():
    pipeline._REMOTE_LOAD_ERRORED_BLOBS.clear()
    pipeline._BLOB_GENERATIONS.clear()
    yield
    pipeline._REMOTE_LOAD_ERRORED_BLOBS.clear()
    pipeline._BLOB_GENERATIONS.clear()


def test_save_succeeds_when_no_concurrent_writer(tmp_path):
    storage = FakeStorage()
    blob_name = "custom_documents.json"
    local_path = tmp_path / blob_name

    loaded = pipeline._load_json_store(storage, blob_name, local_path, dict, _normalize)
    assert loaded["documents"] == []
    assert pipeline._BLOB_GENERATIONS[blob_name] == 0

    loaded["documents"].append({"id": "a"})
    pipeline._save_json_store(storage, blob_name, local_path, loaded, _normalize)

    assert storage.bucket.objects[blob_name]["generation"] == 1
    assert pipeline._BLOB_GENERATIONS[blob_name] == 1


def test_save_rejects_when_another_writer_changed_blob_since_load(tmp_path):
    storage = FakeStorage()
    blob_name = "custom_documents.json"
    local_path = tmp_path / blob_name

    # Seed initial remote state and load it (captures generation baseline).
    storage.bucket.objects[blob_name] = {
        "text": json.dumps({"updated_at": "", "documents": [{"id": "seed"}]}),
        "generation": 5,
    }
    loaded = pipeline._load_json_store(storage, blob_name, local_path, dict, _normalize)
    assert pipeline._BLOB_GENERATIONS[blob_name] == 5

    # Simulate a concurrent writer updating the blob after our load.
    storage.bucket.objects[blob_name] = {
        "text": json.dumps({"updated_at": "", "documents": [{"id": "seed"}, {"id": "concurrent"}]}),
        "generation": 6,
    }

    loaded["documents"].append({"id": "ours"})
    with pytest.raises(RuntimeError, match="another writer changed this blob"):
        pipeline._save_json_store(storage, blob_name, local_path, loaded, _normalize)

    # The concurrent writer's data must survive - our save must not have landed.
    on_disk = json.loads(storage.bucket.objects[blob_name]["text"])
    assert {"id": "concurrent"} in on_disk["documents"]
    assert {"id": "ours"} not in on_disk["documents"]


def test_checkpointed_saves_chain_off_each_others_generation(tmp_path):
    storage = FakeStorage()
    blob_name = "document_enrichment_state.json"
    local_path = tmp_path / blob_name

    loaded = pipeline._load_json_store(storage, blob_name, local_path, dict, _normalize)
    loaded["documents"] = ["one"]
    pipeline._save_json_store(storage, blob_name, local_path, loaded, _normalize)

    # A second checkpoint save in the same process (no reload in between)
    # must succeed against its own prior write, not fail as a "conflict".
    loaded["documents"] = ["one", "two"]
    pipeline._save_json_store(storage, blob_name, local_path, loaded, _normalize)

    on_disk = json.loads(storage.bucket.objects[blob_name]["text"])
    assert on_disk["documents"] == ["one", "two"]


def test_save_without_prior_load_falls_back_to_unconditional_upload(tmp_path):
    storage = FakeStorage()
    blob_name = "rule_summaries.json"
    local_path = tmp_path / blob_name

    # No _load_json_store call happened this process for this blob, so there
    # is no generation baseline; the save must not be blocked.
    pipeline._save_json_store(storage, blob_name, local_path, {"documents": []}, _normalize)

    assert storage.bucket.objects[blob_name]["generation"] == 1
