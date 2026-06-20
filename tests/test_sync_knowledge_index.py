from types import SimpleNamespace

import pytest

import sync_knowledge_index as sync


@pytest.fixture(autouse=True)
def _disable_poll_delay(monkeypatch):
    monkeypatch.setattr(sync, "BATCH_POLL_INTERVAL_SECONDS", 0)


class _Files:
    def __init__(self, statuses=None):
        self.created = []
        self.statuses = statuses or {}

    def create(self, *, file, purpose):
        filename = file[0]
        self.created.append((filename, purpose))
        return SimpleNamespace(id=f"file-{filename}")

    def retrieve(self, file_id, *, vector_store_id):
        return SimpleNamespace(status=self.statuses.get(file_id, "completed"), last_error=None)


class _FileBatches:
    def __init__(self, failed=0, pending=None):
        self.failed = failed
        self.pending = pending or []
        self.calls = []

    def create(self, *, vector_store_id, file_ids):
        self.calls.append((vector_store_id, file_ids))
        status = "in_progress" if self.pending else "completed"
        return SimpleNamespace(id="batch-1", status=status, file_counts=SimpleNamespace(failed=self.failed))

    def retrieve(self, batch_id, *, vector_store_id):
        return SimpleNamespace(id=batch_id, status="completed", file_counts=SimpleNamespace(failed=self.failed))

    def list_files(self, batch_id, *, vector_store_id, filter, limit):
        if filter == "failed":
            ids = ["file-doc-2.txt"] if self.failed else []
        elif filter == "in_progress":
            ids = self.pending
        else:
            ids = []
        return _Page(ids)


class _Page:
    def __init__(self, file_ids):
        self.data = [SimpleNamespace(id=file_id) for file_id in file_ids]

    def has_next_page(self):
        return False


def _client(*, failed=0, pending=None, statuses=None):
    files = _Files(statuses=statuses)
    batches = _FileBatches(failed=failed, pending=pending)
    return SimpleNamespace(
        files=files,
        vector_stores=SimpleNamespace(files=files, file_batches=batches),
    )


def _targets():
    return [
        ("doc-1", {"filename": "doc-1.txt", "_rendered": "one"}),
        ("doc-2", {"filename": "doc-2.txt", "_rendered": "two"}),
    ]


def test_upload_doc_batch_attaches_uploaded_files_once():
    client = _client()

    attached, failures, pending = sync._upload_doc_batch(client, "vs-1", _targets())

    assert failures == []
    assert pending == set()
    assert attached == {
        "doc-1": "file-doc-1.txt",
        "doc-2": "file-doc-2.txt",
    }
    assert len(client.vector_stores.file_batches.calls) == 1
    vector_store_id, file_ids = client.vector_stores.file_batches.calls[0]
    assert vector_store_id == "vs-1"
    assert set(file_ids) == {"file-doc-1.txt", "file-doc-2.txt"}
    assert sorted(client.files.created) == [
        ("doc-1.txt", "assistants"),
        ("doc-2.txt", "assistants"),
    ]


def test_upload_doc_batch_reports_only_failed_attachments():
    client = _client(
        failed=1,
        statuses={"file-doc-2.txt": "failed"},
    )

    attached, failures, pending = sync._upload_doc_batch(client, "vs-1", _targets())

    assert attached == {"doc-1": "file-doc-1.txt"}
    assert pending == set()
    assert failures == [
        {
            "doc_id": "doc-2",
            "stage": "attach",
            "error": "OpenAI vector ingestion failed",
        }
    ]


def test_upload_doc_batch_persists_pending_files_after_poll_timeout(monkeypatch):
    client = _client(pending=["file-doc-2.txt"])
    monkeypatch.setattr(sync, "BATCH_POLL_TIMEOUT_SECONDS", 0)

    attached, failures, pending = sync._upload_doc_batch(client, "vs-1", _targets())

    assert failures == []
    assert attached == {
        "doc-1": "file-doc-1.txt",
        "doc-2": "file-doc-2.txt",
    }
    assert pending == {"doc-2"}


def test_reconcile_pending_docs_removes_failed_and_clears_completed():
    client = _client(statuses={"file-failed": "failed"})
    indexed_docs = {
        "completed": {"file_id": "file-completed", "index_status": "pending"},
        "failed": {"file_id": "file-failed", "index_status": "pending"},
    }

    sync._reconcile_pending_docs(client, "vs-1", indexed_docs)

    assert indexed_docs == {"completed": {"file_id": "file-completed"}}
