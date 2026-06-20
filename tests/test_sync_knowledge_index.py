from types import SimpleNamespace

import sync_knowledge_index as sync


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
    def __init__(self, failed=0):
        self.failed = failed
        self.calls = []

    def create_and_poll(self, *, vector_store_id, file_ids):
        self.calls.append((vector_store_id, file_ids))
        return SimpleNamespace(file_counts=SimpleNamespace(failed=self.failed))


def _client(*, failed=0, statuses=None):
    files = _Files(statuses=statuses)
    batches = _FileBatches(failed=failed)
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

    attached, failures = sync._upload_doc_batch(client, "vs-1", _targets())

    assert failures == []
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

    attached, failures = sync._upload_doc_batch(client, "vs-1", _targets())

    assert attached == {"doc-1": "file-doc-1.txt"}
    assert failures == [
        {
            "doc_id": "doc-2",
            "stage": "attach",
            "error": "failed",
        }
    ]
