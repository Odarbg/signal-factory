import io
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from signal_factory.data.download import RulesNotAcceptedError, download


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_download_invokes_kaggle_cli_and_unzips(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> CompletedProcess[str]:
        calls.append(cmd)
        (tmp_path / "ieee-fraud-detection.zip").write_bytes(
            make_zip(
                {
                    "train_transaction.csv": b"a",
                    "train_identity.csv": b"b",
                    "test_transaction.csv": b"c",
                    "sample_submission.csv": b"d",
                }
            )
        )
        return CompletedProcess(cmd, 0, "", "")

    download(tmp_path, runner=fake_run)

    assert calls[0][:3] == ["kaggle", "competitions", "download"]
    assert "ieee-fraud-detection" in calls[0]
    assert (tmp_path / "train_transaction.csv").read_bytes() == b"a"
    assert (tmp_path / "train_identity.csv").read_bytes() == b"b"
    assert not (tmp_path / "test_transaction.csv").exists()
    assert not (tmp_path / "ieee-fraud-detection.zip").exists()


def test_download_explains_rules_on_403(tmp_path: Path) -> None:
    def fake_run(cmd: list[str]) -> CompletedProcess[str]:
        return CompletedProcess(cmd, 1, "", "403 - Forbidden")

    with pytest.raises(RulesNotAcceptedError, match="competitions/ieee-fraud-detection/rules"):
        download(tmp_path, runner=fake_run)


def test_download_raises_on_other_failures(tmp_path: Path) -> None:
    def fake_run(cmd: list[str]) -> CompletedProcess[str]:
        return CompletedProcess(cmd, 1, "", "401 - Unauthorized")

    with pytest.raises(RuntimeError, match="401"):
        download(tmp_path, runner=fake_run)


def test_download_leaves_unrelated_files_alone(tmp_path: Path) -> None:
    keep = tmp_path / "notes.txt"
    keep.write_text("mine")

    def fake_run(cmd: list[str]) -> CompletedProcess[str]:
        (tmp_path / "ieee-fraud-detection.zip").write_bytes(
            make_zip(
                {
                    "train_transaction.csv": b"a",
                    "train_identity.csv": b"b",
                    "test_identity.csv": b"c",
                }
            )
        )
        return CompletedProcess(cmd, 0, "", "")

    download(tmp_path, runner=fake_run)
    assert keep.read_text() == "mine"
    assert not (tmp_path / "test_identity.csv").exists()
