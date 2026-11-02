"""Download the IEEE-CIS Fraud Detection train split through the Kaggle CLI.

The competition rules forbid redistribution, so the owner accepts them under their own
Kaggle account and this script only wraps the CLI. Nothing here is ever committed.
"""

from __future__ import annotations

import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
TRAIN_FILES = ("train_transaction.csv", "train_identity.csv")

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class RulesNotAcceptedError(RuntimeError):
    """The Kaggle CLI returned 403: the competition rules are not accepted."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def download(dest: Path, runner: Runner = _run) -> None:
    """Download the archive, extract only the two train CSVs into ``dest``, delete the archive."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(dest)]
    result = runner(cmd)
    if result.returncode != 0:
        output = f"{result.stdout}\n{result.stderr}".strip()
        if "403" in output:
            raise RulesNotAcceptedError(
                "Kaggle refused the download (403). Accept the competition rules under "
                f"your own account at {RULES_URL} and run the download again.\n{output}"
            )
        raise RuntimeError(f"Kaggle CLI failed with exit code {result.returncode}:\n{output}")

    archive = dest / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if Path(name).name in TRAIN_FILES:
                zf.extract(name, dest)
    archive.unlink()  # only the archive and the train files were ever written here
