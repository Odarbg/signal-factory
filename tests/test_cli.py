from pathlib import Path

import pytest

from signal_factory import cli


def test_cli_row_a_invokes_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake(split_dir: Path, out_dir: Path) -> None:
        calls.append((split_dir, out_dir))

    monkeypatch.setattr(cli, "run_row_a", fake)
    cli.main(["row-a", str(tmp_path / "split"), str(tmp_path / "out")])
    assert calls == [(tmp_path / "split", tmp_path / "out")]


def test_cli_convert_and_split_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(cli, "convert", lambda raw, out: seen.append(f"convert {raw} {out}"))
    monkeypatch.setattr(cli, "write_split", lambda src, out: seen.append(f"split {src} {out}"))
    cli.main(["convert", str(tmp_path / "raw"), str(tmp_path / "all.parquet")])
    cli.main(["split", str(tmp_path / "all.parquet"), str(tmp_path / "split")])
    assert seen == [
        f"convert {tmp_path / 'raw'} {tmp_path / 'all.parquet'}",
        f"split {tmp_path / 'all.parquet'} {tmp_path / 'split'}",
    ]


def test_cli_stats_prints_collision_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "uid": pd.Series(["a", "a", "b"], dtype=pd.StringDtype("pyarrow")),
            "card2": [1, 2, 3],
            "card3": [1, 1, 1],
            "card4": ["v", "v", "v"],
            "card5": [1, 1, 1],
            "card6": ["d", "d", "d"],
        }
    )
    path = tmp_path / "all.parquet"
    df.to_parquet(path, index=False)
    cli.main(["stats", str(path)])
    out = capsys.readouterr().out
    assert "uids=2" in out and "colliding=1" in out and "rate=0.5" in out


def test_cli_without_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
