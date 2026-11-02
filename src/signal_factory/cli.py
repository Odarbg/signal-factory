"""``signal-factory`` command line: convert, split, stats, row-a."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from signal_factory.ablation.row_a import run_row_a
from signal_factory.data.convert import convert
from signal_factory.data.split import write_split
from signal_factory.data.stats import uid_collision_rate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal-factory", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="raw CSV directory → one Parquet with uid, day, card1_addr1")
    p.add_argument("raw", type=Path)
    p.add_argument("out", type=Path)

    p = sub.add_parser("split", help="all-rows Parquet → dev/validation/final/history files")
    p.add_argument("all", type=Path)
    p.add_argument("out_dir", type=Path)

    p = sub.add_parser("stats", help="print the uid collision statistic")
    p.add_argument("all", type=Path)

    p = sub.add_parser("row-a", help="row A on validation; never reads final.parquet")
    p.add_argument("split_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "convert":
        convert(args.raw, args.out)
    elif args.command == "split":
        counts = write_split(args.all, args.out_dir)
        print(counts)
    elif args.command == "stats":
        import pandas as pd

        stats = uid_collision_rate(pd.read_parquet(args.all))
        print(
            f"uid collisions: uids={stats.uids} colliding={stats.colliding} rate={stats.rate:.4g}"
        )
    elif args.command == "row-a":
        result = run_row_a(args.split_dir, args.out_dir)
        print(result)


if __name__ == "__main__":
    main()
