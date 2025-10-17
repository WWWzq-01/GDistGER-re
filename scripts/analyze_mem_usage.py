#!/usr/bin/env python3
"""Parse `[MEM]` logs and summarise memory usage timelines."""

import argparse
import collections
import math
import re
from typing import Dict, Tuple


ALLOC_RE = re.compile(r"\[MEM\]\[(?P<cat>[^\]]+)\] (?P<tag>.+?) -> (?P<mb>[0-9.]+) MB")
FREE_RE = re.compile(r"\[MEM\]\[(?P<cat>[^\]]+)\] (?P<tag>.+?) freed(?: (?P<mb>[0-9.]+) MB)?")


def humanize(mb: float) -> str:
    if mb >= 1024.0:
        return f"{mb / 1024.0:.2f} GB"
    return f"{mb:.2f} MB"


def parse_log(path: str):
    current: Dict[Tuple[str, str], float] = {}
    per_cat_current: collections.defaultdict[str, float] = collections.defaultdict(float)
    per_cat_peak: Dict[str, float] = collections.defaultdict(float)
    peak_total = 0.0
    peak_event = None
    total = 0.0
    events = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            alloc = ALLOC_RE.search(line)
            if alloc:
                cat = alloc.group("cat").strip()
                tag = alloc.group("tag").strip()
                mb = float(alloc.group("mb"))
                key = (cat, tag)
                prev = current.get(key, 0.0)
                delta = mb - prev
                if math.isclose(delta, 0.0, abs_tol=1e-6):
                    events.append((idx, cat, tag, 0.0, mb))
                    continue
                current[key] = mb
                total += delta
                per_cat_current[cat] += delta
                per_cat_peak[cat] = max(per_cat_peak[cat], per_cat_current[cat])
                events.append((idx, cat, tag, delta, mb))
            else:
                free = FREE_RE.search(line)
                if not free:
                    continue
                cat = free.group("cat").strip()
                tag = free.group("tag").strip()
                key = (cat, tag)
                prev = current.pop(key, 0.0)
                if prev == 0.0:
                    delta = 0.0
                else:
                    delta = -prev
                total += delta
                per_cat_current[cat] += delta
                events.append((idx, cat, tag, delta, 0.0))
            if total > peak_total:
                peak_total = total
                peak_event = (idx, cat, tag)
            for c in list(per_cat_current.keys()):
                per_cat_peak[c] = max(per_cat_peak[c], per_cat_current[c])
    remainder = sorted(current.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "events": events,
        "remainder": remainder,
        "per_cat_current": per_cat_current,
        "per_cat_peak": per_cat_peak,
        "total_current": total,
        "peak_total": peak_total,
        "peak_event": peak_event,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyse [MEM] logs")
    parser.add_argument("log", help="Path to log file")
    parser.add_argument("--top", type=int, default=10, help="Number of entries to display")
    args = parser.parse_args()

    stats = parse_log(args.log)

    print("==== Memory Summary ====")
    print(f"Peak total: {humanize(stats['peak_total'])}")
    if stats["peak_event"]:
        idx, cat, tag = stats["peak_event"]
        print(f"  Reached at line {idx} ({cat}: {tag})")
    print(f"Current total: {humanize(stats['total_current'])}")
    print()

    print("Per-category totals (remaining):")
    for cat, val in sorted(stats["per_cat_current"].items(), key=lambda kv: kv[0]):
        print(f"  {cat:<8} {humanize(val)}")
    print()

    print("Per-category peaks:")
    for cat, val in sorted(stats["per_cat_peak"].items(), key=lambda kv: kv[0]):
        print(f"  {cat:<8} {humanize(val)}")
    print()

    top = stats["remainder"][: args.top]
    if top:
        print(f"Top {len(top)} outstanding allocations:")
        for (cat, tag), mb in top:
            print(f"  {cat:<5} {humanize(mb):>12}  {tag}")
    else:
        print("No outstanding allocations detected.")


if __name__ == "__main__":
    main()
