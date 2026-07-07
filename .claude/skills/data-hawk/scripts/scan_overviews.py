#!/usr/bin/env python3
"""
scan_overviews.py — filesystem scan for the data-hawk skill.

Finds all DATA_OVERVIEW.md files under data/, compares their mtimes against
the embedded timestamp in data/DATA_HAWK_INDEX.md, and reports directories
that have data files but no overview (coverage gaps).

Usage:
    python3 .claude/skills/data-hawk/scripts/scan_overviews.py [project_root]
    Default project_root: current working directory

Output: JSON to stdout.
"""
import sys
import json
import re
import datetime
from pathlib import Path

DATA_EXTENSIONS = {'.csv', '.parquet', '.xlsx', '.xls', '.zip', '.rar', '.json', '.xml', '.dta'}
SKIP_NAMES = {'DATA_OVERVIEW.md', 'DATA_HAWK_INDEX.md', 'README.md', '.DS_Store', 'mexico_states.geojson'}

# Reads only the first 512 bytes of the index to find the embedded timestamp
GENERATED_RE = re.compile(
    r'<!--\s*data-hawk-index:\s*generated=([0-9T:.+Z-]+)', re.IGNORECASE
)

INDEX_RELPATH = 'data/DATA_HAWK_INDEX.md'


def parse_index_timestamp(index_path: Path) -> str | None:
    try:
        header = index_path.read_bytes()[:512].decode('utf-8', errors='replace')
        m = GENERATED_RE.search(header)
        return m.group(1) if m else None
    except OSError:
        return None


def to_posix(iso_str: str) -> float | None:
    try:
        return datetime.datetime.fromisoformat(
            iso_str.replace('Z', '+00:00')
        ).timestamp()
    except (ValueError, AttributeError):
        return None


def sample_data_files(directory: Path, limit: int = 3) -> list[str]:
    found = []
    try:
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.name not in SKIP_NAMES and f.suffix.lower() in DATA_EXTENSIONS:
                found.append(f.name)
                if len(found) >= limit:
                    break
    except PermissionError:
        pass
    return found


def main():
    project_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    data_root = project_root / 'data'
    index_file = project_root / INDEX_RELPATH

    if not data_root.is_dir():
        sys.exit(f"ERROR: data directory not found at {data_root}")

    index_exists = index_file.exists()
    index_mtime = index_file.stat().st_mtime if index_exists else None
    index_generated = parse_index_timestamp(index_file) if index_exists else None
    index_ts = to_posix(index_generated) if index_generated else None

    overviews = []
    overview_dirs: set[Path] = set()

    for path in sorted(data_root.rglob('DATA_OVERVIEW.md')):
        st = path.stat()
        overview_dirs.add(path.parent)
        rel = str(path.relative_to(project_root))
        dataset_dir = str(path.parent.relative_to(data_root))
        is_new = (index_ts is None) or (st.st_mtime > index_ts)
        overviews.append({
            'path': rel,
            'dataset_dir': dataset_dir,
            'mtime': st.st_mtime,
            'mtime_iso': datetime.datetime.fromtimestamp(
                st.st_mtime, tz=datetime.timezone.utc
            ).isoformat(),
            'size_bytes': st.st_size,
            'is_new_or_changed': is_new,
        })

    # Find directories with data files but no DATA_OVERVIEW.md and no ancestor overview
    def has_ancestor_overview(d: Path) -> bool:
        parent = d.parent
        while parent != data_root and parent != project_root:
            if parent in overview_dirs:
                return True
            parent = parent.parent
        return False

    # Collect candidate gap directories (no overview, no ancestor overview)
    candidate_gaps: list[Path] = []
    for dirpath in sorted(data_root.rglob('*')):
        if not dirpath.is_dir() or dirpath in overview_dirs or dirpath == data_root:
            continue
        if has_ancestor_overview(dirpath):
            continue
        if sample_data_files(dirpath):
            candidate_gaps.append(dirpath)

    # Keep only the highest-level gap in each subtree (drop descendants of other gaps)
    candidate_set = set(candidate_gaps)
    gaps = []
    for dirpath in candidate_gaps:
        parent = dirpath.parent
        dominated = False
        while parent != data_root and parent != project_root:
            if parent in candidate_set:
                dominated = True
                break
            parent = parent.parent
        if not dominated:
            gaps.append({
                'path': str(dirpath.relative_to(project_root)),
                'sample_files': sample_data_files(dirpath),
            })

    print(json.dumps({
        'scan_time': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        'project_root': str(project_root),
        'index_path': INDEX_RELPATH,
        'index_exists': index_exists,
        'index_mtime': index_mtime,
        'index_generated': index_generated,
        'overview_count': len(overviews),
        'new_or_changed_count': sum(1 for o in overviews if o['is_new_or_changed']),
        'gap_count': len(gaps),
        'overviews': overviews,
        'gaps': gaps,
    }, indent=2))


if __name__ == '__main__':
    main()
