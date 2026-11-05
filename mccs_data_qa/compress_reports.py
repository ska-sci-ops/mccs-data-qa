"""compress_reports.py -- tar.gz a set's reports/ directory, leaving out the large data files."""

import argparse
import os
import tarfile

EXCLUDE_EXT = ('.h5', '.hkl', '.cal', '.uvx')


def compress_reports(set_dir, exclude=EXCLUDE_EXT):
    """Write `{set_dir}/reports.tar.gz` holding reports/ minus files ending in `exclude`; return its path."""
    out = os.path.join(set_dir, 'reports.tar.gz')

    def skip(ti):
        return None if ti.name.endswith(tuple(exclude)) else ti

    with tarfile.open(out, 'w:gz') as tar:
        tar.add(os.path.join(set_dir, 'reports'), arcname='reports', filter=skip)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('set_dir', help='Set folder containing reports/, e.g. sweeps/set1615_2026-10-18')
    a = p.parse_args()
    out = compress_reports(a.set_dir)
    print(f'{out}: {os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
