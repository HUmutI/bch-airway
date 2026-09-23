#!/usr/bin/env python
"""Stage 8 - verify existing outputs are readable, delete any that are not.

WHY THIS EXISTS. Every stage decides "already done?" by testing whether its
output file exists. That is only safe if a file's existence implies it is
complete. Stages 1, 3, 4 and 5 guarantee that by writing to a temp name and
renaming (os.replace is atomic within a filesystem). AirMorph does NOT - it
writes airway_bin.nii.gz through its own save_itk, which this pipeline cannot
wrap - so killing it mid-write leaves a truncated file that every later stage
would happily accept and the fuse step would choke on days later.

HOW, AND WHY NOT THE OBVIOUS WAY. Decompressing each mask to look at it is far
more expensive than its file size suggests: these are mostly-zero volumes at
full CT resolution, so a 0.3 MB airway_bin.nii.gz inflates to 215 MB and takes
~0.9 s to read. Across the cohort that is ~50 GB of decompression and 15+
minutes - paid on every restart.

Instead this exploits gzip's trailer. A gzip stream ends with a 4-byte ISIZE
field holding the uncompressed length (mod 2^32), and a NIfTI header states
exactly how many bytes the file should contain: vox_offset + voxels x bitpix/8.
Reading the 348-byte header plus the last 4 bytes of the file is O(1), and the
two numbers agree only if every byte of pixel data made it to disk. A write cut
off midway leaves either an unreadable header, a short file, or a trailer that
is really pixel data - all of which mismatch. That is exactly the failure this
needs to catch, at ~1 ms per file instead of ~900 ms.

Anything that fails is deleted so the owning stage recomputes it.

Usage (airmorph venv, no GPU):
    $OR3_ENVS/airmorph/bin/python 08_check_integrity.py --cases $OR3_CASES
    $OR3_ENVS/airmorph/bin/python 08_check_integrity.py --cases ... --dry-run
"""
import argparse
import gzip
import os
import struct
import sys

CHECKED = ["image.nii.gz",
           "airway_bin.nii.gz", "lunglobe.nii.gz", "airway_skeleton.nii.gz",
           "medpseg_airway.nii.gz", "tfenet_airway.nii.gz",
           "union3_lcc.nii.gz", "union3_lcc_srcorient.nii.gz"]


def bad(path):
    """Return a reason string if the file looks incomplete, else None."""
    try:
        size = os.path.getsize(path)
        if size < 400:                      # smaller than a bare NIfTI header
            return f"file is only {size} bytes"

        with gzip.open(path, "rb") as fh:
            hdr = fh.read(348)
        if len(hdr) < 348:
            return "header truncated"

        # Byte order: sizeof_hdr reads 348 natively, or 1543569408 byte-swapped.
        endian = "<"
        if struct.unpack("<i", hdr[0:4])[0] != 348:
            if struct.unpack(">i", hdr[0:4])[0] == 348:
                endian = ">"
            else:
                return "bad sizeof_hdr (not a NIfTI-1 header)"

        dim = struct.unpack(endian + "8h", hdr[40:56])
        bitpix = struct.unpack(endian + "h", hdr[72:74])[0]
        vox_offset = struct.unpack(endian + "f", hdr[108:112])[0]
        if not 1 <= dim[0] <= 7 or bitpix <= 0:
            return f"implausible header (ndim={dim[0]}, bitpix={bitpix})"

        voxels = 1
        for i in range(1, dim[0] + 1):
            voxels *= dim[i]
        expected = int(vox_offset) + voxels * bitpix // 8

        # gzip trailer: CRC32 then ISIZE, each little-endian regardless of the
        # NIfTI payload's byte order.
        with open(path, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            isize = struct.unpack("<I", fh.read(4))[0]

        if isize != expected % (1 << 32):
            return (f"incomplete: header expects {expected} bytes, "
                    f"gzip trailer says {isize}")
        return None
    except Exception as exc:
        return str(exc).strip().splitlines()[-1][:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="report damaged files without deleting them")
    args = ap.parse_args()

    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    checked = removed = 0
    stale = 0

    for i, d in enumerate(dirs, 1):
        # Leftover temp files from an interrupted write are never valid output.
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if f.startswith(".partial."):
                stale += 1
                if not args.dry_run:
                    os.remove(os.path.join(d, f))

        for fname in CHECKED:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            checked += 1
            why = bad(p)
            if why:
                removed += 1
                print(f"  DAMAGED {os.path.basename(d.rstrip('/'))}/{fname}: {why}",
                      flush=True)
                if not args.dry_run:
                    os.remove(p)
        if i % 100 == 0:
            print(f"  checked {i}/{len(dirs)} cases", flush=True)

    verb = "would delete" if args.dry_run else "deleted"
    print(f"integrity: {checked} files checked, {verb} {removed} damaged, "
          f"{stale} stale .partial files")
    # Damaged files are repaired by rerunning the owning stage, so this is not
    # itself a failure - report it and let the pipeline recompute.
    sys.exit(0)


if __name__ == "__main__":
    main()
