"""Reading North Star Imaging DICOM slice stacks (one file per X slice).

Array convention everywhere in this package: index (k, j, i) = (slice, row, col);
millimetre coordinates are index * voxel size, origin at the first voxel centre of
the FULL-resolution stack. Slicer LPS for the exported/legacy nrrds = (i, j, k) * s.
"""
import glob, os
import numpy as np
import pydicom
from concurrent.futures import ThreadPoolExecutor


def slice_files(xdir):
    files = sorted(glob.glob(os.path.join(xdir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"no .dcm in {xdir}")
    return files


def spacing(xdir):
    ds = dcm(slice_files(xdir)[0], stop_before_pixels=True)
    return float(ds.PixelSpacing[0])


def dcm(fn, tries=6, **kw):
    """dcmread with retries: Box Drive placeholders sometimes fail (OSError 22) while being hydrated."""
    import time
    for t in range(tries):
        try:
            return pydicom.dcmread(fn, **kw)
        except OSError:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def resolve_dir(scan):
    """Use the byte-identical local mirror (scan['local_xdir']) when it exists with the same file count."""
    loc = scan.get("local_xdir")
    if loc and os.path.isdir(loc) and len(glob.glob(os.path.join(loc, "*.dcm"))) == len(slice_files(scan["xdir"])):
        return loc
    return scan["xdir"]


def _block_mean_2d(a, f):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def load_downsampled(xdir, f, threads=8):
    """Block-mean downsample by f in all three axes. Returns (vol float32, voxel_mm, s0_mm, raw_max)."""
    files = slice_files(xdir)
    s0 = spacing(xdir)
    n = len(files) - len(files) % f

    def rd(fn):
        a = dcm(fn).pixel_array
        return _block_mean_2d(a.astype(np.float32), f), int(a.max())

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(rd, files[:n]))
    sl = np.stack([r[0] for r in res])
    vol = sl.reshape(n // f, f, *sl.shape[1:]).mean(1).astype(np.float32)
    return vol, s0 * f, s0, max(r[1] for r in res)


def down_to_full_mm(mm_down, f, s0):
    """Centroid (mm on the downsampled grid) -> mm on the full-res grid (block centre offset)."""
    return np.asarray(mm_down) + (f - 1) / 2 * s0


def read_crop(xdir, k0, k1, j0, j1, i0, i1, threads=8):
    """Full-resolution crop [k0:k1, j0:j1, i0:i1] (uint16)."""
    files = slice_files(xdir)
    k0, k1 = max(0, k0), min(len(files), k1)

    def rd(fn):
        a = dcm(fn).pixel_array
        return a[max(0, j0):j1, max(0, i0):i1]

    with ThreadPoolExecutor(threads) as ex:
        return np.stack(list(ex.map(rd, files[k0:k1]))), (k0, max(0, j0), max(0, i0))
