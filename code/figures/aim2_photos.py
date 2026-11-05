"""Chapter 3 photographs, cropped from the originals (the ring photograph is also mirrored).

Fig. 3.4 (fig:aim2-rings-photo): the three rings tested in human donor eyes, side view
on a ruler. Source: 08 - Media & Figures/Ring/Rings Comparison Side.jpg (1536 x 2048 px),
copied unchanged to figures/aim2/sources/photos/. The crop keeps the ruler and the
rings; the box is set by hand (styling) and recorded below. The crop is then mirrored
left to right so that the rings appear in the order original, ceramic, reduced-bulk.

Fig. fig:aim2-board: the integrated control board (revision 1.0) with the Nucleo
microcontroller board mounted. Source: my photograph of 2026-10-06
(20261006_162915234_iOS.heic, 3024 x 4032 px), kept unchanged in
figures/aim2/sources/photos/. In the photograph the horizontal edges of the boards are
level (0.1 deg) but the vertical edges lean by about 1 deg (the camera was not square
to the board), so the image is sheared horizontally by the measured lean of the vertical
edges, which makes them vertical without turning the horizontal ones, and then cropped
to the board; nothing else is changed. Needs pillow-heif.

Run:  python code/figures/aim2_photos.py
"""
import shutil

from PIL import Image

from provenance import FIG, SOURCES, THESIS, record

BOX = THESIS.parents[2]
SRC = BOX / "08 - Media & Figures" / "Ring" / "Rings Comparison Side.jpg"
CROP = (100, 720, 1237, 1000)       # left, top, right, bottom in px of the original; equal margin either side of the rings
BOARD = SOURCES / "photos" / "control-board-20261006_162915234_iOS.heic"
BOARD_LEAN = -0.0165               # dx/dy of the vertical board edges in the original, px per px
                                   # (Nucleo left -0.020, right -0.012, lower board left -0.016)
BOARD_CROP = (490, 845, 2490, 3190)  # px of the rotated photograph


def main():
    (SOURCES / "photos").mkdir(parents=True, exist_ok=True)
    snap = SOURCES / "photos" / SRC.name
    shutil.copy2(SRC, snap)
    dst = FIG / "rings-photo.jpg"
    # mirrored left to right (2026-10-08) so that the rings read original, ceramic, reduced-bulk
    Image.open(snap).crop(CROP).transpose(Image.FLIP_LEFT_RIGHT).save(dst, quality=95)
    record(__file__, [dst], [snap], origins={snap.name: str(SRC.relative_to(BOX)).replace("\\", "/")},
           notes=f"crop box {CROP} (px), then mirrored left to right")
    print("wrote", dst.name)

    import pillow_heif
    from PIL import ImageOps
    pillow_heif.register_heif_opener()
    dst = FIG / "board-photo.jpg"
    im = ImageOps.exif_transpose(Image.open(BOARD)).convert("RGB")
    yc = im.size[1] / 2                                   # shear about the mid-height
    im = im.transform(im.size, Image.AFFINE, (1, BOARD_LEAN, -BOARD_LEAN * yc, 0, 1, 0),
                      resample=Image.BICUBIC)
    im.crop(BOARD_CROP).save(dst, quality=92)
    record(__file__, [dst], [BOARD],
           origins={BOARD.name: "my photograph, 2026-10-06"},
           notes=f"horizontal shear {BOARD_LEAN} px/px about mid-height, crop box {BOARD_CROP} "
                 "(px of the sheared image)")
    print("wrote", dst.name)


if __name__ == "__main__":
    main()
