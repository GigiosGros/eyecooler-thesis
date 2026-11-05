"""Chapter 3 images taken unchanged from the SCR-Eye Computational Model Report.

The retinal temperature profile along the arc length of the retina was exported from
COMSOL only as an image (no data file exists in Computational model/), so the image is
used as it appears in the report: Figure 17, "Retina temperature profile across the
whole arc length at three times" (image 24 in document order, word/media/image24.png).
The bytes are copied unchanged; nothing is redrawn.

Source: 03 - Experiments & Data/Computational model/Reports/SCR/
        SCR-Eye Computational Model Report.docx
Run:    python code/figures/aim2_report_images.py
"""
import zipfile

from provenance import FIG, SOURCES, THESIS, record, sha256

BOX = THESIS.parents[2]
REPORT = (BOX / "03 - Experiments & Data" / "Computational model" / "Reports" / "SCR"
          / "SCR-Eye Computational Model Report.docx")
MEMBERS = {"retina-profile.png": "word/media/image24.png"}   # report Figure 17


def main():
    (SOURCES / "report").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(REPORT)
    for name, member in MEMBERS.items():
        raw = z.read(member)
        snap = SOURCES / "report" / ("SCR-Eye-Report__" + member.split("/")[-1])
        snap.write_bytes(raw)
        dst = FIG / name
        dst.write_bytes(raw)
        record(__file__, [dst], [snap],
               origins={snap.name: f"{REPORT.relative_to(BOX)} :: {member} (report Figure 17)".replace("\\", "/")},
               notes={"report_docx_sha256": sha256(REPORT)})
        print("wrote", name)


if __name__ == "__main__":
    main()
