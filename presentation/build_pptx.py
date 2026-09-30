"""Build the PowerPoint deck used in the pitch video: one full-bleed slide per video slide, narration as speaker notes.
Run after build_video.py: .venv/bin/python presentation/build_pptx.py
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_video import BUILD, NARRATION, OUT  # noqa: E402

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]
for i, text in enumerate(NARRATION, 1):
    slide = prs.slides.add_slide(blank)
    slide.shapes.add_picture(str(BUILD / f"slide{i:02d}.png"), 0, 0, width=prs.slide_width, height=prs.slide_height)
    slide.notes_slide.notes_text_frame.text = text
prs.save(OUT / "eurovision_pitch.pptx")
print(f"{len(NARRATION)} slides -> {OUT / 'eurovision_pitch.pptx'}")
