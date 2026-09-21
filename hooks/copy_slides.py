import logging
import shutil
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.copy_slides")

def on_pre_build(config, **kwargs):
    docs_dir = Path(config["docs_dir"])
    slides_dir = docs_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    lectures_dir = Path("lectures")
    copied = 0

    for module_dir in sorted(lectures_dir.glob("[0-1][0-9]-*")):
        if not module_dir.is_dir():
            continue
        for pdf in module_dir.glob("*.pdf"):
            dest = slides_dir / pdf.name
            # Avoid re-copying if identical to prevent reload loops during mkdocs serve
            if dest.exists():
                src_stat = pdf.stat()
                dst_stat = dest.stat()
                if dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime >= src_stat.st_mtime:
                    continue
            shutil.copy2(pdf, dest)
            copied += 1

    if copied > 0:
        log.info(f"Copied {copied} lecture slide deck(s) to {slides_dir}")
