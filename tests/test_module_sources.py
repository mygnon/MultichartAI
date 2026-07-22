"""Strategy/modules/*.txt must equal a fresh re-extraction from the canonical
Knowledge/*.docx (anti-drift lock)."""
import pytest

from burner.tools.export_modules_from_docx import (KNOWLEDGE, MODULE_DOCX,
                                                   MODULES_DIR, extract_el_text)


@pytest.mark.parametrize("stem", MODULE_DOCX)
def test_txt_matches_docx(stem):
    docx_path = KNOWLEDGE / f"{stem}.docx"
    txt_path = MODULES_DIR / f"{stem}.txt"
    if not docx_path.exists():
        pytest.skip(f"{docx_path.name} absent")
    assert txt_path.exists(), \
        f"{txt_path} missing -- run: py -m burner.tools.export_modules_from_docx"
    assert txt_path.read_text(encoding="ascii") == extract_el_text(docx_path)
