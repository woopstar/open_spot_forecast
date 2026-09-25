"""Every integration module stays below the 30 KB and 1000-line file limit.

See ``.github/memories.md`` → File Size Rules: a file over either limit is
split before more is added to it (#53, #54).
"""

from pathlib import Path

import pytest

INTEGRATION = Path(__file__).parent.parent / "custom_components/open_spot_forecast"
MAX_FILE_BYTES = 30 * 1024
MAX_FILE_LINES = 1000


@pytest.mark.parametrize(
    "path",
    sorted(INTEGRATION.rglob("*.py")),
    ids=lambda path: str(path.relative_to(INTEGRATION)),
)
def test_module_stays_below_the_file_size_limit(path: Path) -> None:
    assert path.stat().st_size < MAX_FILE_BYTES
    assert len(path.read_text(encoding="utf-8").splitlines()) < MAX_FILE_LINES
