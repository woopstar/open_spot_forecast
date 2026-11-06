"""The metadata the Home Assistant Marketplace reads (#244).

Home Assistant 2026.11 ships HACS in core as the Marketplace. It downloads
``hacs.json`` from the release tag and refuses a release whose
``homeassistant`` names a newer core than the user runs. It checks that
``manifest.json`` carries the domain and a version the loader accepts. The
HACS action's ``brands`` check accepts ``brand/icon.png`` in the integration
folder, and its ``integration_categories`` check wants one to three known
categories. These tests pin what those checks read.
"""

import json
import re
import struct
from pathlib import Path
from typing import Any

from awesomeversion import AwesomeVersion, AwesomeVersionStrategy
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_DIR = REPO_ROOT / "custom_components" / "open_spot_forecast"
HACS_JSON = REPO_ROOT / "hacs.json"
MANIFEST = INTEGRATION_DIR / "manifest.json"
BRAND_DIR = INTEGRATION_DIR / "brand"

# The ``energy`` integration categories, from
# https://contributors.home-assistant.io/core/integrations/categories/ (as
# copied for HSEM on 2026-11-03). A value outside the list fails the check.
ENERGY_CATEGORIES = frozenset(
    {
        "energy",
        "energy.backup_power",
        "energy.energy_prices",
        "energy.ev_charging",
        "energy.meters_monitoring",
        "energy.solar_batteries",
    }
)
# hassfest's manifest schema for ``categories``
CATEGORY_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)?$")

# The version formats Home Assistant's loader accepts for a custom integration
LOADABLE_VERSION_STRATEGIES = [
    AwesomeVersionStrategy.CALVER,
    AwesomeVersionStrategy.SEMVER,
    AwesomeVersionStrategy.SIMPLEVER,
    AwesomeVersionStrategy.BUILDVER,
    AwesomeVersionStrategy.PEP440,
]

# The oldest core CI tests against (requirements.txt pins 2026.10.x)
OLDEST_SUPPORTED_HOMEASSISTANT = Version("2026.10.0")


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _pinned_homeassistant() -> Version:
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        name, sep, version = line.partition("==")
        if sep and name.strip() == "homeassistant":
            return Version(version.strip())
    raise AssertionError("requirements.txt does not pin homeassistant")


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG file"
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def test_hacs_json_declares_the_keys_the_marketplace_reads() -> None:
    hacs = _load(HACS_JSON)

    assert isinstance(hacs.get("name"), str) and hacs["name"]
    assert hacs.get("zip_release") is True
    # release.yml uploads this asset
    assert hacs.get("filename") == "open_spot_forecast.zip"
    assert isinstance(hacs.get("homeassistant"), str)


def test_the_minimum_core_is_the_one_ci_tests() -> None:
    raw = _load(HACS_JSON)["homeassistant"]
    minimum = Version(raw)

    assert minimum >= OLDEST_SUPPORTED_HOMEASSISTANT, (
        f"hacs.json offers releases to Home Assistant {minimum}, older than "
        f"any core CI tests ({OLDEST_SUPPORTED_HOMEASSISTANT})"
    )
    assert minimum <= _pinned_homeassistant(), (
        "hacs.json demands a newer Home Assistant than CI tests against"
    )
    # The Marketplace compares it with AwesomeVersion
    assert AwesomeVersion(raw).strategy == AwesomeVersionStrategy.CALVER


def test_the_manifest_declares_the_required_keys() -> None:
    manifest = _load(MANIFEST)

    for key in ("domain", "name", "version", "documentation", "issue_tracker"):
        assert isinstance(manifest.get(key), str) and manifest[key], key
    assert isinstance(manifest.get("codeowners"), list) and manifest["codeowners"]
    assert manifest["domain"] == "open_spot_forecast"


def test_the_manifest_version_is_loadable() -> None:
    """A version the loader rejects rolls the install back."""
    AwesomeVersion(
        _load(MANIFEST)["version"], ensure_strategy=LOADABLE_VERSION_STRATEGIES
    )


def test_the_manifest_categories_are_valid() -> None:
    categories = _load(MANIFEST).get("categories")

    assert isinstance(categories, list), "the integration_categories check warns"
    assert 1 <= len(categories) <= 3
    assert len(set(categories)) == len(categories)
    for category in categories:
        assert CATEGORY_PATTERN.fullmatch(category), category
        assert category in ENERGY_CATEGORIES, category
        parent, _, sub = category.partition(".")
        if sub:
            assert parent not in categories, f"hassfest rejects {parent} with {sub}"


def test_the_brand_icons_ship_inside_the_integration() -> None:
    """The release ZIP is built from the integration folder."""
    assert _png_size(BRAND_DIR / "icon.png") == (256, 256)
    assert _png_size(BRAND_DIR / "icon@2x.png") == (512, 512)
