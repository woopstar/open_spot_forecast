"""The schema library Home Assistant validates with, as ``vol`` (#195).

Home Assistant replaced voluptuous with probatio, the same public API under
another name. 2026.9 already runs on it behind the old name (it aliases
``import voluptuous`` to probatio at startup); from 2026.10 its own
signatures take ``probatio.Schema`` (``async_show_form(data_schema=...)``,
``services.async_register(schema=...)``). A Home Assistant without probatio
still has voluptuous. Every module of the integration, and its tests, take
``vol`` from here, so the schemas are built with the library the running
Home Assistant uses and type-check against the pinned version.
"""

try:
    import probatio as vol
except ImportError:  # a Home Assistant from before probatio
    import voluptuous as vol  # type: ignore[no-redef]

__all__ = ["vol"]
