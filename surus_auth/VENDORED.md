# Vendored: surus-auth v2.2.6

Copied byte-for-byte from `github.com/surusgop/surus-auth`, tag `v2.2.6`
(commit `a00f367c2dcc7f43b0e501982784d9bd5fbfb222`), `src/surus_auth/`.
Only the stdlib-only core and the ASGI adapter are included: `__init__.py`,
`_engine.py`, `asgi.py`. License: MIT (see the upstream `pyproject.toml`).

Why vendored instead of `pip install`: this image's Python environment is
sealed. `template_helpers/build_hermes_image.py` builds Hermes' frozen
dependency graph and only *verifies* `requirements.txt` against it; the
wrapper venv has no pip, so a git dependency (which would also need `GH_PAT`
at build time) cannot be installed. surus-auth's core and ASGI adapter have no
third-party dependencies, so copying them is safe.

To update: copy the same three files from a newer tag, keep them unmodified,
and update the tag and commit above.
