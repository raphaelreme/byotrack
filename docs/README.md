# ByoTrack Documentation

The ![documentation](https://byotrack.readthedocs.io/en/latest/) is automatically built and published on ReadTheDocs.

# Build the documentation locally

First setup your development environment with uv (see the Contributing Guide).

In addition, you need to install [pandoc](https://pandoc.org/installing.html) executable
(in addition to the python wrapping lib which is included in the pyproject.toml).

Then build the docs using sphinx
``` bash
$ cd docs/
$ uv run python -m sphinx -E -T source/ build/
```
