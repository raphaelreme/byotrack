# ByoTrack Documentation

The ![documentation](https://byotrack.readthedocs.io/en/latest/) is automatically built and published on ReadTheDocs.

# Build the documentation locally

First install extra dependencies:

```bash
$ pip install -r docs/requirements.txt
```

Then build the docs using sphinx
``` bash
$ cd docs/
$ python -m sphinx -E -T source/ build/
```
