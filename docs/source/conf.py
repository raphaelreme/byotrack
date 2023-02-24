# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "ByoTrack"
copyright = "2023, Raphael Reme"
author = "Raphael Reme"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "nbsphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_title = "ByoTrack"

# -- Options for EPUB output
epub_show_urls = "footnote"


# -- Options for autodoc

# Don't keep undoc members, yields a lot of duplicates with cls variable and property...
# Private or special members should be included by hand if needed
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
