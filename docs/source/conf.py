# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../.'))

import oti_gp

# -- Project information -----------------------------------------------------

project = 'JetGP'
copyright = '2025, Samuel Roberts'
author = 'Samuel Roberts'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",       # Auto-generate documentation from docstrings
    "sphinx.ext.napoleon",      # Supports Google and NumPy style docstrings
    "sphinx.ext.viewcode",      # Adds links to source code
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxarg.ext",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "jupyter_sphinx"
]
bibtex_bibfiles = ["GP_bib.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

templates_path = ['_templates']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/JetGP_logo.png"
html_static_path = ['_static']
html_theme_options = {
    "logo": {"text": release},
    "content_footer_items": ["last-updated"],
    "navigation_depth": 4,
    "repository_url": "https://github.com/Samm-Py/oti_gp",
    "repository_branch": "dev",
    "path_to_docs": "docs/source/",
    "use_source_button": True,
    "collapse_navigation" : True,
    "launch_buttons": {},
    "home_page_in_toc": True,
    "use_repository_button": True,
}
# extensions.append("autoapi.extension")
# extensions.append("numpydoc")
# autoapi_type = 'python'
# autoapi_dirs = ['../../oti_gp']
