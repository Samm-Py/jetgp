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
from pathlib import Path
import shutil

sys.path.insert(0, os.path.abspath('../../.'))

import jetgp
import subprocess

# -- Coverage Generation -----------------------------------------------------

import glob

def generate_coverage():
    """Generate coverage report before building docs."""
    # Get paths
    conf_dir = Path(__file__).parent  # docs/source
    project_root = conf_dir.parent.parent  # project root
    test_dir = project_root / "unit_tests"
    coverage_output = conf_dir / "coverage"
    
    print("=" * 70)
    print("GENERATING COVERAGE REPORT")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Test directory: {test_dir}")
    print(f"Coverage output: {coverage_output}")
    
    if not test_dir.exists():
        print(f"⚠ WARNING: Test directory not found at {test_dir}")
        print("Skipping coverage generation.")
        return
    
    # Ensure output directory parent exists
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all test files using glob
    test_files = sorted(glob.glob(str(test_dir / "*.py")))
    print(f"\nFound {len(test_files)} test files:")
    for tf in test_files:
        print(f"  - {Path(tf).name}")
    
    print("\nRunning tests with coverage...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
        ] + test_files + [  # Add test files as separate arguments
            "--cov=jetgp",
            f"--cov-report=html:{coverage_output}",
            "--cov-report=term-missing",
            "-v"
        ],
        cwd=str(project_root),
        text=True
    )
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr)
    
    if result.returncode == 0:
        print(f"\n✓ Coverage report generated successfully at: {coverage_output}")
    else:
        print(f"\n⚠ Tests completed with return code {result.returncode}")
        print("Coverage report may still be available")
    
    print("=" * 70)


def copy_coverage_to_build(app, exception):
    """Copy coverage HTML to build directory after build completes."""
    if exception is None:  # Only if build succeeded
        source_coverage = Path(app.srcdir) / "coverage"
        build_coverage = Path(app.outdir) / "_static" / "coverage"
        
        if source_coverage.exists():
            print("=" * 70)
            print(f"Copying coverage from {source_coverage} to {build_coverage}")
            if build_coverage.exists():
                shutil.rmtree(build_coverage)
            shutil.copytree(source_coverage, build_coverage)
            print("✓ Coverage files copied to build directory")
            print("=" * 70)
        else:
            print(f"⚠ WARNING: Coverage source not found at {source_coverage}")

def setup(app):
    """Sphinx setup hook."""
    app.connect('build-finished', copy_coverage_to_build)

# Generate coverage when building docs
generate_coverage()

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
html_static_path = ['_static']  # Remove 'coverage' from here
html_theme_options = {
    "logo": {"text": release},
    "content_footer_items": ["last-updated"],
    "navigation_depth": 4,
    "repository_url": "https://github.com/Samm-Py/jetgp",
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