import time

project = 'HybrA-Filterbanks'
author = 'Daniel Haider, Felix Perfler'
copyright = "{}, {}".format(time.strftime("%Y"), author)
release = "2025.03"
version = "2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_multiversion'
]

templates_path = ['_templates']

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "versioning.html",
    ],
}

smv_remote_whitelist = r"^origin$"
smv_branch_whitelist = r"^main$"

html_theme = 'alabaster'
html_last_updated_fmt = "%c"
master_doc = "index"
pygments_style = "friendly"

