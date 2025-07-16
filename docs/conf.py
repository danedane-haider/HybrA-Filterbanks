import time
import tomli
with open("../pyproject.toml", "rb") as f:
    toml = tomli.load(f)

project = 'HybrA-Filterbanks'
author = 'Daniel Haider, Felix Perfler'
copyright = "{}, {}".format(time.strftime("%Y"), author)
release = toml['project']['version']
version = toml['project']['version'].split('.')[0]

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

