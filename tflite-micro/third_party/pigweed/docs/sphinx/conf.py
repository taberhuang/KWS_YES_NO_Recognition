# Copyright 2020 The Pigweed Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Pigweed's Sphinx configuration."""


from datetime import date
import json
import os
from pathlib import Path
import sys


from pw_console.pigweed_code_style import PigweedCodeStyle
from pw_console.pigweed_code_style import PigweedCodeLightStyle


# Determine whether the docs are being built with Bazel or GN.
is_bazel_build = True
try:
    from python.runfiles import runfiles  # type: ignore
except ImportError:
    is_bazel_build = False


# The suffix of source filenames.
source_suffix = ['.rst']

# The master toctree document.  # inclusive-language: ignore
master_doc = 'index'

# General information about the project.
project = 'Pigweed'
copyright = f'{date.today().year} The Pigweed Authors'  # pylint: disable=redefined-builtin

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.1'
# The full version, including alpha/beta/rc tags.
release = '0.1.0'


# Pygments plugin approach (https://pygments.org/docs/plugins/) for getting
# Sphinx to use our custom styles doesn't work. Use this approach instead:
# https://stackoverflow.com/q/48615629/1669860
def pygments_monkeypatch_style(mod_name, cls):
    import sys
    import pygments.styles

    cls_name = cls.__name__
    mod = type(__import__('os'))(mod_name)
    setattr(mod, cls_name, cls)
    setattr(pygments.styles, mod_name, mod)
    sys.modules['pygments.styles.' + mod_name] = mod
    from pygments.styles import STYLE_MAP

    STYLE_MAP[mod_name] = mod_name + '::' + cls_name


pygments_monkeypatch_style('pigweed_code_style', PigweedCodeStyle)
pygments_monkeypatch_style('pigweed_code_light_style', PigweedCodeLightStyle)


# //docs/sphinx/_extensions must be added to the system path so that Sphinx
# knows where to find the Sphinx extensions that have been custom-built
# for pigweed.dev. The path resolution changes depending on whether we're
# building pigweed.dev with GN or Bazel.
if is_bazel_build:
    sys.path.append(str(Path('_extensions').resolve()))
else:  # GN build
    pw_root = os.environ['PW_ROOT']
    sys.path.append(f'{pw_root}/docs/_extensions')

extensions = [
    "breathe",
    "bug",  # Custom extension to normalize Pigweed bug links.
    "kconfig",
    "module_metadata",
    "modules_index",
    "pigweed_live",
    "pw_status_codes",
    "pw_docgen.sphinx.google_analytics",  # Enables optional Google Analytics
    "seed_metadata",
    "sitemap",  # Custom extension to handle pigweed.dev sitemap nuances.
    "sphinx.ext.autodoc",  # Automatic documentation for Python code
    "sphinx.ext.napoleon",  # Parses Google-style docstrings
    "sphinxarg.ext",  # Automatic documentation of Python argparse
    "sphinxcontrib.doxylink",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_copybutton",  # Copy-to-clipboard button on code blocks
    "sphinx_reredirects",
]

# When a user clicks the copy-to-clipboard button the `$ ` prompt should not be
# copied: https://sphinx-copybutton.readthedocs.io/en/latest/use.html
copybutton_prompt_text = "$ "

_DIAG_HTML_IMAGE_FORMAT = 'SVG'
blockdiag_html_image_format = _DIAG_HTML_IMAGE_FORMAT
nwdiag_html_image_format = _DIAG_HTML_IMAGE_FORMAT
seqdiag_html_image_format = _DIAG_HTML_IMAGE_FORMAT
actdiag_html_image_format = _DIAG_HTML_IMAGE_FORMAT
rackdiag_html_image_format = _DIAG_HTML_IMAGE_FORMAT
packetdiag_html_image_format = _DIAG_HTML_IMAGE_FORMAT

# Tell m2r to parse links to .md files and add them to the build.
m2r_parse_relative_links = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'Pigweed'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If false, no module index is generated.
html_domain_indices = True

html_favicon = 'https://storage.googleapis.com/pigweed-media/pw_logo.ico'
html_logo = 'https://storage.googleapis.com/pigweed-media/pw_logo.svg'

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/pigweed.css",
    # We could potentially merge the Google Fonts stylesheets into a single network
    # request but we already preconnect with the service in //docs/sphinx/layout/layout.html
    # so the performance impact of keeping these as 3 separate calls should be
    # negligible.
    "https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap",
    "https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap",
    "https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap",
    # FontAwesome for mermaid and sphinx-design
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
]

html_js_files = [
    # Do not list pigweed.js here. This will cause it to get loaded in <head>.
    # To improve load performance we modified //docs/sphinx/layout/layout.html
    # to load pigweed.js at the end of <body> instead.
    # "js/pigweed.js",
]

html_extra_path = [
    # Note: In this repo the file lives at //docs/sphinx/blog/rss.xml but during the
    # Sphinx build it's copied to the root of the website, https://pigweed.dev/rss.xml
    'blog/rss.xml',
]
if is_bazel_build:
    # In the Bazel build, the fully built rustdoc site is present in the Sphinx
    # site's sources directory. Specifying the rustdoc directory here instructs
    # Sphinx to copy over the entire directory to its output.
    html_extra_path.append('rustdoc')
    # Also copy over the Doxygen-generated HTML subsite.
    html_extra_path.append('doxygen')

html_theme_options = {
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#navigation-bar-dropdown-links
    'header_links_before_dropdown': 5,
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#icon-links
    'icon_links': [
        {
            'name': 'Source code',
            'url': 'https://cs.opensource.google/pigweed/pigweed/',
            'icon': 'fa-solid fa-code',
        },
        {
            'name': 'Issue tracker',
            'url': 'https://pwbug.dev',
            'icon': 'fa-solid fa-bug',
        },
        {
            'name': 'Discord',
            'url': 'https://discord.com/channels/691686718377558037/691686718377558040',
            'icon': 'fa-brands fa-discord',
        },
    ],
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/branding.html
    'logo': {
        'text': 'Pigweed',
        'image_light': 'https://storage.googleapis.com/pigweed-media/pw_logo.svg',
        'image_dark': 'https://storage.googleapis.com/pigweed-media/pw_logo.svg',
    },
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#configure-the-navbar-center-alignment
    'navbar_align': 'right',
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/styling.html#configure-pygments-theme
    'pygments_light_style': 'pigweed_code_light_style',
    'pygments_dark_style': 'pigweed_code_style',
}

if 'LUCI_IS_TRY' in os.environ and os.environ['LUCI_IS_TRY'] == '1':
    html_theme_options['announcement'] = (
        "You are viewing Pigweed's docs on an automatically generated staging "
        'site. <b>The content on this staging site may be incorrect or '
        "unapproved.</b> Pigweed's official, approved docs are only published "
        'at <a href="https://pigweed.dev">pigweed.dev</a>.'
    )

# sphinx-sitemap needs this:
# https://sphinx-sitemap.readthedocs.io/en/latest/getting-started.html#usage
html_baseurl = 'https://pigweed.dev/'

# The lefthand "Section Navigation" section is empty for these docs. Hide it.
html_sidebars = {
    'changelog': [],
    'size_optimizations': [],
    'index': [],
    'overview': [],
    'toolchain': [],
}

html_context = {
    'default_mode': 'dark',
}

if 'GOOGLE_ANALYTICS_ID' in os.environ:
    google_analytics_id = os.environ['GOOGLE_ANALYTICS_ID']

# https://sphinx-sitemap.readthedocs.io/en/latest/advanced-configuration.html
sitemap_url_scheme = '{link}'

mermaid_init_js = '''
mermaid.initialize({
  // Mermaid is manually started in //docs/sphinx/_static/js/pigweed.js.
  startOnLoad: false,
  // sequenceDiagram Note text alignment
  noteAlign: "left",
  // Set mermaid theme to the current furo theme
  theme: localStorage.getItem("theme") == "dark" ? "dark" : "default"
});
'''

# Output file base name for HTML help builder.
htmlhelp_basename = 'Pigweeddoc'

# Client-side redirects. See //docs/sphinx/contributing/docs/website.rst.
#
# TODO: https://pwbug.dev/430133030 - Tidy up the redirects.
with open('./redirects.json', 'r') as f:
    redirects_data = json.load(f)
redirects = redirects_data

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', 'pigweed', 'Pigweed', ['Google'], 1)]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        'index',
        'Pigweed',
        'Pigweed',
        'Google',
        'Pigweed',
        'Firmware framework',
        'Miscellaneous',
    ),
]

templates_path = ['layout']
exclude_patterns = ['docs/templates/**']

doxygen_xml_path = (
    './_doxygen/xml/' if is_bazel_build else './../../../doxygen/xml/'
)
# The location of the generated Doxygen tagfile. Note that this must
# match the final location of the generated Doxygen HTML from the
# perspective of the Sphinx build system. The organization of the
# source files (as Sphinx sees it) can be viewed by running this:
# bazelisk build //docs:sources
tagfile_path = os.path.abspath("doxygen/doxygen/index.tag")
# The relative path that Doxylink should use when creating links.
doxygen_site_path = "./doxygen"
doxylink = {
    "doxylink": (tagfile_path, doxygen_site_path),
}

breathe_projects = {
    # Assuming doxygen output is at out/docs/doxygen/
    # This dir should be relative to out/docs/gen/docs/pw_docgen_tree/
    "Pigweed": doxygen_xml_path,
}
breathe_default_project = "Pigweed"
breathe_debug_trace_directives = False
# (b/295023422) Disable the inaccurate `#include` statements that are generated
# when `doxygennamespace` is used.
breathe_show_include = False

# Treat these as valid attributes in function signatures.
cpp_id_attributes = [
    "PW_EXTERN_C_START",
    "PW_NO_LOCK_SAFETY_ANALYSIS",
]
# This allows directives like this to work:
# .. cpp:function:: inline bool try_lock_for(
#     chrono::SystemClock::duration timeout) PW_EXCLUSIVE_TRYLOCK_FUNCTION(true)
cpp_paren_attributes = [
    "PW_EXCLUSIVE_TRYLOCK_FUNCTION",
    "PW_EXCLUSIVE_LOCK_FUNCTION",
    "PW_UNLOCK_FUNCTION",
    "PW_NO_SANITIZE",
]
# inclusive-language: disable
# Info on cpp_id_attributes and cpp_paren_attributes
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-cpp_id_attributes
# inclusive-language: enable

# Disable Python type hints
# autodoc_typehints = 'none'

# Break class and function signature arguments into one arg per line if the
# total length exceeds 130 characters. 130 seems about right for keeping one or
# two parameters on a single line.
maximum_signature_line_length = 130


def do_not_skip_init(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False  # never skip __init__ functions
    return would_skip


# Problem: CSS files aren't copied after modifying them. Solution:
# https://github.com/sphinx-doc/sphinx/issues/2090#issuecomment-572902572
def env_get_outdated(app, env, added, changed, removed):
    return ['index']


def setup(app):
    app.add_css_file('css/pigweed.css')
    app.connect('env-get-outdated', env_get_outdated)
    app.connect("autodoc-skip-member", do_not_skip_init)
