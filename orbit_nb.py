"""Orbit theme for Jupyter Notebooks — Palatino, sepia, RISE slides."""

from IPython.display import HTML, display as _display

_CSS = """
<style>
/* ── Orbit theme for Jupyter Notebooks ── */

/* Markdown cells: Palatino + sepia background */
.rendered_html, .text_cell_render,
.jp-RenderedHTMLCommon, .jp-RenderedMarkdown {
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, 'TeX Gyre Pagella', serif !important;
    font-size: 17px !important;
    line-height: 1.65 !important;
    color: #222 !important;
}

/* Sepia background for markdown cells */
.text_cell, .jp-MarkdownCell,
.text_cell_render, .jp-RenderedHTMLCommon,
.jp-MarkdownOutput {
    background-color: #f5f0e8 !important;
}

/* Paragraphs */
.rendered_html p, .jp-RenderedHTMLCommon p {
    text-align: justify;
    margin-bottom: 0.9em;
}

/* Headings */
.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4,
.jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3, .jp-RenderedHTMLCommon h4 {
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Links */
.rendered_html a, .jp-RenderedHTMLCommon a {
    color: #1a4ed8;
    text-decoration: none;
}
.rendered_html a:hover, .jp-RenderedHTMLCommon a:hover {
    text-decoration: underline;
}

/* Code */
.rendered_html code, .jp-RenderedHTMLCommon code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.88em;
    background: #f6f8fa;
    padding: 0.15em 0.35em;
    border-radius: 3px;
}

/* Tables */
.rendered_html table, .jp-RenderedHTMLCommon table {
    border-collapse: collapse;
    margin: 1.2em auto;
}
.rendered_html th, .rendered_html td,
.jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td {
    border: 1px solid #ddd;
    padding: 6px 12px;
}
.rendered_html th, .jp-RenderedHTMLCommon th {
    background: #f2f2f2;
    font-weight: 600;
}

/* Blockquotes */
.rendered_html blockquote, .jp-RenderedHTMLCommon blockquote {
    border-left: 4px solid #ddd;
    padding-left: 12px;
    color: #555;
    margin: 1em 0;
}

/* ── RISE / Reveal.js slides ── */

.reveal .slides {
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 180% !important;
    line-height: 1.5;
    color: #222;
}
.reveal .slides h1 { font-size: 2.0em !important; font-weight: 600; }
.reveal .slides h2 { font-size: 1.6em !important; font-weight: 600; }
.reveal .slides h3 { font-size: 1.3em !important; font-weight: 600; }
.reveal .slides p  { text-align: left; margin-bottom: 0.6em; }
.reveal .slides code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.75em;
}
.reveal .slides table { font-size: 0.85em; }
.reveal .slides img { max-width: 90%; max-height: 70vh; }
.reveal .slides .MathJax { font-size: 110%; }

/* Sepia background in slides too */
.reveal .slides section {
    background-color: #f5f0e8 !important;
}
</style>
"""

_display(HTML(_CSS))
