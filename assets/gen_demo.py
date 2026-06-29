#!/usr/bin/env python3
"""Generate the Quipu 'pain -> solution' animated terminal SVG for the README.

Self-contained: pure stdlib, no recording tools, no ffmpeg. Emits a GitHub-
friendly animated SVG (CSS keyframe reveal). Run: python assets/gen_demo.py

Static-fallback note: animation-fill-mode:both makes non-animating renderers
(npm, no-CSS viewers) show all lines visible. t=0 rasterization (e.g. GitHub
camo) remains a residual risk — verify on a live GitHub render; fallback plan
if it blanks = ship a static hero image instead.

TODO(public-repo): drive Act 2 from real `python -m quipu search` output
(model-free BM25 path) instead of the scripted line, and add a Pillow GIF
fallback for environments where GitHub camo rasterizes the SVG.
"""
import html, pathlib

# (start_time_s, text, color, font_size, glyph)
# glyph=None: plain text  |  glyph="knot": knot path rendered before text
FRAMES = [
    (0.3, "# fresh Claude session — yesterday's context is gone", "#636B7A", 14, None),
    (0.9, "you ›  what did we decide about the DB, and why?", "#C9D1D9", 15, None),
    (1.6, "claude ›  I don't have any record of that decision…", "#F85149", 15, None),
    (2.5, "────────────────  + quipu memory  ────────────────", "#6366F1", 13, None),
    (3.0, "you ›  what did we decide about the DB?", "#C9D1D9", 15, None),
    (3.5, "claude ›  You chose SQLite over DuckDB — single-file,", "#A5B4FC", 15, None),
    (3.8, "          zero-config; FTS5 gives BM25 in-engine.", "#A5B4FC", 15, None),
    (4.1, " recalled from quipu · R3 · 12 ms · 0 cloud", "#C7D2FE", 13, "knot"),
    (4.5, "Every decision, knotted into one local cord. Local. One file. No cloud.", "#F59E0B", 15, None),
]
W, TOP, LH = 780, 56, 28
H = TOP + len(FRAMES) * LH + 20

TITLE = "Quipu — AI agents forget between sessions; Quipu remembers"
DESC  = "Animated terminal demo: an AI session without Quipu loses context; with Quipu it recalls the SQLite decision instantly."

BAR_COLOR = "#12192B"
BG_COLOR  = "#0D1117"
KNOT_COLOR = "#F59E0B"


def _knot_path(x, y):
    """Return SVG path string for a small knot glyph at text baseline y."""
    ky = y - 9
    return (f"M {x+1},{ky+3} C {x+5},{ky} {x+8},{ky+2} {x+9},{ky+5} "
            f"C {x+10},{ky+8} {x+6},{ky+10} {x+5},{ky+8} "
            f"C {x+4},{ky+6} {x+5},{ky+4} {x+6},{ky+5}")


rows, css = [], []
for i, (t, txt, col, sz, glyph) in enumerate(FRAMES):
    y = TOP + 20 + i * LH
    weight = ' font-weight="700"' if i == len(FRAMES) - 1 else ""
    if glyph == "knot":
        rows.append(
            f'<path class="l{i}" d="{_knot_path(22, y)}" fill="none"'
            f' stroke="{KNOT_COLOR}" stroke-width="1.8"'
            f' stroke-linecap="round" stroke-linejoin="round"/>'
        )
        rows.append(
            f'<text x="37" y="{y}" fill="{col}" font-size="{sz}"'
            f' class="ln l{i}">{html.escape(txt)}</text>'
        )
    else:
        rows.append(
            f'<text x="22" y="{y}" fill="{col}" font-size="{sz}"{weight}'
            f' class="ln l{i}">{html.escape(txt)}</text>'
        )
    css.append(f".l{i}{{animation:sh .15s ease {t}s both}}")

payoff_t = FRAMES[-1][0]

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="SFMono-Regular,Consolas,Menlo,monospace" role="img">
<title>{html.escape(TITLE)}</title>
<desc>{html.escape(DESC)}</desc>
<style>@keyframes sh{{from{{opacity:0;transform:translateX(-4px)}}to{{opacity:1;transform:translateX(0)}}}} .ln{{white-space:pre}}</style>
<rect width="{W}" height="{H}" rx="11" fill="{BG_COLOR}"/>
<rect width="{W}" height="36" rx="11" fill="{BAR_COLOR}"/><rect y="24" width="{W}" height="12" fill="{BAR_COLOR}"/>
<circle cx="22" cy="18" r="6.5" fill="#FF5F56"/><circle cx="44" cy="18" r="6.5" fill="#FFBD2E"/><circle cx="66" cy="18" r="6.5" fill="#27C93F"/>
<text x="{W//2}" y="23" fill="#8B949E" font-size="12.5" text-anchor="middle">quipu — memory for any AI agent</text>
<style>{''.join(css)}</style>
{chr(10).join(rows)}
</svg>'''
out = pathlib.Path(__file__).parent / "demo.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes, {len(FRAMES)} frames, payoff at {payoff_t}s)")
