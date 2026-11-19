"""Figure 1 of the paper: a membership inference attack on synthetic data (a schematic).

Writes paper/Fig/hero.tex; build it with  cd paper/Fig && tectonic -X compile hero.tex
The table values are random; each candidate's cells are the first genes of the same
person's training row, so members can be matched by eye.
"""
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "paper" / "Fig" / "hero.tex"

PRE = r"""\documentclass[tikz,border=2pt]{standalone}
\usepackage{fontspec}
\setmainfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usetikzlibrary{arrows.meta,calc,decorations.pathreplacing}
\definecolor{ink}{HTML}{0B0B0B}
\definecolor{ink2}{HTML}{52514E}
\definecolor{muted}{HTML}{898781}
\definecolor{hair}{HTML}{C3C2B7}
\definecolor{grid}{HTML}{E1E0D9}
\definecolor{expr}{HTML}{3D4A5C}
\definecolor{syn}{HTML}{1E9A6A}
\definecolor{risk}{HTML}{D23A2E}
\newcommand{\ttl}{\fontsize{8}{9.6}\selectfont\bfseries}
\newcommand{\txt}{\fontsize{7}{8.4}\selectfont}
\newcommand{\ann}{\fontsize{6}{7.2}\selectfont}
\tikzset{
  >={Stealth[length=4pt,width=3.4pt]},
  flow/.style={->,draw=ink2,line width=0.7pt},
  ann/.style={font=\ann,text=ink2,align=center,inner sep=1.5pt},
  txt/.style={font=\txt,text=ink,align=center,inner sep=1.5pt},
  pname/.style={font=\ann,text=ink2,anchor=east,inner sep=1pt},
  box/.style={draw=hair,rounded corners=2pt,line width=0.6pt,fill=white,
              font=\txt,text=ink,align=center,inner sep=2pt},
  nn/.style={circle,draw=ink2,fill=white,line width=0.45pt,inner sep=0pt,minimum size=4.2pt},
}
\begin{document}
\begin{tikzpicture}
"""
END = "\\end{tikzpicture}\n\\end{document}\n"

def shade(v, col="expr", lo=8, hi=78):
    return f"{col}!{int(lo + (hi - lo) * v)}!white"


def table(x0, ytop, rows, pitch, sq, col="expr"):
    """rows: list of value lists. Returns tikz and the y centre of each row."""
    out, ys = [], []
    for i, row in enumerate(rows):
        y = ytop - pitch * (i + 1)
        ys.append(y + sq / 2)
        for j, v in enumerate(row):
            out.append(f"\\fill[{shade(v, col)}] ({x0 + pitch * j:.3f},{y:.3f}) rectangle ++({sq},{sq});")
    return "\n".join(out), ys



rng = random.Random(4)
N_GENES, N_ADV = 12, 7
TRAIN = ["Ana", "Ben", "Cal", "Dee"]
OUTSIDE = ["Ian", "Joy", "Max", "Liv"]
CANDS = [("Cal", .92), ("Ian", .18), ("Ana", .80), ("Joy", .55), ("Dee", .38), ("Max", .12), ("Liv", .26)]
types = [[rng.random() for _ in range(N_GENES)] for _ in range(3)]
prof = lambda t: [min(1, max(0, v + rng.gauss(0, .17))) for v in types[t]]
people = {n: prof(i % 3) for i, n in enumerate(TRAIN + OUTSIDE)}
synthetic = [prof(rng.randrange(3)) for _ in range(6)]
nT, nS, nC = len(TRAIN), len(synthetic), len(CANDS)

P, S = 0.17, 0.152
pre = PRE.replace(r"name/.style={font=\ann,text=ink2", r"name/.style={font=\ann,text=muted")
t = [pre]
x0, ytop = 1.0, 0.0
tik, ys = table(x0, ytop, [people[n] for n in TRAIN], P, S); t.append(tik)
for n, y in zip(TRAIN, ys):
    t.append(f"\\node[pname] at ({x0 - 0.04},{y:.3f}) {{{n}}};")
tw = P * N_GENES
t.append(f"\\node[txt,anchor=north] at ({x0 + tw / 2:.2f},{ytop - P * nT - 0.06:.2f}) "
         r"{\textbf{Training data}\\{\ann patients $\times$ genes}};")
mid = ytop - P * nT / 2
gx = x0 + tw + 1.75
t.append(f"\\node[box,minimum width=1.55cm,minimum height=1.0cm] (gen) at ({gx:.2f},{mid:.2f}) {{}};")
t.append(f"\\node[font=\\txt,text=ink,anchor=south] at ({gx:.2f},{mid + 0.52:.2f}) {{\\textbf{{Generator}}}};")
pts = [(-.48, .22), (-.18, .36), (.12, .12), (.45, .3), (-.38, -.2), (-.02, -.3), (.36, -.16), (.5, -.38)]
edges = [(0, 1), (0, 4), (1, 2), (2, 3), (2, 5), (4, 5), (5, 6), (6, 3), (6, 7), (1, 3)]
for a, b in edges:
    t.append(f"\\draw[hair!80!ink2,line width=0.45pt] ({gx + pts[a][0]:.3f},{mid + pts[a][1]:.3f}) -- ({gx + pts[b][0]:.3f},{mid + pts[b][1]:.3f});")
for x, y in pts:
    t.append(f"\\node[nn] at ({gx + x:.3f},{mid + y:.3f}) {{}};")
t.append(f"\\draw[flow] ({x0 + tw + 0.08:.2f},{mid:.2f}) -- node[above,ann] {{train}} (gen.west);")
sx = gx + 1.75
sytop = mid + P * nS / 2
tik, _ = table(sx, sytop, synthetic, P, S, col="syn"); t.append(tik)
t.append(f"\\draw[flow] (gen.east) -- node[above,ann] {{sample}} ({sx - 0.08:.2f},{mid:.2f});")
t.append(f"\\node[txt,anchor=north] at ({sx + tw / 2:.2f},{sytop - P * nS - 0.06:.2f}) "
         r"{\textbf{Synthetic data} {\ann (released)}};")
# adversary
low = min(ytop - P * nT - 0.5, sytop - P * nS - 0.35)
cy = low - 0.95
tik, cys = table(x0, cy, [people[n][:N_ADV] for n, _ in CANDS], P, S); t.append(tik)
for (n, _), y in zip(CANDS, cys):
    t.append(f"\\node[pname] at ({x0 - 0.04},{y:.3f}) {{{n}}};")
cw = P * N_ADV
t.append(f"\\node[txt,anchor=north] at ({x0 + cw / 2:.2f},{cy - P * nC - 0.06:.2f}) {{\\textbf{{Candidates}}}};")
t.append(f"\\draw[draw=ink2,dashed,line width=0.5pt,rounded corners=3pt] "
         f"({x0 - 0.6:.2f},{cy + 0.62:.2f}) rectangle ({sx + tw - 0.2:.2f},{cy - P * nC - 0.5:.2f});")
t.append(f"\\node[ann,anchor=north west,text=ink2] at ({x0 - 0.58:.2f},{cy + 0.6:.2f}) {{\\textbf{{The adversary}}}};")
cmid = cy - P * nC / 2
t.append(f"\\node[box,minimum width=1.55cm,minimum height=0.8cm,fill=risk!6,draw=risk!45] (atk) at ({gx:.2f},{cmid:.2f}) "
         r"{\textbf{Attack}};")
t.append(f"\\draw[flow] ({x0 + cw + 0.08:.2f},{cmid:.2f}) -- (atk.west);")
t.append(f"\\draw[flow] ({sx - 0.08:.2f},{sytop - P * (nS - 1):.2f}) to[out=-150,in=60] (atk.north east);")
scx = sx + 0.55
for (n, s), y in zip(CANDS, cys):
    t.append(f"\\node[pname] at ({scx - 0.06:.2f},{y:.3f}) {{{n}}};")
    t.append(f"\\fill[risk!{int(4 + 88 * s)}!white] ({scx:.2f},{y - S / 2:.3f}) rectangle ++({2.2 * S:.3f},{S:.3f});")
t.append(f"\\draw[flow] (atk.east) -- ({scx - 0.55:.2f},{cmid:.2f});")
t.append(f"\\node[ann,anchor=south,align=center] at ({scx + 0.17:.2f},{cy:.2f}) {{membership\\\\confidence}};")
t.append(END)
OUT.write_text("\n".join(t))
print("written")
