"""Figure colours.

Instantiated from a validated categorical/sequential/diverging palette.  Colours
are assigned by the job they do, never by series order:

  categorical  telling generators apart in a ROC or sweep plot -- fixed slot
               order, never cycled; every series also carries a direct label, so
               identity never rests on hue alone
  diverging    the grid heatmap, where 0.5 AUC is "no better than guessing" and
               the reader's question is which side of it a cell falls on.  Red
               is the high pole (leakage), grey is the neutral midpoint, blue is
               below chance.

Swap the hex values here to retarget a different design system; nothing else in
the plotting code refers to a colour directly.
"""

from __future__ import annotations

# Fixed categorical slot order.  Add series by taking the next slot.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Diverging poles and neutral midpoint.
DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8a84"
GRID = "#e4e3df"

#: stable generator -> slot mapping, so a generator keeps its colour across
#: every figure even when a plot shows only a subset
GENERATOR_COLORS = {
    "mvn": CATEGORICAL[0],
    "cvae": CATEGORICAL[1],
    "nd": CATEGORICAL[2],
    "pgm": CATEGORICAL[3],
}

ATTACK_COLORS = {
    "mahalamia": CATEGORICAL[0],
    "melomia_cvae": CATEGORICAL[1],
    "melomia_nd": CATEGORICAL[2],
    "mamamia": CATEGORICAL[3],
}

GENERATOR_LABELS = {
    "mvn": "MVN",
    "cvae": "CVAE",
    "nd": "NoisyDiffusion",
    "pgm": "DP-PGM ($\\varepsilon$=10)",
}

ATTACK_LABELS = {
    "mahalamia": "MahalaMIA",
    "mamamia": "MAMA-MIA",
    "melomia_nd": "MeLoMIA-ND",
    "melomia_cvae": "MeLoMIA-CVAE",
}


def diverging_cmap(name: str = "leakage"):
    """Two-pole map with a neutral grey midpoint -- never a rainbow."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
    )


def apply_style() -> None:
    """Recessive chrome, thin marks, text in ink tokens rather than series hues."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
