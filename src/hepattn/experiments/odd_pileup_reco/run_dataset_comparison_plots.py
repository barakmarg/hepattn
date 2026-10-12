"""Overlaid PU200 vs fully simulated PU200: dataset-level distribution comparison.

The training sample (``ttbar_pu0_overlay_pu200``) is built by overlaying
Poisson(200) pileup events onto pileup-free ttbar, rather than by simulating
PU200 directly. This asks the obvious question: does the result look like the
real thing? Seven distributions are drawn for both samples on shared bins --
calorimeter occupancy and energy, track multiplicity and energy, and the
particle<->cluster incidence structure.

This is a *dataset* check, not a model check: no checkpoint, no predictions, no
Hungarian matching. That is why it lives here rather than in
``run_paper_performance_plots.py``, which is a map-reduce over H5 prediction
shards and whose ``render_all()`` consumes a merged H5 aggregate.

Ported from ``PileupODD/scripts/compare_pu0overlay_vs_pu200_stats.py``, restyled
to match the paper figures and split into standalone single-panel PDFs.

Usage::

    python run_dataset_comparison_plots.py --out-dir dataset_comparison

Every default is pinned, so a re-run cannot silently give a different sample
size or a different pair of datasets.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Importing the paper module applies its rcParams -- the HEP tick style (inward
# on all four sides, minor ticks visible), the major-only grid, and the base
# font sizes. Sharing them is the point: these figures must sit next to the
# performance figures without looking like they came from a different paper.
from hepattn.experiments.odd_pileup_reco.run_paper_performance_plots import (  # noqa: F401
    GLOWUP_COLOR,
    TARGET_COLOR,
    save_fig,
)

import matplotlib.pyplot as plt

# ── Defaults ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/storage/agrp/barakma/PileupODD/data")
#: The sample the model is trained on: PU0 ttbar with Poisson(200) pileup overlaid.
DEFAULT_OVERLAY_DIR = DATA_ROOT / "ttbar_pu0_overlay_pu200"
#: The reference: ttbar simulated at PU200 directly.
DEFAULT_FULLSIM_DIR = DATA_ROOT / "ttbar_pu200"
DEFAULT_N_EVENTS = 1000

# Kept short: at a large legend font, "Overlaid PU200 (training sample)" was
# wider than the 7.5 in axes. _fit_headroom() handles the legend's height, but
# nothing shrinks its width, so the label carries that constraint.
OVERLAY_LABEL = "Overlaid PU200 (training)"
FULLSIM_LABEL = "Fully simulated PU200"

#: Tables read from each dataset. One row per event, list-valued columns.
TABLES = ("calo_clusters", "tracks", "target_particles_deps")


# ── IO, per-shard extraction, and the cache ───────────────────────────────
#
# Everything is computed one shard at a time and cached, for two reasons.
# Memory: a 10k-event pass over three list-valued tables held in one frame runs
# to many GB, while a shard at a time stays flat. Cost: the extraction is the
# expensive part (~3 s/shard), and 10k events means 110 shards, so a re-run that
# only changes styling should not pay it again.
#
# Every quantity here is computable within a shard and then concatenated --
# the two incidence counts group by (event_id, idx) and events never span
# shards -- so a per-shard cache is exact, not an approximation.

def _shard_index(path: Path) -> str:
    """``calo_clusters-00042.parquet`` -> ``00042``."""
    return path.stem.rsplit("-", 1)[-1]


def _compress(a: np.ndarray) -> np.ndarray:
    """Narrow an extracted array before it is cached.

    These arrays only ever feed histogram binning, so float64 buys nothing:
    the energies carry ~7 significant digits and the counts are small integers.
    Narrowing roughly halves both the cache on disk and the time spent reading
    it back, which is most of the cost of a cached re-run.
    """
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int32)
    return a


def _extract_shard(directory: Path, idx: str) -> dict:
    """Read one shard's three tables and reduce them to the panel quantities."""
    d = {t: pl.read_parquet(directory / f"{t}-{idx}.parquet") for t in TABLES}
    n = d["calo_clusters"].height
    return {"n_events": n,
            "arrays": {k: _compress(spec[0](d)) for k, spec in PANELS.items()}}


def collect(directory: Path, n_events: int, state_dir: Path, sample: str,
            force: bool) -> tuple[dict[str, np.ndarray], int]:
    """Gather the panel quantities over enough whole shards to cover ``n_events``.

    Returns the concatenated arrays and the event count actually used. Whole
    shards only: the flat per-cluster and per-track arrays carry no event
    boundary, so a partial shard could not be trimmed to an event count. Shard
    sizes differ between the two samples (1000 events for the overlay sample,
    100 for the fully simulated one), so the shard count is derived per sample.
    """
    shards = [_shard_index(p) for p in sorted(directory.glob("calo_clusters-*.parquet"))]
    if not shards:
        sys.exit(f"FATAL: no calo_clusters-*.parquet in {directory}")

    cache_dir = state_dir / sample
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts: list[dict] = []
    total = 0
    n_cached = n_computed = 0
    for idx in shards:
        if total >= n_events:
            break
        cache_path = cache_dir / f"{idx}.pkl"
        part = None
        if cache_path.is_file() and not force:
            try:
                with open(cache_path, "rb") as fh:
                    part = pickle.load(fh)
                # A shard cached before a panel was added lacks that quantity.
                # Re-extract it rather than fail, so adding a panel never needs
                # --force; changing an existing extractor still does.
                if not set(PANELS) <= set(part["arrays"]):
                    part = None
                else:
                    n_cached += 1
            except Exception as exc:          # noqa: BLE001 - corrupt cache -> recompute
                print(f"  cache unreadable for {sample}/{idx} ({exc}); recomputing",
                      file=sys.stderr)
        if part is None:
            part = _extract_shard(directory, idx)
            n_computed += 1
            with open(cache_path, "wb") as fh:
                pickle.dump(part, fh, protocol=pickle.HIGHEST_PROTOCOL)
        parts.append(part)
        total += part["n_events"]

    if total < n_events:
        sys.exit(f"FATAL: {directory.name} has only {total} events across "
                 f"{len(shards)} shard(s), need {n_events}")

    arrays = {k: np.concatenate([p["arrays"][k] for p in parts]) for k in PANELS}
    print(f"  {sample:8s} {directory.name}: {len(parts)} shard(s), {total} events "
          f"({n_cached} cached, {n_computed} computed)")
    return arrays, total



# ── Per-quantity extractors: each takes one shard's tables, returns a 1-D array ──

def n_clusters_per_event(d: dict) -> np.ndarray:
    return d["calo_clusters"]["cluster_id"].list.len().to_numpy()


def cluster_energy(d: dict) -> np.ndarray:
    return d["calo_clusters"]["total_cluster_energy"].explode().drop_nulls().to_numpy()


def sum_cluster_energy_per_event(d: dict) -> np.ndarray:
    return d["calo_clusters"]["total_cluster_energy"].list.sum().to_numpy()


def n_tracks_per_event(d: dict) -> np.ndarray:
    return d["tracks"]["pt"].list.len().to_numpy()


def track_energy(d: dict) -> np.ndarray:
    """E = pt * cosh(eta), the massless approximation."""
    flat = d["tracks"].select(["pt", "eta"]).explode(["pt", "eta"]).drop_nulls()
    return flat["pt"].to_numpy().astype(np.float64) * np.cosh(
        flat["eta"].to_numpy().astype(np.float64))


def _deps_count(d: dict, group: str, counted: str) -> np.ndarray:
    """Distinct ``counted`` per (event, ``group``) in the incidence table.

    ``particle_idx`` and ``cluster_idx`` are positional *within* an event, so
    the group-by carries ``event_id`` and no cross-event or cross-shard join on
    an index is ever performed -- which is also what makes the per-shard cache
    exact rather than an approximation.
    """
    return (
        d["target_particles_deps"].lazy()
        .select(["event_id", "particle_idx", "cluster_idx"])
        .explode(["particle_idx", "cluster_idx"])
        .group_by(["event_id", group])
        .agg(pl.col(counted).n_unique().alias("n"))
        .collect()["n"].to_numpy()
    )


def particles_per_cluster(d: dict) -> np.ndarray:
    return _deps_count(d, "cluster_idx", "particle_idx")


def clusters_per_particle(d: dict) -> np.ndarray:
    return _deps_count(d, "particle_idx", "cluster_idx")



def _flat(d: dict, table: str, col: str) -> np.ndarray:
    """One list column of one table, flattened over every object in the shard."""
    return d[table][col].explode().drop_nulls().to_numpy().astype(np.float64)


def track_eta(d: dict) -> np.ndarray:
    return _flat(d, "tracks", "eta")


def track_phi(d: dict) -> np.ndarray:
    return _flat(d, "tracks", "phi")


def cluster_eta(d: dict) -> np.ndarray:
    return _flat(d, "calo_clusters", "cluster_eta")


def cluster_phi(d: dict) -> np.ndarray:
    return _flat(d, "calo_clusters", "cluster_phi")


# ── Panel definitions ─────────────────────────────────────────────────────
# name -> (extractor, xlabel, n_bins, integer-x?, log-x?, log-y?)
#
# log-y only where the distribution actually falls over orders of magnitude.
# The per-event multiplicities are unimodal, and on a linear axis the overlay
# vs full-simulation offset is immediately visible instead of being squashed.
PANELS = {
    "clusters_per_event": (
        n_clusters_per_event, "Clusters / event", 60, True, False, False),
    "cluster_energy": (
        cluster_energy, "Cluster energy [GeV]", 80, False, True, True),
    "track_energy": (
        track_energy, r"Track energy $p_T \cosh\eta$ [GeV]", 80, False, True, True),
    "particles_per_cluster": (
        particles_per_cluster, "Target particles / cluster", 60, True, False, True),
    "clusters_per_particle": (
        clusters_per_particle, "Clusters / target particle", 30, True, False, True),
    "sum_cluster_energy": (
        sum_cluster_energy_per_event, "Sum cluster energy / event [GeV]", 60, False, False, False),
    "tracks_per_event": (
        n_tracks_per_event, "Tracks / event", 60, True, False, False),
    # Angular acceptance. Linear y: both are broad and flat-topped, and the
    # comparison of interest is a few-percent shape difference, not a tail.
    "track_eta": (track_eta, r"Track $\eta$", 70, False, False, False),
    "track_phi": (track_phi, r"Track $\phi$", 72, False, False, False),
    "cluster_eta": (cluster_eta, r"Cluster $\eta$", 70, False, False, False),
    "cluster_phi": (cluster_phi, r"Cluster $\phi$", 72, False, False, False),
}


#: Minimum entries a bin should hold. Poisson noise per bin is 1/sqrt(N), so 40
#: entries is ~16% -- visible but not dominant. The per-event quantities have
#: only one entry per event (1000), where the requested 60 bins would leave ~15
#: per bin and ~26% noise. On a log axis that noise is visually compressed; on a
#: linear one it reads as structure that is not there, which is exactly what made
#: the linear tracks/event panel look worse than its log-scaled predecessor.
#: The flat per-cluster/per-track arrays have millions of entries and are never
#: capped by this.
MIN_ENTRIES_PER_BIN = 40


def _shared_edges(a: np.ndarray, b: np.ndarray, n_bins: int,
                  integer: bool, log_x: bool) -> np.ndarray:
    """Bin edges spanning both samples, so the two histograms are comparable.

    ``n_bins`` is an upper bound, not a target: it is reduced when the smaller
    sample cannot fill that many bins to :data:`MIN_ENTRIES_PER_BIN`.
    """
    n_bins = max(10, min(n_bins, min(a.size, b.size) // MIN_ENTRIES_PER_BIN))
    if log_x:
        a, b = a[a > 0], b[b > 0]
        return np.logspace(np.log10(min(a.min(), b.min())),
                           np.log10(max(a.max(), b.max())), n_bins)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if integer:
        step = max(1, int(np.ceil((int(hi) - int(lo) + 1) / n_bins)))
        return np.arange(int(lo) - 0.5, int(hi) + 0.5 + step, step)
    return np.linspace(lo, hi, n_bins)


def _drawn_points(ax) -> tuple[np.ndarray, np.ndarray]:
    """Every (x, y) vertex actually drawn in the axes.

    ``histtype="stepfilled"`` is one Polygon and the step outline is a Line2D,
    so both artist families have to be walked.
    """
    xs, ys = [], []
    for patch in ax.patches:
        xy = np.asarray(patch.get_xy(), dtype=float)
        if xy.ndim == 2 and len(xy):
            xs.append(xy[:, 0]); ys.append(xy[:, 1])
    for line in ax.lines:
        xy = np.asarray(line.get_xydata(), dtype=float)
        if xy.ndim == 2 and len(xy):
            xs.append(xy[:, 0]); ys.append(xy[:, 1])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _fit_headroom(ax, legend, log_y: bool, pad: float = 0.04) -> None:
    """Raise the top of the y axis until the legend clears the data beneath it.

    Two things matter here, and only the first is obvious.

    A fixed headroom multiplier only works for one legend size -- the box grows
    with its font, so a factor tuned at one size puts the bars through the
    legend at another. So the rendered legend is measured instead.

    Less obvious: what has to clear the legend is the data *under the legend*,
    not the tallest bar anywhere. The legend sits in the upper right, and on a
    steeply falling spectrum the curve below it is orders of magnitude under the
    peak on the far left. Fitting to the global maximum therefore reserves a
    huge empty band and stretches the axis for nothing. Restricting the maximum
    to the legend's own x-span lets the legend drop to where the data actually
    is, and the axis shrinks with it.
    """
    fig = ax.figure
    fig.canvas.draw()                       # no extent until the legend is drawn
    box = legend.get_window_extent()
    frac = box.transformed(ax.transAxes.inverted()).y0 - pad
    if not 0.05 < frac < 1.0:               # pathological legend; leave it alone
        return

    xs, ys = _drawn_points(ax)
    if not len(xs):
        return
    # Legend x-span in data coordinates; transData handles the log axis.
    (lx0, _), (lx1, _) = ax.transData.inverted().transform(
        [(box.x0, box.y0), (box.x1, box.y1)])
    ylo, _ = ax.get_ylim()
    ys = ys[np.isfinite(ys)]
    if log_y:
        ylo = max(ylo, 1e-9)
        keep = ys > ylo
        xs, ys = xs[keep], ys[keep]
    if not len(ys):
        return

    under = (xs >= min(lx0, lx1)) & (xs <= max(lx0, lx1))
    local = float(ys[under].max()) if under.any() else float(ys.max())
    overall = float(ys.max())

    # TWO constraints, and the axis has to satisfy both:
    #   1. the data *under the legend* must sit below the legend's bottom edge;
    #   2. the tallest bar *anywhere* must still fit inside the axis.
    # Only the first was applied before, which silently clipped the peak off
    # every panel whose maximum is not beneath the legend -- the symmetric
    # per-event distributions, and worst of all the falling spectra, where the
    # legend's x-span sits far out in a tail orders of magnitude below the peak.
    if log_y:
        lo = np.log10(ylo)
        top = 10 ** max(lo + (np.log10(local) - lo) / frac,
                        np.log10(overall) + 0.05 * (np.log10(overall) - lo))
    else:
        top = max(ylo + (local - ylo) / frac, ylo + (overall - ylo) * 1.05)
    ax.set_ylim(ylo, top)


def draw_panel(ax, key: str, overlay: np.ndarray, fullsim: np.ndarray,
               legend_fontsize: float = 16.8, tick_labelsize: float | None = None,
               axis_label_fontsize: float | None = None,
               n_overlay: int | None = None, n_fullsim: int | None = None) -> None:
    """Draw one comparison panel. Shared by the standalone figures and the grid."""
    _, xlabel, n_bins, integer, log_x, log_y = PANELS[key]
    edges = _shared_edges(overlay, fullsim, n_bins, integer, log_x)

    # Overlay is the sample under test, so it gets the filled GLOW-UP treatment;
    # full simulation is the reference, so it gets the dashed Target outline.
    handles, labels = [], []

    ax.hist(overlay, bins=edges, histtype="stepfilled", alpha=0.45, linewidth=1.6,
            density=True, color=GLOWUP_COLOR, edgecolor=GLOWUP_COLOR)
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=GLOWUP_COLOR,
                                 edgecolor=GLOWUP_COLOR, alpha=0.45))
    labels.append(OVERLAY_LABEL)

    ax.hist(fullsim, bins=edges, histtype="step", linestyle="--", linewidth=2.0,
            density=True, color=TARGET_COLOR)
    handles.append(plt.Line2D([], [], color=TARGET_COLOR, linestyle="--", linewidth=2.0))
    labels.append(FULLSIM_LABEL)

    if n_overlay is not None and n_fullsim is not None:
        # Own legend row with an invisible handle, the same idiom the jet panels
        # use for the PUPPI caveat line. The two counts are shown separately
        # rather than as "N each": whole shards are used and the two samples
        # have different shard sizes, so they need not come out equal.
        handles.append(plt.Line2D([], [], color="none"))
        labels.append(f"({n_overlay} vs {n_fullsim} events)"
                      if n_overlay != n_fullsim else f"({n_overlay} events each)")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(max(ylo, 1e-9), yhi)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel("Density", fontsize=axis_label_fontsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    legend = ax.legend(handles=handles, labels=labels, fontsize=legend_fontsize,
                       loc="upper right", framealpha=0.9, markerscale=1.6,
                       handlelength=2.2)
    _fit_headroom(ax, legend, log_y)


def plot_single(key: str, overlay: np.ndarray, fullsim: np.ndarray,
                n_overlay: int, n_fullsim: int):
    """Standalone single-panel figure, sized like the paper's jet panels."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Smaller legend than the jet panels: these labels are full sample names
    # rather than short method names, so they need the width.
    draw_panel(ax, key, overlay, fullsim, legend_fontsize=16.1,
               tick_labelsize=22, axis_label_fontsize=20.4,
               n_overlay=n_overlay, n_fullsim=n_fullsim)
    fig.tight_layout()
    # tight_layout changes the axes size, and with it the legend's height as a
    # fraction of the axes, so the fit is redone against the final geometry.
    _fit_headroom(ax, ax.get_legend(), PANELS[key][5])
    return fig


def plot_overview(data: dict, n_overlay: int, n_fullsim: int):
    """Every panel in one two-column grid, for a quick look rather than the paper."""
    nrows = (len(data) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 5.5 * nrows))
    flat = axes.ravel()
    for ax, (key, (overlay, fullsim)) in zip(flat, data.items()):
        draw_panel(ax, key, overlay, fullsim, legend_fontsize=10)
    for ax in flat[len(data):]:       # odd panel count leaves one empty cell
        ax.axis("off")
    fig.suptitle("Overlaid vs fully simulated PU200 ttbar "
                 f"({n_overlay} vs {n_fullsim} events)")
    fig.tight_layout()
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR,
                   help="the overlaid PU200 training sample")
    p.add_argument("--fullsim-dir", type=Path, default=DEFAULT_FULLSIM_DIR,
                   help="the directly simulated PU200 reference")
    p.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS,
                   help=f"events per sample (default: {DEFAULT_N_EVENTS})")
    p.add_argument("--out-dir", type=Path, default=Path("dataset_comparison"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--state-dir", type=Path, default=None,
                   help="per-shard cache directory (default: <out-dir>/state)")
    p.add_argument("--force", action="store_true",
                   help="re-extract every shard, ignoring the cache. Needed when "
                        "an extractor or a panel definition changes -- the cache "
                        "is keyed on the shard file alone, not on this code.")
    p.add_argument("--no-overview", dest="overview", action="store_false", default=True,
                   help="skip the 4x2 composite figure")
    args = p.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else (args.out_dir / "state")
    print(f"Target {args.n_events} events per sample -> out_dir={args.out_dir}")
    print(f"  shard cache: {state_dir}" + ("  (--force: recomputing)" if args.force else ""))
    overlay, n_overlay = collect(args.overlay_dir, args.n_events, state_dir, "overlay", args.force)
    fullsim, n_fullsim = collect(args.fullsim_dir, args.n_events, state_dir, "fullsim", args.force)

    data = {key: (overlay[key], fullsim[key]) for key in PANELS}

    for key, (a, b) in data.items():
        print(f"  {key:26s} overlay n={a.size:>9d} median={np.median(a):10.4g} | "
              f"fullsim n={b.size:>9d} median={np.median(b):10.4g}")

    for key, (a, b) in data.items():
        save_fig(plot_single(key, a, b, n_overlay, n_fullsim),
                 args.out_dir, f"overlay_vs_fullsim_{key}", args.dpi)
    if args.overview:
        save_fig(plot_overview(data, n_overlay, n_fullsim),
                 args.out_dir, "overlay_vs_fullsim_overview", args.dpi)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
