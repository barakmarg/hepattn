"""PUPPI with full charged-particle subtraction from calo clusters.

Extends ``puppi_perfect``: in addition to truth ``tracks_mask`` (LV/PU per
track) and per-cluster HS-charged subtraction, we ALSO subtract the
PU-charged calo deposits from each cluster. The α-shape PUPPI weight then
only has to discriminate HS-neutral vs PU-neutral in the residual — the
job PUPPI was actually designed for.

Operates on the all-vertices chunked dataset:

    /storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked/

which provides calo_clusters, tracks, target_particles (HS + PU with
``vertex_primary`` labels), and target_particles_deps (per-particle-
per-cluster energy attribution).

Pipeline:

    tracks (LV: w=1, PU: w=0)        clusters (e − hs_charged_e − pu_charged_e)
            │                                       │
            └───────────────┬───────────────────────┘  ← PUPPI α-shape on neutral residual
                            ▼
                       jet clustering

Truth-HS jets are clustered from particles with ``vertex_primary == 1``.

Public API:

- ``load_charged_subtract_events`` — load parquets, build per-event arrays
- ``compute_puppi_weights_charged_subtract`` — pad to 2D, subtract charged
  energy, call ``compute_puppi_weights``
- ``cluster_puppi_charged_subtract_jets`` — cluster jets from LV tracks +
  PUPPI-weighted clusters
- ``cluster_truth_hs_jets_from_events`` — HS-only particle jets for comparison
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hepattn.experiments.odd_pileup_reco.puppi import compute_puppi_weights


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _pad_to_2d(arr_list, dtype, max_len):
    n = len(arr_list)
    out = np.zeros((n, max_len), dtype=dtype)
    for i, a in enumerate(arr_list):
        if len(a):
            out[i, : len(a)] = a
    return out


def load_charged_subtract_events(
    parquet_dir: str | Path,
    *,
    event_start: int = 0,
    event_stop: int | None = None,
    event_ids: list[int] | np.ndarray | None = None,
) -> dict:
    """Load clusters + tracks + per-cluster charged-energy deps + HS particles.

    If ``event_ids`` is given, only those events are loaded, returned in the
    same order as ``event_ids`` (raises if any id is missing). When
    ``event_ids`` is None, events are returned in sorted-event-id order
    sliced by ``[event_start, event_stop)``.

    Returns a dict of per-event 1-D numpy arrays (object dtype). Node order is
    ``[tracks, clusters]`` within each event. The fields are:

    Node arrays (per-event 1-D, total length = n_tracks + n_clusters):
      ``node_pt``, ``node_eta``, ``node_phi``, ``node_e``  — float32
      ``node_is_track``  — bool (True for tracks, False for clusters)
      ``node_valid``     — bool (always True for loaded particles)
      ``tracks_mask``    — int8: 1 = LV track (HS), 0 = PU track. Clusters: 0.
      ``calo_hs_charged_e``, ``calo_pu_charged_e``  — float32, per-node;
                            zero for tracks; for clusters, the truth sum of
                            energy deposits from particles with ``has_track``
                            split by ``vertex_primary``.

    HS-particle arrays for truth-jet clustering:
      ``hs_part_pt``, ``hs_part_eta``, ``hs_part_phi`` — float32

    Extras: ``event_id`` (int64) — the source event_id in load order.
    """
    import polars as pl

    pdir = Path(parquet_dir)
    files_clu = sorted(pdir.glob("calo_clusters-*.parquet"))
    files_tr  = sorted(pdir.glob("tracks-*.parquet"))
    files_tp  = sorted(pdir.glob("target_particles-*.parquet"))
    files_dep = sorted(pdir.glob("target_particles_deps-*.parquet"))
    if not (len(files_clu) == len(files_tr) == len(files_tp) == len(files_dep)):
        raise RuntimeError(
            f"Shard count mismatch in {pdir}: clusters={len(files_clu)} "
            f"tracks={len(files_tr)} particles={len(files_tp)} deps={len(files_dep)}"
        )
    if not files_clu:
        raise FileNotFoundError(f"No calo_clusters-*.parquet in {pdir}")

    out: dict[str, list] = {
        "node_pt": [], "node_eta": [], "node_phi": [], "node_e": [],
        "node_is_track": [], "node_valid": [], "tracks_mask": [],
        "calo_hs_charged_e": [], "calo_pu_charged_e": [],
        "hs_part_pt": [], "hs_part_eta": [], "hs_part_phi": [],
        "event_id": [],
    }

    # Selection mode: a sorted set of wanted event_ids, or a slice.
    selected: set[int] | None = None
    if event_ids is not None:
        selected = set(int(x) for x in np.asarray(event_ids).ravel().tolist())

    # Pass 1: read each shard, build event_id -> per-shard rows for the events we want.
    # For event_ids mode we keep a global dict; otherwise we stream.
    if selected is not None:
        rows_by_eid: dict[int, tuple[dict, dict, dict, dict]] = {}
        for fc, ft, fp, fd in zip(files_clu, files_tr, files_tp, files_dep):
            df_c = pl.read_parquet(fc)
            if not df_c["event_id"].is_in(list(selected)).any():
                continue
            df_t = pl.read_parquet(ft)
            df_p = pl.read_parquet(fp)
            df_d = pl.read_parquet(fd)
            for row_c, row_t, row_p, row_d in zip(
                df_c.iter_rows(named=True), df_t.iter_rows(named=True),
                df_p.iter_rows(named=True), df_d.iter_rows(named=True),
            ):
                eid = int(row_c["event_id"])
                if eid in selected:
                    rows_by_eid[eid] = (row_c, row_t, row_p, row_d)

        missing = [int(e) for e in np.asarray(event_ids).ravel().tolist()
                   if int(e) not in rows_by_eid]
        if missing:
            raise KeyError(f"Missing {len(missing)} requested event_ids in parquets, "
                           f"first few: {missing[:5]}")
        iterator = (rows_by_eid[int(e)] for e in np.asarray(event_ids).ravel().tolist())
        total = len(event_ids)
    else:
        def _stream():
            n_skipped = 0
            n_yielded = 0
            target = event_stop - event_start if event_stop is not None else None
            for fc, ft, fp, fd in zip(files_clu, files_tr, files_tp, files_dep):
                if target is not None and n_yielded >= target:
                    return
                df_c = pl.read_parquet(fc).sort("event_id")
                df_t = pl.read_parquet(ft).sort("event_id")
                df_p = pl.read_parquet(fp).sort("event_id")
                df_d = pl.read_parquet(fd).sort("event_id")
                if not (df_c.shape[0] == df_t.shape[0] == df_p.shape[0] == df_d.shape[0]):
                    raise RuntimeError(f"Row count mismatch in shard {fc.name}")
                for row_c, row_t, row_p, row_d in zip(
                    df_c.iter_rows(named=True), df_t.iter_rows(named=True),
                    df_p.iter_rows(named=True), df_d.iter_rows(named=True),
                ):
                    if n_skipped < event_start:
                        n_skipped += 1
                        continue
                    if target is not None and n_yielded >= target:
                        return
                    yield (row_c, row_t, row_p, row_d)
                    n_yielded += 1
        iterator = _stream()
        total = None

    for rows in iterator:
        row_c, row_t, row_p, row_d = rows
        assert row_c["event_id"] == row_t["event_id"] == \
               row_p["event_id"] == row_d["event_id"], \
               "event_id misalignment across parquets"

        # ---- clusters ----
        cl_e   = np.asarray(row_c["total_cluster_energy"], dtype=np.float32)
        cl_eta = np.asarray(row_c["cluster_eta"], dtype=np.float32)
        cl_phi = np.asarray(row_c["cluster_phi"], dtype=np.float32)
        n_cl = cl_e.shape[0]
        cl_pt  = cl_e / np.cosh(np.clip(cl_eta, -10.0, 10.0))

        # ---- tracks ----
        tr_pt  = np.asarray(row_t["pt"],  dtype=np.float32)
        tr_eta = np.asarray(row_t["eta"], dtype=np.float32)
        tr_phi = np.asarray(row_t["phi"], dtype=np.float32)
        # Some tracks lack vertex info (None). Replace with -1 so
        # ``tr_vp == 1`` evaluates to False for them (non-LV → PU).
        tr_vp_raw = row_t["vertex_primary"]
        tr_vp = np.fromiter(
            (-1 if v is None else int(v) for v in tr_vp_raw),
            dtype=np.int64, count=len(tr_vp_raw),
        )
        n_tr = tr_pt.shape[0]
        tr_e = tr_pt * np.cosh(np.clip(tr_eta, -10.0, 10.0))  # massless proxy

        # ---- per-cluster charged-deps aggregation ----
        p_vp = np.asarray(row_p["vertex_primary"], dtype=np.int64)
        p_ht = np.asarray(row_p["has_track"], dtype=bool)
        deps_p = np.asarray(row_d["particle_idx"], dtype=np.int64)
        deps_c = np.asarray(row_d["cluster_idx"],  dtype=np.int64)
        deps_e = np.asarray(row_d["total_energy_deps_in_cluster"], dtype=np.float32)

        vp_per_dep = p_vp[deps_p]
        ht_per_dep = p_ht[deps_p]
        hs_ch = ht_per_dep & (vp_per_dep == 1)
        pu_ch = ht_per_dep & (vp_per_dep != 1)

        cluster_hs_e = np.zeros(n_cl, dtype=np.float32)
        cluster_pu_e = np.zeros(n_cl, dtype=np.float32)
        if hs_ch.any():
            np.add.at(cluster_hs_e, deps_c[hs_ch], deps_e[hs_ch])
        if pu_ch.any():
            np.add.at(cluster_pu_e, deps_c[pu_ch], deps_e[pu_ch])

        # ---- combined node arrays [tracks then clusters] ----
        node_pt   = np.concatenate([tr_pt,  cl_pt])
        node_eta  = np.concatenate([tr_eta, cl_eta])
        node_phi  = np.concatenate([tr_phi, cl_phi])
        node_e    = np.concatenate([tr_e,   cl_e])
        node_is_t = np.concatenate([np.ones(n_tr, dtype=bool),
                                    np.zeros(n_cl, dtype=bool)])
        node_vld  = np.ones(n_tr + n_cl, dtype=bool)
        # tracks_mask: 1 for LV (vertex_primary==1), 0 otherwise. Clusters: 0.
        tracks_msk = np.concatenate([
            (tr_vp == 1).astype(np.int8),
            np.zeros(n_cl, dtype=np.int8),
        ])
        calo_hs   = np.concatenate([np.zeros(n_tr, dtype=np.float32), cluster_hs_e])
        calo_pu   = np.concatenate([np.zeros(n_tr, dtype=np.float32), cluster_pu_e])

        # ---- HS truth particles (vertex_primary==1) for truth-jet clustering ----
        hs_mask = (p_vp == 1)
        hs_pt  = np.asarray(row_p["pt"],  dtype=np.float32)[hs_mask]
        hs_eta = np.asarray(row_p["eta"], dtype=np.float32)[hs_mask]
        hs_phi = np.asarray(row_p["phi"], dtype=np.float32)[hs_mask]

        out["node_pt"].append(node_pt)
        out["node_eta"].append(node_eta)
        out["node_phi"].append(node_phi)
        out["node_e"].append(node_e)
        out["node_is_track"].append(node_is_t)
        out["node_valid"].append(node_vld)
        out["tracks_mask"].append(tracks_msk)
        out["calo_hs_charged_e"].append(calo_hs)
        out["calo_pu_charged_e"].append(calo_pu)
        out["hs_part_pt"].append(hs_pt)
        out["hs_part_eta"].append(hs_eta)
        out["hs_part_phi"].append(hs_phi)
        out["event_id"].append(int(row_c["event_id"]))

    return {k: np.array(v, dtype=object if k != "event_id" else np.int64)
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# PUPPI weights wrapper
# ---------------------------------------------------------------------------

def compute_puppi_weights_charged_subtract(
    events: dict,
    *,
    subtract_pu_charged: bool = True,
    R0: float = 0.139,
    rms_pt_min: float = 0.080,
    min_neutral_pt: float = 0.504,
    min_neutral_pt_slope: float = 1.837,
    min_weight: float = 0.051,
    eta_max_extrap: float = 2.543,
    apply_lv_adjust: bool = False,
    n_pu_proxy=None,
) -> np.ndarray:
    """Compute per-node PUPPI weights with HS+PU charged calo subtraction.

    Pads the per-event arrays to 2D ``(n_events, max_nodes)`` and calls
    ``compute_puppi_weights``. Cluster ``node_e`` is overridden to the
    neutral-only residual ``max(e − hs_c − pu_c, 0)`` (or only ``e − hs_c``
    if ``subtract_pu_charged=False``, which reproduces ``puppi_perfect``).

    Returns a 2-D array of weights aligned to the padded node grid; the
    matching ``cluster_puppi_charged_subtract_jets`` slices to per-event
    length when clustering.
    """
    n_events = len(events["node_pt"])
    max_n = max((len(e) for e in events["node_pt"]), default=0)

    f32_keys = ("node_pt", "node_eta", "node_phi", "node_e",
                "calo_hs_charged_e", "calo_pu_charged_e")
    bool_keys = ("node_is_track", "node_valid")
    i8_keys = ("tracks_mask",)

    data = {}
    for k in f32_keys:
        data[k] = _pad_to_2d(events[k], np.float32, max_n)
    for k in bool_keys:
        data[k] = _pad_to_2d(events[k], bool, max_n)
    for k in i8_keys:
        data[k] = _pad_to_2d(events[k], np.int8, max_n)

    # Override cluster node_e with neutral residual.
    is_track = data["node_is_track"]
    e = data["node_e"]
    hs_c = data["calo_hs_charged_e"]
    pu_c = data["calo_pu_charged_e"] if subtract_pu_charged else np.zeros_like(hs_c)
    residual_e = np.maximum(e - hs_c - pu_c, 0.0)
    data["node_e"] = np.where(is_track, e, residual_e).astype(np.float32)

    return compute_puppi_weights(
        data,
        R0=R0,
        rms_pt_min=rms_pt_min,
        min_neutral_pt=min_neutral_pt,
        min_neutral_pt_slope=min_neutral_pt_slope,
        min_weight=min_weight,
        eta_max_extrap=eta_max_extrap,
        apply_lv_adjust=apply_lv_adjust,
        n_pu_proxy=n_pu_proxy,
    )


# ---------------------------------------------------------------------------
# Jet clustering
# ---------------------------------------------------------------------------

def cluster_puppi_charged_subtract_jets(
    events: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
    *,
    weights: np.ndarray | None = None,
    subtract_pu_charged: bool = True,
    **puppi_kwargs,
) -> dict:
    """Cluster jets from LV tracks (weight=1, full pT) + clusters (PUPPI-weighted
    neutral-residual pT). PU tracks fall out automatically (weight=0).

    Returns a dict keyed ``puppi_jet_*``.
    """
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    if weights is None:
        weights = compute_puppi_weights_charged_subtract(
            events, subtract_pu_charged=subtract_pu_charged, **puppi_kwargs,
        )

    n_events = len(events["node_pt"])
    pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
    try:
        from tqdm import tqdm
        it = tqdm(range(n_events), desc="Jets (puppi-charged-sub)")
    except ImportError:
        it = range(n_events)

    for i in it:
        is_t = events["node_is_track"][i].astype(bool)
        eta  = events["node_eta"][i].astype(np.float32)
        phi  = events["node_phi"][i].astype(np.float32)
        e    = events["node_e"][i].astype(np.float32)
        hs_c = events["calo_hs_charged_e"][i].astype(np.float32)
        pu_c = events["calo_pu_charged_e"][i].astype(np.float32) \
               if subtract_pu_charged else np.zeros_like(hs_c)
        residual_e = np.maximum(e - hs_c - pu_c, 0.0)
        cluster_pt = residual_e / np.cosh(np.clip(eta, -10.0, 10.0))
        track_pt = events["node_pt"][i].astype(np.float32)
        node_pt = np.where(is_t, track_pt, cluster_pt)

        w = np.asarray(weights[i, : len(is_t)], dtype=np.float32)
        weighted_pt = w * node_pt
        valid = events["node_valid"][i].astype(bool)
        sel = (valid & np.isfinite(weighted_pt) & np.isfinite(eta)
               & np.isfinite(phi) & (weighted_pt > 0))
        ptetaphi = np.stack([weighted_pt, eta, phi], axis=-1)
        out = _cluster_jets_single(ptetaphi, sel, 0.5, jet_R, min_constituents, min_pt)
        pt_l.append(out[0]); eta_l.append(out[1]); phi_l.append(out[2])
        m_l.append(out[3]);  nc_l.append(out[4])

    return {
        "puppi_jet_pt":     np.array(pt_l, dtype=object),
        "puppi_jet_eta":    np.array(eta_l, dtype=object),
        "puppi_jet_phi":    np.array(phi_l, dtype=object),
        "puppi_jet_mass":   np.array(m_l, dtype=object),
        "puppi_jet_nconst": np.array(nc_l, dtype=object),
    }


def cluster_truth_hs_jets_from_events(
    events: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
) -> dict:
    """Cluster jets from HS truth particles (vertex_primary==1) stored in events."""
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    n_events = len(events["hs_part_pt"])
    pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
    try:
        from tqdm import tqdm
        it = tqdm(range(n_events), desc="Jets (truth-hs)")
    except ImportError:
        it = range(n_events)
    for i in it:
        pt = events["hs_part_pt"][i].astype(np.float32)
        eta = events["hs_part_eta"][i].astype(np.float32)
        phi = events["hs_part_phi"][i].astype(np.float32)
        sel = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & (pt > 0)
        ptetaphi = np.stack([pt, eta, phi], axis=-1)
        out = _cluster_jets_single(ptetaphi, sel, 0.5, jet_R, min_constituents, min_pt)
        pt_l.append(out[0]); eta_l.append(out[1]); phi_l.append(out[2])
        m_l.append(out[3]);  nc_l.append(out[4])

    return {
        "truth_jet_pt":     np.array(pt_l, dtype=object),
        "truth_jet_eta":    np.array(eta_l, dtype=object),
        "truth_jet_phi":    np.array(phi_l, dtype=object),
        "truth_jet_mass":   np.array(m_l, dtype=object),
        "truth_jet_nconst": np.array(nc_l, dtype=object),
    }
