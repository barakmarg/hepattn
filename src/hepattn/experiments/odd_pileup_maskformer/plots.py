"""
Physics validation plots for pileup removal.

All PhysicsPlotter methods accept plain numpy arrays and return matplotlib Figure objects.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class PhysicsPlotter:
    """Static helper class to generate physics validation figures."""

    @staticmethod
    def plot_energy_correlation(pred_frac: np.ndarray, total_e: np.ndarray, true_hs_e: np.ndarray) -> Figure:
        """2D histogram: predicted HS energy vs truth HS energy. Tests linearity and bias."""
        pred_hs_e = pred_frac * total_e
        emax = max(np.percentile(true_hs_e, 99), np.percentile(pred_hs_e, 99), 1.0)
        fig, ax = plt.subplots(figsize=(7, 6))
        h = ax.hist2d(true_hs_e, pred_hs_e, bins=100, range=[[0, emax], [0, emax]], norm="log", cmap="viridis")
        ax.plot([0, emax], [0, emax], "r--", lw=1, label="Ideal")
        ax.set_xlabel("True Hard Scatter Energy [GeV]")
        ax.set_ylabel("Predicted Hard Scatter Energy [GeV]")
        ax.set_title("Calo: Truth vs Predicted HS Energy")
        plt.colorbar(h[3], ax=ax, label="Count (log)")
        ax.legend()
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_frac_correlation(pred_frac: np.ndarray, true_frac: np.ndarray) -> Figure:
        """2D histogram: predicted HS fraction vs truth HS fraction."""
        fig, ax = plt.subplots(figsize=(7, 6))
        h = ax.hist2d(true_frac, pred_frac, bins=100, range=[[0, 1], [0, 1]], norm="log", cmap="plasma")
        ax.plot([0, 1], [0, 1], "r--", lw=1, label="Ideal")
        ax.set_xlabel("True HS Fraction")
        ax.set_ylabel("Predicted HS Fraction")
        ax.set_title("Calo: Truth vs Predicted HS Fraction")
        plt.colorbar(h[3], ax=ax, label="Count (log)")
        ax.legend()
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_frac_distribution(pred_frac: np.ndarray, true_frac: np.ndarray) -> Figure:
        """Overlaid 1D histograms of predicted and truth HS fraction in [0, 1]."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(true_frac, bins=100, range=(0, 1), density=True,
                alpha=0.7, color="steelblue", label="Truth")
        ax.hist(pred_frac, bins=100, range=(0, 1), density=True,
                alpha=0.7, color="tomato", label="Predicted")
        ax.set_xlabel("HS Fraction")
        ax.set_ylabel("Normalised Density")
        ax.set_title("Calo: HS Fraction Distribution")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_energy_residual(pred_frac: np.ndarray, total_e: np.ndarray, true_hs_e: np.ndarray) -> Figure:
        """(E_pred − E_true) / E_true distribution. Tests calibration."""
        pred_hs_e = pred_frac * total_e
        mask = true_hs_e > 0.1
        if mask.sum() == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No clusters with E > 0.1 GeV", ha="center", va="center")
            return fig
        residual = (pred_hs_e[mask] - true_hs_e[mask]) / (true_hs_e[mask] + 1e-3)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(np.clip(residual, -3, 3), bins=100, color="steelblue", edgecolor="none")
        ax.axvline(0, color="red", lw=1, ls="--", label="Ideal")
        ax.set_xlabel("(E_pred − E_true) / E_true")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_title("Calo: HS Energy Residual")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_hs_energy_distribution(pred_frac: np.ndarray, total_e: np.ndarray, true_hs_e: np.ndarray) -> Figure:
        """Overlaid 1D histograms of predicted vs truth HS energy (log-log scale)."""
        pred_hs_e = pred_frac * total_e
        # Filter to positive energies for log scale
        mask_true = true_hs_e > 1e-3
        mask_pred = pred_hs_e > 1e-3
        if mask_true.sum() == 0 and mask_pred.sum() == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No clusters with E > 0", ha="center", va="center")
            return fig
        emax = max(
            np.percentile(true_hs_e[mask_true], 99.5) if mask_true.any() else 1.0,
            np.percentile(pred_hs_e[mask_pred], 99.5) if mask_pred.any() else 1.0,
            1.0,
        )
        emin = 1e-2
        bins = np.logspace(np.log10(emin), np.log10(emax), 80)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(true_hs_e[mask_true], bins=bins, density=True,
                alpha=0.7, color="steelblue", label="Truth HS Energy")
        ax.hist(pred_hs_e[mask_pred], bins=bins, density=True,
                alpha=0.7, color="tomato", label="Predicted HS Energy")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Hard Scatter Energy [GeV]")
        ax.set_ylabel("Normalised Density")
        ax.set_title("Calo: Predicted vs Truth HS Energy Distribution")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_errors_by_energy(
        mask_pred: np.ndarray, mask_truth: np.ndarray,
        total_e: np.ndarray, true_hs_e: np.ndarray,
        pred_frac: np.ndarray,
    ) -> dict:
        """FP/FN HS-energy histograms binned by total cluster energy.

        Returns a dict of {bin_label: Figure}, one figure per energy bin.
        FP (blue): predicted HS energy (pred_frac * total_e) for clusters that are actually PU.
        FN (red):  true HS energy for clusters that are actually HS but predicted as PU.
        """
        mask_pred = mask_pred.astype(bool)
        mask_truth = mask_truth.astype(bool)
        pred_hs_e = pred_frac * total_e
        e_bins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]

        figs = {}
        for lo, hi in zip(e_bins[:-1], e_bins[1:]):
            in_bin = (total_e >= lo) & (total_e < hi)
            fp = in_bin & mask_pred & ~mask_truth
            fn = in_bin & ~mask_pred & mask_truth

            all_vals = np.concatenate([
                pred_hs_e[fp] if fp.any() else np.array([]),
                true_hs_e[fn] if fn.any() else np.array([]),
            ])
            e_max = max(all_vals.max() if len(all_vals) > 0 else 1.0, 1e-3)
            bins = np.linspace(0, e_max, 40)

            fig, ax = plt.subplots(figsize=(7, 5))
            if fp.any():
                ax.hist(pred_hs_e[fp], bins=bins, alpha=0.7, color="steelblue",
                        label=f"FP — pred HS energy (n={fp.sum()})")
            if fn.any():
                ax.hist(true_hs_e[fn], bins=bins, alpha=0.7, color="tomato",
                        label=f"FN — true HS energy (n={fn.sum()})")

            hi_label = f"{hi:.0f}" if not np.isinf(hi) else "∞"
            ax.set_title(f"Calo Mask Errors: E_total ∈ [{lo:.1f}, {hi_label}) GeV\n"
                         f"FP = predicted HS energy  |  FN = true HS energy")
            ax.set_xlabel("HS Energy [GeV]  (FP: predicted · FN: true)")
            ax.set_ylabel("Count")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)
            plt.tight_layout()

            key = f"{lo:.1f}_{hi_label}"
            figs[key] = fig

        return figs

    @staticmethod
    def plot_calo_mistag_vs_eta(
        mask_pred: np.ndarray, mask_truth: np.ndarray, eta: np.ndarray,
    ) -> Figure:
        """Pileup mistag rate (FPR) vs cluster η for calo clusters."""
        mask_pred = mask_pred.astype(bool)
        mask_truth = mask_truth.astype(bool)
        is_true_pu = ~mask_truth

        bins = np.linspace(-5, 5, 25)
        centers = (bins[:-1] + bins[1:]) / 2
        mistag, err = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (eta >= lo) & (eta < hi) & is_true_pu
            n = m.sum()
            if n > 0:
                fpr = mask_pred[m].sum() / n
                mistag.append(fpr)
                err.append(np.sqrt(fpr * (1 - fpr) / n))
            else:
                mistag.append(np.nan)
                err.append(np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(centers, mistag, yerr=err, fmt="x-", capsize=3, color="tomato")
        ax.set_xlabel("Cluster η")
        ax.set_ylabel("Pileup Mistag Rate (FPR)")
        ax.set_title("Calo Pileup Mistag Rate vs η")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_event_hs_energy_ratio(
        pred_frac: np.ndarray, total_e: np.ndarray,
        true_hs_e: np.ndarray, event_idx: np.ndarray,
    ) -> Figure:
        """Histogram of per-event (sum predicted HS energy / sum true HS energy).

        A perfect model gives a distribution centred at 1.
        Values < 1 mean the model under-predicts total HS energy per event;
        values > 1 mean over-prediction.
        """
        pred_hs_e = pred_frac * total_e
        # Remap event_idx to 0..N-1 so bincount works without gaps
        _, inv = np.unique(event_idx, return_inverse=True)
        sum_pred = np.bincount(inv, weights=pred_hs_e)
        sum_true = np.bincount(inv, weights=true_hs_e)
        valid = sum_true > 0
        ratios = sum_pred[valid] / sum_true[valid]

        fig, ax = plt.subplots(figsize=(7, 5))
        if len(ratios) == 0:
            ax.text(0.5, 0.5, "No events with true HS energy > 0", ha="center", va="center")
            plt.tight_layout()
            return fig

        clipped = np.clip(ratios, 0, 3)
        ax.hist(clipped, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
        ax.axvline(1.0, color="red", lw=1.5, ls="--", label="Ideal (1.0)")
        mean, std = ratios.mean(), ratios.std()
        ax.set_xlabel("Σ Predicted HS Energy / Σ True HS Energy  (per event)")
        ax.set_ylabel("Events")
        ax.set_yscale("log")
        ax.set_title("Per-Event HS Energy Ratio")
        ax.legend(title=f"mean={mean:.3f}, std={std:.3f}\nN events={len(ratios)}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_neutral_energy_frac(neutral_e: np.ndarray, total_e: np.ndarray) -> Figure:
        """Histogram of neutral HS energy fraction (%) per cluster (truth only)."""
        total_safe = np.where(total_e > 0, total_e, 1.0)
        neutral_pct = neutral_e / total_safe * 100.0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(neutral_pct, bins=100, range=(0, 100), color="steelblue", alpha=0.8, edgecolor="none")
        mean_n = neutral_pct[total_e > 0].mean() if (total_e > 0).any() else 0.0
        ax.axvline(mean_n, color="red", lw=1.5, ls="--", label=f"mean={mean_n:.1f}%")
        ax.set_xlabel("Neutral HS Energy / Total Cluster Energy  [%]")
        ax.set_ylabel("Clusters")
        ax.set_yscale("log")
        ax.set_title("Neutral HS Energy Fraction per Cluster  (Truth)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_charged_energy_frac(charged_e: np.ndarray, total_e: np.ndarray) -> Figure:
        """Histogram of charged HS energy fraction (%) per cluster (truth only)."""
        total_safe = np.where(total_e > 0, total_e, 1.0)
        charged_pct = charged_e / total_safe * 100.0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(charged_pct, bins=100, range=(0, 100), color="darkorange", alpha=0.8, edgecolor="none")
        mean_c = charged_pct[total_e > 0].mean() if (total_e > 0).any() else 0.0
        ax.axvline(mean_c, color="red", lw=1.5, ls="--", label=f"mean={mean_c:.1f}%")
        ax.set_xlabel("Charged HS Energy / Total Cluster Energy  [%]")
        ax.set_ylabel("Clusters")
        ax.set_yscale("log")
        ax.set_title("Charged HS Energy Fraction per Cluster  (Truth)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_cluster_energy_dist(total_e: np.ndarray) -> Figure:
        """Histogram of total calo cluster energy (truth only)."""
        valid = total_e[total_e > 0]
        fig, ax = plt.subplots(figsize=(7, 5))
        if len(valid) > 0:
            lo = np.log10(valid.min())
            hi = np.log10(np.percentile(valid, 99.5))
            bins = np.logspace(lo, hi, 80)
            ax.hist(valid, bins=bins, color="seagreen", alpha=0.8, edgecolor="none")
            ax.set_xscale("log")
        ax.set_xlabel("Total Cluster Energy  [GeV]")
        ax.set_ylabel("Clusters")
        ax.set_yscale("log")
        ax.set_title("Total Calo Cluster Energy Distribution  (Truth)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_hs_energy_residual_by_type(
        pred_neutral: np.ndarray, truth_neutral: np.ndarray,
        pred_charged: np.ndarray, truth_charged: np.ndarray,
    ) -> Figure:
        """Histogram of per-event energy residual (pred - truth), split by neutral/charged.

        Inputs are per-event sums (already aggregated over clusters).
        """
        resid_neutral = pred_neutral - truth_neutral
        resid_charged = pred_charged - truth_charged

        fig, ax = plt.subplots(figsize=(7, 5))
        if len(resid_neutral) == 0:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
            plt.tight_layout()
            return fig

        lo = min(np.percentile(resid_neutral, 1), np.percentile(resid_charged, 1))
        hi = max(np.percentile(resid_neutral, 99), np.percentile(resid_charged, 99))
        bins = np.linspace(lo, hi, 60)

        ax.hist(resid_neutral, bins=bins, alpha=0.7, color="steelblue", label="Neutral (trackless)", edgecolor="none")
        ax.hist(resid_charged, bins=bins, alpha=0.7, color="darkorange", label="Charged (tracked)", edgecolor="none")
        ax.axvline(0.0, color="red", lw=1.5, ls="--", label="Ideal (0)")

        ax.set_xlabel("Σ Predicted Mask HS Energy − Σ Truth Mask HS Energy  [GeV]")
        ax.set_ylabel("Events")
        ax.set_yscale("log")
        ax.set_title("Per-Event HS Energy Residual by Particle Type")
        ax.legend(title=(
            f"Neutral: μ={resid_neutral.mean():.2f}, σ={resid_neutral.std():.2f}\n"
            f"Charged: μ={resid_charged.mean():.2f}, σ={resid_charged.std():.2f}\n"
            f"N={len(resid_neutral)}"
        ))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_hs_energy_ratio_by_type(
        pred_neutral: np.ndarray, truth_neutral: np.ndarray,
        pred_charged: np.ndarray, truth_charged: np.ndarray,
    ) -> Figure:
        """Histogram of per-event energy ratio (pred / truth), split by neutral/charged.

        Inputs are per-event sums. Events with truth sum == 0 are excluded.
        """
        valid_n = truth_neutral > 0
        valid_c = truth_charged > 0
        ratio_neutral = pred_neutral[valid_n] / truth_neutral[valid_n]
        ratio_charged = pred_charged[valid_c] / truth_charged[valid_c]

        fig, ax = plt.subplots(figsize=(7, 5))
        if len(ratio_neutral) == 0 and len(ratio_charged) == 0:
            ax.text(0.5, 0.5, "No events with truth energy > 0", ha="center", va="center")
            plt.tight_layout()
            return fig

        clip_n = np.clip(ratio_neutral, 0, 3) if len(ratio_neutral) > 0 else np.array([])
        clip_c = np.clip(ratio_charged, 0, 3) if len(ratio_charged) > 0 else np.array([])
        bins = np.linspace(0, 3, 60)

        if len(clip_n) > 0:
            ax.hist(clip_n, bins=bins, alpha=0.7, color="steelblue", label="Neutral (trackless)", edgecolor="none")
        if len(clip_c) > 0:
            ax.hist(clip_c, bins=bins, alpha=0.7, color="darkorange", label="Charged (tracked)", edgecolor="none")
        ax.axvline(1.0, color="red", lw=1.5, ls="--", label="Ideal (1.0)")

        ax.set_xlabel("Σ Predicted Mask HS Energy / Σ Truth Mask HS Energy  (per event)")
        ax.set_ylabel("Events")
        ax.set_yscale("log")
        ax.set_title("Per-Event HS Energy Ratio by Particle Type")

        legend_parts = []
        if len(ratio_neutral) > 0:
            legend_parts.append(f"Neutral: μ={ratio_neutral.mean():.3f}, σ={ratio_neutral.std():.3f}")
        if len(ratio_charged) > 0:
            legend_parts.append(f"Charged: μ={ratio_charged.mean():.3f}, σ={ratio_charged.std():.3f}")
        legend_parts.append(f"N neutral={len(ratio_neutral)}, N charged={len(ratio_charged)}")
        ax.legend(title="\n".join(legend_parts))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_score_distribution(probs: np.ndarray, truth: np.ndarray) -> Figure:
        """1D score histogram: HS tracks vs PU tracks overlaid."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(probs[truth == 1], bins=50, range=(0, 1), density=True,
                alpha=0.7, color="steelblue", label="Hard Scatter")
        ax.hist(probs[truth == 0], bins=50, range=(0, 1), density=True,
                alpha=0.7, color="tomato", label="Pileup")
        ax.set_xlabel("Model Score (0=Pileup, 1=HS)")
        ax.set_ylabel("Normalised Density")
        ax.set_title("Track Score Distribution")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_efficiency_vs_pt(probs: np.ndarray, truth: np.ndarray, pt: np.ndarray,
                                    threshold: float = 0.5) -> Figure:
        """Signal recall (efficiency) vs log-pT with Poisson error bars."""
        bins = np.logspace(np.log10(0.3), np.log10(200), 25)
        centers = np.sqrt(bins[:-1] * bins[1:])
        is_pred_hs = probs > threshold
        is_true_hs = truth == 1
        eff, err = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pt >= lo) & (pt < hi) & is_true_hs
            n = m.sum()
            if n > 0:
                e = is_pred_hs[m].sum() / n
                eff.append(e)
                err.append(np.sqrt(e * (1 - e) / n))
            else:
                eff.append(np.nan)
                err.append(np.nan)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(centers, eff, yerr=err, fmt="o-", capsize=3, color="steelblue")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Track pT [GeV]")
        ax.set_ylabel("Signal Efficiency (Recall)")
        ax.set_ylim(0, 1.1)
        ax.set_title("Track Signal Efficiency vs pT")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_f1_vs_pt(probs: np.ndarray, truth: np.ndarray, pt: np.ndarray,
                            threshold: float = 0.5) -> Figure:
        """Binary F1 score vs log-pT with error bars."""
        bins = np.logspace(np.log10(0.3), np.log10(200), 25)
        centers = np.sqrt(bins[:-1] * bins[1:])
        is_pred_hs = probs > threshold
        is_true_hs = truth == 1
        f1_vals, f1_errs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pt >= lo) & (pt < hi)
            n = m.sum()
            if n > 0:
                tp = (is_pred_hs[m] & is_true_hs[m]).sum()
                fp = (is_pred_hs[m] & ~is_true_hs[m]).sum()
                fn = (~is_pred_hs[m] & is_true_hs[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_vals.append(f1)
                f1_errs.append(np.sqrt(f1 * (1 - f1) / n))
            else:
                f1_vals.append(np.nan)
                f1_errs.append(np.nan)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(centers, f1_vals, yerr=f1_errs, fmt="o-", capsize=3, color="steelblue")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Track pT [GeV]")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Track F1 vs pT")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_pileup_rejection_vs_pt(probs: np.ndarray, truth: np.ndarray, pt: np.ndarray,
                                    threshold: float = 0.5) -> Figure:
        """Pileup rejection (1 − FPR) vs log-pT."""
        bins = np.logspace(np.log10(0.3), np.log10(200), 25)
        centers = np.sqrt(bins[:-1] * bins[1:])
        is_pred_hs = probs > threshold
        is_true_pu = truth == 0
        rej, err = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pt >= lo) & (pt < hi) & is_true_pu
            n = m.sum()
            if n > 0:
                fpr = is_pred_hs[m].sum() / n
                rej.append(1 - fpr)
                err.append(np.sqrt(fpr * (1 - fpr) / n))
            else:
                rej.append(np.nan)
                err.append(np.nan)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(centers, rej, yerr=err, fmt="o-", capsize=3, color="tomato")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Track pT [GeV]")
        ax.set_ylabel("Pileup Rejection (1 − FPR)")
        ax.set_ylim(0, 1.1)
        ax.set_title("Pileup Rejection vs pT")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_mistag_rate_vs_eta(probs: np.ndarray, truth: np.ndarray, eta: np.ndarray,
                                threshold: float = 0.5) -> Figure:
        """Pileup mistag rate (FPR) vs η. Checks forward-region degradation."""
        bins = np.linspace(-4, 4, 25)
        centers = (bins[:-1] + bins[1:]) / 2
        is_pred_hs = probs > threshold
        is_true_pu = truth == 0
        mistag, err = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (eta >= lo) & (eta < hi) & is_true_pu
            n = m.sum()
            if n > 0:
                fpr = is_pred_hs[m].sum() / n
                mistag.append(fpr)
                err.append(np.sqrt(fpr * (1 - fpr) / n))
            else:
                mistag.append(np.nan)
                err.append(np.nan)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(centers, mistag, yerr=err, fmt="x-", capsize=3, color="tomato")
        ax.set_xlabel("Track η")
        ax.set_ylabel("Pileup Mistag Rate (FPR)")
        ax.set_title("Pileup Mistag Rate vs η")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_score_by_pt(probs: np.ndarray, truth: np.ndarray, pt: np.ndarray) -> Figure:
        """Normalised score histograms for HS tracks, stacked by pT bin."""
        pt_bins = [0.0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 1000.0]
        colors = cm.plasma(np.linspace(0, 0.9, len(pt_bins) - 1))
        is_true_hs = truth == 1
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (lo, hi) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
            label = f"> {lo} GeV" if hi == 1000.0 else f"{lo}–{hi} GeV"
            m = (pt >= lo) & (pt < hi) & is_true_hs
            if m.sum() > 5:
                ax.hist(probs[m], bins=50, range=(0, 1), density=True,
                        histtype="step", lw=2, color=colors[i], label=label)
        ax.set_xlabel("Model Score (0=Pileup, 1=HS)")
        ax.set_ylabel("Normalised Density")
        ax.set_title("HS Track Score Distribution by pT")
        ax.set_yscale("log")
        ax.legend(title="Track pT", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_z0_distribution(probs: np.ndarray, truth: np.ndarray, z0: np.ndarray,
                                   threshold: float = 0.5) -> Figure:
        """Overlaid z0 histograms: predicted HS tracks vs truth HS tracks."""
        is_true_hs = truth == 1
        is_pred_hs = probs > threshold
        z0_range = np.percentile(np.abs(z0[is_true_hs | is_pred_hs] if (is_true_hs | is_pred_hs).any() else z0), 99)
        z0_range = max(z0_range, 1.0)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(z0[is_true_hs], bins=100, range=(-z0_range, z0_range), density=True,
                alpha=0.7, color="steelblue", label="Truth HS Tracks")
        ax.hist(z0[is_pred_hs], bins=100, range=(-z0_range, z0_range), density=True,
                alpha=0.7, color="tomato", label="Predicted HS Tracks (score > 0.5)")
        ax.set_xlabel("Track z0 [mm]")
        ax.set_ylabel("Normalised Density")
        ax.set_title("z0 Distribution: Truth vs Predicted HS Tracks")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_pt_distribution(probs: np.ndarray, truth: np.ndarray, pt: np.ndarray,
                                   threshold: float = 0.5) -> Figure:
        """Overlaid pT histograms: predicted HS tracks vs truth HS tracks (log scale)."""
        is_true_hs = truth == 1
        is_pred_hs = probs > threshold
        pt_max = max(np.percentile(pt[is_true_hs | is_pred_hs] if (is_true_hs | is_pred_hs).any() else pt, 99), 1.0)
        bins = np.logspace(np.log10(0.3), np.log10(pt_max), 60)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(pt[is_true_hs], bins=bins, density=True,
                alpha=0.7, color="steelblue", label="Truth HS Tracks")
        ax.hist(pt[is_pred_hs], bins=bins, density=True,
                alpha=0.7, color="tomato", label="Predicted HS Tracks (score > 0.5)")
        ax.set_xscale("log")
        ax.set_xlabel("Track pT [GeV]")
        ax.set_ylabel("Normalised Density")
        ax.set_title("pT Distribution: Truth vs Predicted HS Tracks")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_track_f1_vs_threshold(probs: np.ndarray, truth: np.ndarray, pt: np.ndarray) -> Figure:
        """Track F1 vs classification threshold, one line per pT bin [0,1,2,5,10,20,∞] GeV."""
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
        bin_edges = [0, 1, 2, 5, 10, 20, np.inf]
        bin_labels = ["0–1 GeV", "1–2 GeV", "2–5 GeV", "5–10 GeV", "10–20 GeV", "> 20 GeV"]
        colors = cm.plasma(np.linspace(0, 0.88, len(bin_labels)))
        is_true_hs = truth == 1

        fig, ax = plt.subplots(figsize=(8, 5))
        for (lo, hi), label, color in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_labels, colors):
            m = (pt >= lo) & (pt < hi)
            if m.sum() < 5:
                continue
            f1_vals = []
            for thr in thresholds:
                is_pred = probs[m] > thr
                tp = (is_pred & is_true_hs[m]).sum()
                fp = (is_pred & ~is_true_hs[m]).sum()
                fn = (~is_pred & is_true_hs[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_vals.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            ax.plot(thresholds, f1_vals, "o-", color=color, label=label)

        ax.axvline(0.5, color="gray", lw=1, ls=":", label="Default (0.5)")
        ax.set_xlabel("Classification Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(thresholds)
        ax.set_title("Track F1 vs Threshold by pT Bin")
        ax.legend(title="Track pT", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_f1_vs_threshold(probs: np.ndarray, truth: np.ndarray, total_e: np.ndarray) -> Figure:
        """Calo mask F1 vs classification threshold, one line per cluster energy bin [0,1,2,5,10,20,∞] GeV."""
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
        bin_edges = [0, 1, 2, 5, 10, 20, np.inf]
        bin_labels = ["0–1 GeV", "1–2 GeV", "2–5 GeV", "5–10 GeV", "10–20 GeV", "> 20 GeV"]
        colors = cm.plasma(np.linspace(0, 0.88, len(bin_labels)))
        is_true_hs = truth.astype(bool)

        fig, ax = plt.subplots(figsize=(8, 5))
        for (lo, hi), label, color in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_labels, colors):
            m = (total_e >= lo) & (total_e < hi)
            if m.sum() < 5:
                continue
            f1_vals = []
            for thr in thresholds:
                is_pred = probs[m] > thr
                tp = (is_pred & is_true_hs[m]).sum()
                fp = (is_pred & ~is_true_hs[m]).sum()
                fn = (~is_pred & is_true_hs[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_vals.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            ax.plot(thresholds, f1_vals, "o-", color=color, label=label)

        ax.axvline(0.5, color="gray", lw=1, ls=":", label="Default (0.5)")
        ax.set_xlabel("Classification Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(thresholds)
        ax.set_title("Calo Mask F1 vs Threshold by Cluster Energy Bin")
        ax.legend(title="Cluster Energy", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_f1_vs_threshold_by_hs_frac(
        probs: np.ndarray, truth: np.ndarray, true_frac: np.ndarray,
    ) -> Figure:
        """Calo mask F1 vs classification threshold, one line per HS fraction bin [0,0.2,0.4,...,1.0]."""
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
        colors = cm.viridis(np.linspace(0.05, 0.92, len(bin_labels)))
        is_true_hs = truth.astype(bool)

        fig, ax = plt.subplots(figsize=(8, 5))
        for (lo, hi), label, color in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_labels, colors):
            m = (true_frac >= lo) & (true_frac < hi)
            if m.sum() < 5:
                continue
            f1_vals = []
            for thr in thresholds:
                is_pred = probs[m] > thr
                tp = (is_pred & is_true_hs[m]).sum()
                fp = (is_pred & ~is_true_hs[m]).sum()
                fn = (~is_pred & is_true_hs[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_vals.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            ax.plot(thresholds, f1_vals, "o-", color=color, label=label)

        ax.axvline(0.5, color="gray", lw=1, ls=":", label="Default (0.5)")
        ax.set_xlabel("Classification Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(thresholds)
        ax.set_title("Calo Mask F1 vs Threshold by HS Fraction Bin")
        ax.legend(title="E_HS / E_cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def _compute_f1_recall_vs_threshold(probs: np.ndarray, truth: np.ndarray) -> tuple:
        """Return (thresholds, f1_vals, recall_vals) arrays."""
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
        is_true = truth.astype(bool)
        f1_vals, rec_vals = [], []
        for thr in thresholds:
            pred = probs > thr
            tp = (pred & is_true).sum()
            fp = (pred & ~is_true).sum()
            fn = (~pred & is_true).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_vals.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            rec_vals.append(rec)
        return thresholds, np.array(f1_vals), np.array(rec_vals)

    @staticmethod
    def plot_calo_mask_f1_recall_neutral(
        probs: np.ndarray, truth: np.ndarray,
        neutral_e: np.ndarray, charged_e: np.ndarray,
    ) -> Figure:
        """F1 and recall vs threshold when ground truth = true HS neutral clusters.

        Positive class: clusters where neutral_e > charged_e AND true HS.
        Negative class: all other clusters.
        Shows how well the model recovers HS energy from trackless particles.
        """
        is_true_hs = truth.astype(bool)
        truth_neutral = ((neutral_e > charged_e) & is_true_hs).astype(int)
        n_pos = truth_neutral.sum()

        thresholds, f1_vals, rec_vals = PhysicsPlotter._compute_f1_recall_vs_threshold(probs, truth_neutral)

        fig, (ax_f1, ax_rec) = plt.subplots(1, 2, figsize=(13, 5))
        for ax, vals, ylabel in [
            (ax_f1, f1_vals, "F1 Score"),
            (ax_rec, rec_vals, "Recall"),
        ]:
            ax.plot(thresholds, vals, "o-", color="steelblue")
            ax.axvline(0.5, color="gray", lw=1, ls=":", label="Default (0.5)")
            ax.set_xlabel("Classification Threshold")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"True HS Neutral Clusters — {ylabel}  (n={n_pos})")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_f1_recall_charged(
        probs: np.ndarray, truth: np.ndarray,
        neutral_e: np.ndarray, charged_e: np.ndarray,
    ) -> Figure:
        """F1 and recall vs threshold when ground truth = true HS charged clusters.

        Positive class: clusters where charged_e >= neutral_e AND true HS.
        Negative class: all other clusters.
        Shows how well the model recovers HS energy from tracked particles.
        """
        is_true_hs = truth.astype(bool)
        truth_charged = ((charged_e >= neutral_e) & is_true_hs).astype(int)
        n_pos = truth_charged.sum()

        thresholds, f1_vals, rec_vals = PhysicsPlotter._compute_f1_recall_vs_threshold(probs, truth_charged)

        fig, (ax_f1, ax_rec) = plt.subplots(1, 2, figsize=(13, 5))
        for ax, vals, ylabel in [
            (ax_f1, f1_vals, "F1 Score"),
            (ax_rec, rec_vals, "Recall"),
        ]:
            ax.plot(thresholds, vals, "o-", color="darkorange")
            ax.axvline(0.5, color="gray", lw=1, ls=":", label="Default (0.5)")
            ax.set_xlabel("Classification Threshold")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"True HS Charged Clusters — {ylabel}  (n={n_pos})")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_hs_frac_category_counts(true_frac: np.ndarray) -> Figure:
        """Bar chart of calo hit counts by HS fraction category.

        Bins (exclusive):
            frac = 0          pure pileup
            0 < frac ≤ 1%
            1% < frac ≤ 5%
            5% < frac ≤ 10%
            frac > 10%
        Plus total as a dashed line annotation.
        """
        frac = np.asarray(true_frac, dtype=float)
        total = len(frac)

        labels = [
            "frac = 0",
            "0 < frac ≤ 1%",
            "1% < frac ≤ 5%",
            "5% < frac ≤ 10%",
            "frac > 10%",
        ]
        masks = [
            frac == 0,
            (frac > 0) & (frac <= 0.01),
            (frac > 0.01) & (frac <= 0.05),
            (frac > 0.05) & (frac <= 0.10),
            frac > 0.10,
        ]
        counts = [m.sum() for m in masks]
        colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452", "#8172b2"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{cnt:,}\n({100 * cnt / total:.1f}%)" if total > 0 else "0",
                ha="center", va="bottom", fontsize=8,
            )
        ax.axhline(total, color="gray", lw=1.2, ls="--", label=f"Total = {total:,}")
        ax.set_ylabel("Number of Calo Hits")
        ax.set_title("Calo Hits by HS Energy Fraction Category")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_metrics_vs_eta(
        mask_pred: np.ndarray, mask_truth: np.ndarray,
        eta: np.ndarray,
    ) -> Figure:
        """F1 and Recall vs cluster η."""
        mask_pred = mask_pred.astype(bool)
        mask_truth = mask_truth.astype(bool)

        bins = np.linspace(-5, 5, 26)
        centers = (bins[:-1] + bins[1:]) / 2

        f1_vals, recall_vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (eta >= lo) & (eta < hi)
            if m.sum() > 0:
                tp = (mask_pred[m] & mask_truth[m]).sum()
                fp = (mask_pred[m] & ~mask_truth[m]).sum()
                fn = (~mask_pred[m] & mask_truth[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_vals.append(f1)
                recall_vals.append(rec)
            else:
                f1_vals.append(np.nan)
                recall_vals.append(np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(centers, f1_vals, "o-", color="steelblue", label="F1")
        ax.plot(centers, recall_vals, "s-", color="tomato", label="Recall")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xlabel("Cluster η")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Calo Mask F1 and Recall vs η")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_metrics_vs_phi(
        mask_pred: np.ndarray, mask_truth: np.ndarray,
        phi: np.ndarray,
    ) -> Figure:
        """F1 and Recall vs cluster φ."""
        mask_pred = mask_pred.astype(bool)
        mask_truth = mask_truth.astype(bool)

        bins = np.linspace(-np.pi, np.pi, 26)
        centers = (bins[:-1] + bins[1:]) / 2

        f1_vals, recall_vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (phi >= lo) & (phi < hi)
            if m.sum() > 0:
                tp = (mask_pred[m] & mask_truth[m]).sum()
                fp = (mask_pred[m] & ~mask_truth[m]).sum()
                fn = (~mask_pred[m] & mask_truth[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_vals.append(f1)
                recall_vals.append(rec)
            else:
                f1_vals.append(np.nan)
                recall_vals.append(np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(centers, f1_vals, "o-", color="steelblue", label="F1")
        ax.plot(centers, recall_vals, "s-", color="tomato", label="Recall")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xlabel("Cluster φ [rad]")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Calo Mask F1 and Recall vs φ")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_metrics_vs_cluster_energy(
        mask_pred: np.ndarray, mask_truth: np.ndarray,
        total_e: np.ndarray,
    ) -> Figure:
        """F1 and Recall vs cluster total energy (log-spaced bins)."""
        mask_pred = mask_pred.astype(bool)
        mask_truth = mask_truth.astype(bool)

        e_pos = total_e[total_e > 0]
        if len(e_pos) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No clusters with E > 0", ha="center", va="center")
            return fig

        emin = max(np.percentile(e_pos, 1), 1e-2)
        emax = np.percentile(e_pos, 99)
        bins = np.logspace(np.log10(emin), np.log10(emax), 25)
        centers = np.sqrt(bins[:-1] * bins[1:])

        f1_vals, recall_vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (total_e >= lo) & (total_e < hi)
            if m.sum() > 0:
                tp = (mask_pred[m] & mask_truth[m]).sum()
                fp = (mask_pred[m] & ~mask_truth[m]).sum()
                fn = (~mask_pred[m] & mask_truth[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_vals.append(f1)
                recall_vals.append(rec)
            else:
                f1_vals.append(np.nan)
                recall_vals.append(np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(centers, f1_vals, "o-", color="steelblue", label="F1")
        ax.plot(centers, recall_vals, "s-", color="tomato", label="Recall")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Cluster Total Energy [GeV]")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Calo Mask F1 and Recall vs Cluster Energy")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calo_mask_metrics_vs_hs_frac(
        mask_pred: np.ndarray, mask_truth: np.ndarray,
        true_frac: np.ndarray,
    ) -> Figure:
        """F1 and Recall vs true HS energy fraction (E_HS / E_cluster)."""
        mask_pred = mask_pred.astype(bool)
        mask_truth = mask_truth.astype(bool)

        bins = np.linspace(0, 1, 26)
        centers = (bins[:-1] + bins[1:]) / 2

        f1_vals, recall_vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (true_frac >= lo) & (true_frac < hi)
            if m.sum() > 0:
                tp = (mask_pred[m] & mask_truth[m]).sum()
                fp = (mask_pred[m] & ~mask_truth[m]).sum()
                fn = (~mask_pred[m] & mask_truth[m]).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_vals.append(f1)
                recall_vals.append(rec)
            else:
                f1_vals.append(np.nan)
                recall_vals.append(np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(centers, f1_vals, "o-", color="steelblue", label="F1")
        ax.plot(centers, recall_vals, "s-", color="tomato", label="Recall")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xlabel("True HS Fraction (E_HS / E_cluster)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Calo Mask F1 and Recall vs HS Energy Fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_roc_curve(probs: np.ndarray, truth: np.ndarray) -> Figure:
        """ROC curve with AUC score."""
        from sklearn.metrics import auc, roc_curve
        is_hs = truth == 1
        if is_hs.sum() == 0 or (~is_hs).sum() == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data for ROC", ha="center", va="center")
            return fig
        fpr, tpr, _ = roc_curve(truth, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate (Pileup Mistag)")
        ax.set_ylabel("True Positive Rate (HS Efficiency)")
        ax.set_title("Track ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_deltaR_window_analysis(
        stats: dict,
        window_sizes: list | None = None,
    ) -> Figure:
        """Line plot of mean and max delta R within Morton-sorted window vs window size.

        Parameters
        ----------
        stats : dict
            Output of compute_deltaR_window_stats:
            {window_size -> {"mean_dR": list[float], "max_dR": list[float]}}.
        window_sizes : list of int, optional
            Ordered window sizes to plot. Defaults to sorted keys of stats.
        """
        if window_sizes is None:
            window_sizes = sorted(stats.keys())

        mean_means = [np.mean(stats[w]["mean_dR"]) for w in window_sizes]
        mean_maxs  = [np.mean(stats[w]["max_dR"])  for w in window_sizes]
        q1_means   = [np.percentile(stats[w]["mean_dR"], 25) for w in window_sizes]
        q3_means   = [np.percentile(stats[w]["mean_dR"], 75) for w in window_sizes]
        q1_maxs    = [np.percentile(stats[w]["max_dR"],  25) for w in window_sizes]
        q3_maxs    = [np.percentile(stats[w]["max_dR"],  75) for w in window_sizes]

        xs = np.array(window_sizes)
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(xs, mean_means, "o-", color="steelblue", lw=2, ms=7, label="Mean δR (within window)")
        ax.fill_between(xs, q1_means, q3_means, alpha=0.2, color="steelblue", label="Q1–Q3 (mean δR)")

        ax.plot(xs, mean_maxs, "s--", color="tomato", lw=2, ms=7, label="Mean max δR (within window)")
        ax.fill_between(xs, q1_maxs, q3_maxs, alpha=0.15, color="tomato", label="Q1–Q3 (max δR)")

        ax.set_xlabel("Window Size (# nodes)")
        ax.set_ylabel("δR  (η-φ space)")
        ax.set_title("Delta R Coverage vs Window Size (Morton-sorted nodes)")
        ax.set_xticks(window_sizes)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def _plot_component_window_analysis(
        stats: dict,
        key: str,
        ylabel: str,
        title: str,
        color: str,
        window_sizes: list | None = None,
    ) -> Figure:
        if window_sizes is None:
            window_sizes = sorted(stats.keys())
        means = [np.mean(stats[w][key]) for w in window_sizes]
        q1    = [np.percentile(stats[w][key], 25) for w in window_sizes]
        q3    = [np.percentile(stats[w][key], 75) for w in window_sizes]
        xs = np.array(window_sizes)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, means, "o-", color=color, lw=2, ms=7, label=f"Mean {ylabel}")
        ax.fill_between(xs, q1, q3, alpha=0.2, color=color, label="Q1–Q3")
        ax.set_xlabel("Window Size (# nodes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(window_sizes)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_eta_window_analysis(stats: dict, window_sizes: list | None = None) -> Figure:
        """Mean |Δη| within Morton-sorted window vs window size."""
        return PhysicsPlotter._plot_component_window_analysis(
            stats, "mean_deta", "|Δη|",
            "Eta Coverage vs Window Size (Morton-sorted nodes)",
            "steelblue", window_sizes,
        )

    @staticmethod
    def plot_phi_window_analysis(stats: dict, window_sizes: list | None = None) -> Figure:
        """Mean |Δφ| within Morton-sorted window vs window size."""
        return PhysicsPlotter._plot_component_window_analysis(
            stats, "mean_dphi", "|Δφ|",
            "Phi Coverage vs Window Size (Morton-sorted nodes)",
            "darkorange", window_sizes,
        )
