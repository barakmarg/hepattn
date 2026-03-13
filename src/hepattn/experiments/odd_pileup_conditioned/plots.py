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
        ax.set_title("Calo: HS Energy Residual")
        ax.legend()
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
