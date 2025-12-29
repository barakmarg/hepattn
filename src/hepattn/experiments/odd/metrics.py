"""
ODD Particle Flow Metrics.

Utility classes and functions for computing mask inference metrics.
"""

import torch


class MaskInference:
    """
    Static methods for mask prediction inference and metrics.

    Used to convert model outputs to discrete predictions and evaluate performance.
    """

    @staticmethod
    def basic_sigmoid(pred: torch.Tensor) -> torch.Tensor:
        """
        Assign nodes to particles if they have high matching probability.

        Can assign a node to more than one particle.

        Args:
            pred: Raw logits of shape (num_particles, num_nodes)

        Returns:
            Boolean mask of shape (num_particles, num_nodes)
        """
        return pred.sigmoid() > 0.5

    @staticmethod
    def basic_argmax(pred: torch.Tensor) -> torch.Tensor:
        """
        Assign nodes to the particle with highest probability.

        Each node can only be assigned to one particle.

        Args:
            pred: Raw logits of shape (num_particles, num_nodes)

        Returns:
            Boolean mask of shape (num_particles, num_nodes)
        """
        idx = pred.argmax(-2)
        pred = torch.full_like(pred, False).bool()
        pred[idx, torch.arange(len(idx))] = True
        return pred

    @staticmethod
    def weighted_argmax(
        pred: torch.Tensor,
        class_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Assign nodes weighted by class prediction confidence.

        This is the approach used in the MaskFormer paper.
        Each node can only be assigned to one particle.

        Args:
            pred: Raw logits of shape (num_particles, num_nodes)
            class_preds: Class logits of shape (num_particles, num_classes)

        Returns:
            Boolean mask of shape (num_particles, num_nodes)
        """
        idx = (pred.softmax(-2) * class_preds.max(-1)[0].unsqueeze(-1)).argmax(-2)
        pred = torch.zeros_like(pred).bool()
        pred[idx, torch.arange(len(idx))] = True
        return pred

    @staticmethod
    def exact_match(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute exact match rate between predicted and target masks.

        Args:
            pred: Predicted boolean mask of shape (num_particles, num_nodes)
            tgt: Target boolean mask of shape (num_particles, num_nodes)

        Returns:
            Fraction of particles with perfectly matched masks
        """
        if len(tgt) == 0:
            return torch.tensor(torch.nan)
        return (pred == tgt).all(-1).float().mean()

    @staticmethod
    def eff(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute efficiency (recall) of node assignment.

        Fraction of true nodes that are correctly predicted.

        Args:
            pred: Predicted boolean mask of shape (num_particles, num_nodes)
            tgt: Target boolean mask of shape (num_particles, num_nodes)

        Returns:
            Mean efficiency across particles
        """
        return ((pred & tgt).sum(-1) / tgt.sum(-1)).mean()

    @staticmethod
    def pur(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute purity (precision) of node assignment.

        Fraction of predicted nodes that are correct.

        Args:
            pred: Predicted boolean mask of shape (num_particles, num_nodes)
            tgt: Target boolean mask of shape (num_particles, num_nodes)

        Returns:
            Mean purity across particles
        """
        return ((pred & tgt).sum(-1) / pred.sum(-1)).mean()


# TODO: Add additional metric classes as needed
# class RegressionMetrics:
#     """Metrics for regression tasks (energy, momentum, etc.)."""
#
#     @staticmethod
#     def resolution(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
#         """Compute relative resolution (pred - tgt) / tgt."""
#         return ((pred - tgt) / tgt).std()
#
#     @staticmethod
#     def bias(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
#         """Compute mean bias."""
#         return ((pred - tgt) / tgt).mean()


# TODO: Add jet-level metrics if needed
# class JetMetrics:
#     """Metrics for jet reconstruction performance."""
#     pass
