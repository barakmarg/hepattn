"""
ODD Pileup Removal + Reconstruction Experiment.

Three-stream MaskFormer:
  Stream A: Track HS classification
  Stream B: Calo HS mask + energy fraction
  Stream C: Particle reconstruction (classification, mask, incidence, regression)
"""

from hepattn.experiments.odd_pileup_reco_pu_cond.pflow_data import ODDDataModule, ODDDatasetPileup
from hepattn.experiments.odd_pileup_reco_pu_cond.lightning_module import ODDPFlowTwoStream
from hepattn.experiments.odd_pileup_reco_pu_cond.predictionwriter import PflowPredictionWriter
from hepattn.experiments.odd_pileup_reco_pu_cond.eval_data import load_eval_data_from_h5, run_forward_pass
from hepattn.experiments.odd_pileup_reco_pu_cond.reco_analysis import load_pflow_data, run_reco_analysis

__all__ = [
    "ODDDatasetPileup",
    "ODDDataModule",
    "ODDPFlowTwoStream",
    "PflowPredictionWriter",
    "load_eval_data_from_h5",
    "run_forward_pass",
    "load_pflow_data",
    "run_reco_analysis",
]
