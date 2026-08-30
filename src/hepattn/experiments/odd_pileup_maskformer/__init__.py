"""Backwards-compatibility shim — NOT a live experiment.

The ODD pileup experiments were consolidated into
``hepattn.experiments.odd_pileup_reco`` on the ``odd-paper`` branch, and the
real ``odd_pileup_maskformer`` experiment code was removed. This package
survives for one reason only: **checkpoint loading**.

Lightning stores the model's instantiation config inside every ``.ckpt`` file
(``checkpoint["hyper_parameters"]``). Checkpoints trained before the
consolidation embed the literal string::

    hepattn.experiments.odd_pileup_maskformer.decoder.PileupMaskFormerDecoder

``ODDPFlowTwoStream.load_from_checkpoint()`` rebuilds the model by importing
that path, so without this shim *every existing checkpoint* — including the one
used for the paper results — fails to load with::

    module 'hepattn.experiments.odd_pileup_maskformer' has no attribute 'decoder'

Training from scratch never touches this package: ``configs/base.yaml`` points
at the real, vendored module in ``odd_pileup_reco``.

Do not add anything else here. If you ever re-train such that no surviving
checkpoint references this path, the whole package can be deleted.
"""
