"""Backwards-compatibility shim for old checkpoints — no implementation here.

The real class now lives at
``hepattn.experiments.odd_pileup_reco.pileup_maskformer_decoder``; this module
only re-exports it under the pre-consolidation import path.

Why it is needed: Lightning bakes the model's class paths into every ``.ckpt``
(``checkpoint["hyper_parameters"]``). Checkpoints trained before the ODD
experiments were consolidated reference
``hepattn.experiments.odd_pileup_maskformer.decoder.PileupMaskFormerDecoder``,
and ``load_from_checkpoint()`` imports that exact string when rebuilding the
model. Deleting this module therefore breaks ``run_forward_pass`` on every
pre-existing checkpoint, the paper checkpoint included.

Live code must NOT import from here — ``configs/base.yaml`` and
``odd_pileup_reco/decoder.py`` both point at the vendored module directly.
See this package's ``__init__.py`` for the full story.
"""

from hepattn.experiments.odd_pileup_reco.pileup_maskformer_decoder import PileupMaskFormerDecoder

__all__ = ["PileupMaskFormerDecoder"]
