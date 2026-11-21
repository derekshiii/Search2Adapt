"""
Vision Foundation Model (VFM) runners for Search2Adapt.

Each VFM runs in its own conda environment, so the runners are launched as
standalone subprocesses:

    search2adapt/vfm/biomedparse_3d_standalone.py   (BiomedParse v2, 3D)
    search2adapt/vfm/sam3_3d_standalone.py          (SAM 3 video predictor, 3D)

The modules in this package are imported by the Search2Adapt toolbox, which
only builds the subprocess command lines — it never imports torch, hydra or
the VFM packages themselves.
"""

__all__ = ["biomedparse_3d_standalone", "sam3_3d_standalone", "nnunet_io"]
