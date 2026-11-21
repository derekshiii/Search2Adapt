"""
BiomedParse v2 3D inference backend.

Vendored from the BiomedParse v2 release (https://github.com/microsoft/BiomedParse,
Apache-2.0) so that Search2Adapt does not depend on the upstream checkout for
the inference loop itself. The upstream checkout is still required for the
model definition (``src/``) and the hydra configuration
(``configs/model/biomedparse_3D.yaml``) — point ``--biomedparse_root`` at it.

Changes relative to the upstream ``inference.py``:

* hard-coded dataset/checkpoint paths replaced by module-level configuration
  that the standalone runner overrides;
* ``from utils import ...`` / ``from npz_utils import ...`` replaced by
  package-relative imports;
* the hydra config directory is resolved from ``BIOMEDPARSE_ROOT`` instead of
  the current working directory.
"""

from .inference import (
    _build_prompt_text,
    load_case,
    main,
    merge_multiclass_masks,
    postprocess,
    process_single_file,
    save_as_nii,
)

__all__ = [
    "_build_prompt_text",
    "load_case",
    "main",
    "merge_multiclass_masks",
    "postprocess",
    "process_single_file",
    "save_as_nii",
]
