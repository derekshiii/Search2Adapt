# Dataset Preparation

Search2Adapt operates on **3D NIfTI volumes** in the nnU-Net dataset layout. The target dataset is the only data the framework ever touches — no source data is required at any point (that is what makes the setting *source-free*).

## Expected layout

```
$nnUNet_raw/Dataset500_AMOS_MR/
├── imagesTr/
│   ├── AMOS_0001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── AMOS_0001.nii.gz
│   └── ...
└── dataset.json
```

- `imagesTr/` — target images. The converter looks for the `*_0000.nii.gz` modality suffix; single-modality datasets should still carry it.
- `labelsTr/` — **optional**. Ground truth is only read by the evaluation step (`search2adapt evaluate`) and by the post-hoc metrics in `search2adapt/metrics.py`. The search itself never sees it, which is what keeps the pipeline honest in the SFUDA setting.

`dataset.json` is optional but recommended — the distillation stage reads `channel_names` from it so the student model keeps the source model's channel naming.

## Label IDs and text prompts

The Perceptor maps each requested organ to a **label ID** and a **text prompt** that is passed to BiomedParse. The prompt map is a plain `{label_id: prompt}` dictionary:

```python
{
    "1": "liver",
    "2": "spleen",
    "3": "left kidney",
    "4": "right kidney",
}
```

The prompt string is what the VFM is conditioned on, so it should name the organ and the anatomy it lives in ("liver in the upper abdomen" is more reliable than "liver" on out-of-distribution data).

Label IDs must be consistent between the prompt map and `labelsTr/` if you intend to evaluate DICE per organ.

## Datasets used in the paper

The paper evaluates 13 adaptation directions over 24 anatomical targets. Each needs a target dataset in the layout above.

| Target | Dataset | Notes |
|---|---|---|
| Abdominal CT / MR | [AMOS](https://arxiv.org/abs/2208.08326) | 15 abdominal organs, multi-center CT and MR |
| Brain MRI | [BraTS-2024](https://arxiv.org/abs/2405.18368) | subregions ET, TC, SNFH, RC; modalities T1n, T1c, T2w, T2f |
| Cardiac MR | [ACDC](https://github.com/creatis-IAIGS/acdc_dataset) | LV, MYO, RV |
| Cardiac US | [CAMUS](https://www.creatis.insa-lyon.fr/valentin.oliveira/camus/) | LV, MYO, LA |
| Polyp | [Kvasir-SEG](https://www.kaggle.com/datasets/andrewmvd/polyp-segmentation) and [CVCDB](http://www.cvc.climatc.unizar.es/?page_id=391) | 2D endoscopic frames |
| Fundus | [RIGA+](https://zenodo.org/records/6325549) | source RIGA (BinRushed, Magrabia) → targets BASE1/2/3 |

### Abdominal (AMOS) — MR→CT and CT→MR

AMOS ships CT and MR volumes with voxel-wise annotations for 15 organs. Use the CT subset as the target when adapting MR→CT, and the MR subset for CT→MR.

The 15 organ labels used in Table 1 are: Spleen, Right Kidney, Left Kidney, Gallbladder, Esophagus, Liver, Stomach, Aorta, Vena Cava, Pancreas, Right Adrenal, Left Adrenal, Duodenum, Bladder, Prostate/Uterus.

### Brain (BraTS-2024)

BraTS volumes are co-registered across modalities, so a single 4-channel NIfTI (`_0000.nii.gz` holds channel 0) can be split per modality before the run, or the modalities can be treated as separate target datasets. The four annotated subregions are:

| Label | Region |
|---|---|
| 1 | Necrotic tumor core (NCR) |
| 2 | Surrounding non-enhancing FLAIR hyperintensity (SNFH) |
| 3 | Enhancing tumor (ET) |
| 4 | Resection cavity (RC) |

### Cardiac (ACDC / CAMUS)

ACDC provides 3D cardiac MR; CAMUS provides 2D ultrasound. For the cross-modality settings (MR→US, US→MR) the paper evaluates LV and MYO. The US volumes must be stacked into a 3D volume before they can be processed — the framework itself is volumetric only.

### Polyp (Kvasir / CVCDB)

Both datasets are 2D. Stack the frames of each video into a 3D volume along the frame axis and keep the original frame order so the metrics stay comparable with the paper.

### Fundus (RIGA+)

RIGA+ covers five domains. The paper fixes the source to RIGA (BinRushed + Magrabia) and adapts to the MESSIDOR subsets BASE1, BASE2, BASE3, evaluating the optic disc (OD) and optic cup (OC).

## Converting to the BiomedParse input format

The framework converts the nnU-Net dataset into BiomedParse NPZ files automatically (via `search2adapt/vfm/nnunet_io.py`). To run the conversion by hand:

```python
from search2adapt.vfm.nnunet_io import convert_nnunet_to_biomedparse

convert_nnunet_to_biomedparse(
    input_dir="/path/to/Dataset500_AMOS_MR",
    output_dir="/tmp/bp_npz",
    text_prompts={"1": "liver", "2": "spleen"},
)
```

Then run the standalone BiomedParse predictor:

```bash
python -m search2adapt.vfm.biomedparse_3d_standalone \
  --input_dir  /tmp/bp_npz \
  --output_dir /tmp/bp_out \
  --checkpoint "$SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT" \
  --biomedparse_root "$SEARCH2ADAPT_BIOMEDPARSE_ROOT" \
  --gpus 0 \
  --save_nii
```

In a normal pipeline run you never do this by hand — the Executor handles the conversion and the subprocess call.

## Split discipline

The paper generates pseudo-labels and distills **strictly on the training split**. Keep the target training split in `imagesTr/` and hold the evaluation split outside the dataset directory (or in a separate `imagesTe/`) so it can never be sampled by the Perceptor or the search.

## Evaluation

```bash
search2adapt evaluate \
  --result-dir ./S2A_Results/AMOS_MR \
  --gt-dir /path/to/Dataset500_AMOS_MR/labelsTr \
  --label-names liver,spleen,kidney_left,kidney_right
```

`--label-ids` selects explicit label IDs; `--label-names` assigns names positionally. Without either, every non-zero label in the ground truth is evaluated.
