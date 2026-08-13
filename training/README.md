The original Graph-pMHC repository ships inference-only code. Everything in this repo related to 
training (training loop, checkpoint-loading verification, diagnostics, evaluation harness) was written independently using the public model/config. Notebooks are exported from Google Colab.

## Gpmhc Baseline Reproduction ##
A training/evaluation loop was implemented from-scratch to verify that the published checkpoint and 
reported performance (~0.82 AP) can be reproduced independently, and to serve as a base for further
fine-tuning.

`train_HLAII_baseline.py` loads the published `model_final.pth` checkpoint and evaluates it on the 
held-out test set before any optimizer step, to confirm the reconstructed architecture + data pipeline
reproduce the reference ~0.82 AP. 
`train_v4_DPDRDQhet.ipynb` - Colab notebook used to set up the environment and run the scripts 
(heterodimer-only training subset)

`train_HLAII_baseline_chemannot.py` loads reproduced baseline and fine-tunes with altered 
hyperparameters and chemistry aware edge features (charge, hydrophobicity, hydrogen-bond compatibility
on intermolecular peptide-mhc edges) reaching ~0.86 test AP. This result motivated the next direction
below, as it suggests the idea that graph enumeration itself (single allotypic followed by multi-allotypic
graphs) could be improved; i.e. improved heterodimer performance would improve graph enumeration-mediated
deconvolution

`train_v6_NetMHC_DQaugment.ipynb` - notebook evaluating whether additional DQ coverage generated 
from NetMHCIIpan 4.3 EL data would improve performance. Candidate peptides were screened using 
exact/core-level similarity criteria to reduce redundancy within the NetMHC-derived set, then longer 
peptides were converted into 9-mer sliding-window examples with complete 5-aa N- and C-terminal flanks.
This produced ~9.4k unique DQ training examples appended to the existing heterodimer-only training set,
increasing DQ sequence coverage while preserving the original Graph-pMHC schema and train/test structure.

`train_v2_DQtraintest.ipynb` is an earlier exploratory notebook building the DQ-only train/test split 
from `Presentation_df_w_preds.csv` (with peptide and allotype-level leakage checks), sanity-checks in 
the pipeline, and runs an initial fine-tuning pass. Included as a prelude showing the debugging process.

### Planned next steps ###
Exploring whether training on single-allotypic samples before introducing multi-allotypic (more 
clinically relevant) samples improve inherent deconvolution via graph enumeration and therefore
downstream ranking and overall model performance.

Molecular Dynamics Simulation pilot: use OpenMM (convenient integration) to identify biophysical 
features amongst high-error DQ, low-error DQ, and control DP/DR AF2 structures that further build
the static adjacency matrix. 
Low-cost improvements: 

  per-residue flexibility (root mean square flexibility)
  
  contact persistence
  
  SASA - solvent accessible surface area
  

### Verified ###
Checkpoint reproduction: confirmed, loading `model_final.pth` and evaluating on heterodimer test set
reproduces the published reference

Fine-tuning result: an early run (altered hyperparameters and additional chemistry-aware edge features)
reached ~0.86 test AP. This is a provisional metric as it was obtained without a held-out validation
split, so epoch/config selection was implicitly made against the test set.

DQ augmentation: adding ~9.4k NetMHCIIpan-derived DQ examples did not produce a meaningful improvement 
in aggregate performance. The most likely explanation is that the augmentation provided limited novel 
sequence coverage and remained substantially redundant with the existing training distribution.

### Notes ###
These scripts and notebooks were developed and run against a Google Drive/Colab working
directory (`/content/drive/MyDrive/gpmhc/gpmhc_train/`) which won't match this repo's layout. Paths
shown in the notebooks reflect that original working setup; if running this repo (forked under a new
`training/` folder alongside the original code), adjust paths to be relative to wherever its cloned,
or drop files into the same relative position.

License: This repo imports and builds on Genentech's `gpmhc` package, licensed under the non-commercial
variant of Apache2.0. Use of this repo is thus subject to the same terms.

### Requirements ###
See `train_v4_DRDRDQhet.ipynb` for the full colab environment setup (Python 3.10, PyTorch2.12+cu121,
DGL1.1.3+cu121, fastai 2.7.14). CUDA is required as the architecture's forward pass hardcodes GPU
tensors and has no CPU code path. 





