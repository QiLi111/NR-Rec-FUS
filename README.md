
# Nonrigid Reconstruction of Freehand Ultrasound without a Tracker (MICCAI2024)

This repository is the official implementation for "[Nonrigid Reconstruction of Freehand Ultrasound without a Tracker](https://arxiv.org/abs/2407.05767)". It contains algorithms for freehand ultrasound reocnstruction, simultaneously estimating rigid transformations among US frames and a nonrigid deformation, optimised by a regularised registration network.

## Install conda environment
``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
``` 

## Data
This repository contains a few samples of the full data set.
The full data set is available [here](https://zenodo.org/doi/10.5281/zenodo.11178508), associated with the TUS-REC Challenge, MICCAI 2024. We have reorganised the data set to provide a clearer data structure. For more information, please visit the [Challenge website](https://github-pages.ucl.ac.uk/tus-rec-challenge/) and [GitHub repo](https://github.com/QiLi111/tus-rec-challenge_baseline). 


The example data set is stored in a `.h5` file, including train (6 scans), val (2 scans), and test (2 scans) sets. The indices of samped scans are indicated by `.json` files. The following shows the data structure.
```
    scans_res4_forth.h5/
    │
    ├── /subXXX_framesXX - US frames in a scan
    ├── /subXXX_tformsXX - associated transformations for each frame, from tracker space to optical camera space
    ├── num_frames - number of frames in each scan
    ├── name_scan - scan protocol 

    fold_XX_seqlenXXX_sub_forth.json/
    │
    ├── indices_in_use - the indices of samped scans from the original whole data set
```

## Train
The two objectives, rigid transformations and nonrigid deformation, can be either optimised using meta-learning or combined by weighting in an end-to-end manner. 

* End-to-end training
    ``` bash
    python3 train_ete.py --config config/config_ete.json
    ``` 

* Meta training
    ``` bash
    python3 train_meta.py --config config/config_meta.json
    ``` 
## Test
``` bash
python3 test.py
``` 



