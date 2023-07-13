# SHREC2023 - Track 2: Sketch-based - Team TikTorch

## 1. Installation
<span id='2-install'></span>

Before installing the repo, we need to install the CUDA driver version >=11.6

We will create a new conda environment:

```bash
conda create -n animar python=3.8 -y
conda activate animar
bash install.sh
```

## 2. Download resources

We need to download the resources: trained weights, training set and test sets:

```bash
python tools/download.py
```

**NOTE**. If you don't see any logs in the terminal, please terminate that process (ctrl + C) and run the command again until it works... (this is a unexpected behavior of `gdown` library).


Now, the structure of `data` folder will look like this:

```

`─ data/
   `-- best_sketch_weights/         # best model
   |   `-- weights/
   |   |   `-- *.pth
   |   `-- *.json
   |   `-- ...
   |
   `─ sketch_data/                  # training set
   |   `-- SketchANIMAR2023/
   │   |   `-- 3D_Model_References/
   |   |   |   `-- generated_sketches/
   |   |   |   |   `-- ...
   |   |   |   `-- References.csv
   |   |   `-- ...
   |   `-- ...  
   |  
   `─ public_test/                  # test set
   │   `-- CroppedSketchQuery_Test/
   │   |   `-- *.jpg
   │   `-- SketchQuery_Test.csv
   │   `-- ...
   |
   `-- ... 
```

## 3. Inference

Retrieval command:

```
python retrieve_sketch_ringview.py \
    --exps-path ./data/best_sketch_weights \
    --rings-path ./data/sketch_data/SketchANIMAR2023/3D_Model_References/generated_sketches \
    --obj-csv-path ./data/sketch_data/SketchANIMAR2023/3D_Model_References/References.csv \
    --skt-data-path ./data/public_test/CroppedSketchQuery_Test \
    --skt-csv-path ./data/public_test/SketchQuery_Test.csv \
    --output-path sketch_predicts
```

The retrieval results will be on the directory `sketch_predicts/ringview_predict_{num}`. We use the file `submission.csv` in this directory to submit the results.

## 4. References

This repository is based on the [official baseline](https://github.com/nhtlongcs/SHREC23-ANIMAR-BASELINE) of the organizers.
