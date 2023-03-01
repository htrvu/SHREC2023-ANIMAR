# SHREC23-ANIMAR

## Install dependencies (if necessary)

Before installing the repo, we need to install the CUDA driver version >=11.6

```bash
$ conda env create -f animar.yml
$ conda activate animar
```

## Dataset structure

Resources:

- Original data: [GDrive](https://drive.google.com/drive/folders/1lox1J_C3XXYpXeGdHh34QnXnvx9Vhj4Y?usp=share_link)

- Generated models (for RingView method, removed `depth` and `mask` folders): [GDrive](https://drive.google.com/file/d/1UsnawE_BLqK6vzJ0RGAa6KtXniJm3JkL/view?usp=sharing)

- Cropped sketches: ...

```
./
├─ data/
│  ├─ TextANIMAR2023/
│  │  ├─ 3D_Model_References/
│  │  │  ├─ References/
│  │  ├─ Train/
│  │  │  ├─ *GT_Train.csv
│  │  │  ├─ *Train.csv
|
│  ├─ SketchANIMAR2023/
│  │  ├─ 3D_Model_References/
│  │  │  ├─ References/
|  |  |  ├─ generated_models/
│  │  ├─ Train/
│  │  │  ├─ SketchQuery_Train/
│  │  │  ├─ *GT_Train.csv
│  │  │  ├─ *Train.csv
├─ ...
```

## Training for sketch-based (track 2)

Current available models:

- CNN backbone: ResNet (`resnetXX`), EfficientNet (`efficientnet_bX`)
- PCL model: CurveNet
- View sequence embedder: LSTM/BiLSTM (`bilstm`), Transformer Encoder (`mha`)
- Ring sequence embedder: Transformer Encoder (support multi MHA layers)

MLP for embedding to common space can be in the shinking (default) or expanding mode.

### Point-cloud Method

The meaning and values of arguments:

```
python train_sketch_pcl.py --help
```

For example:

```
python train_sketch_pcl.py \
    --pcl-model curvenet \
    --cnn-backbone efficientnet_b2 \
    --obj-data-path data/SketchANIMAR2023/3D_Model_References/References \
    --skt-data-path data/SketchANIMAR2023/Train/SketchQuery_Train \
    --train-csv-path data/csv/train_skt.csv \
    --test-csv-path data/csv/test_skt.csv \
    --batch-size 4 \
    --epochs 50 \
    --latent-dim 256 \
    --output-path exps \
    --lr-obj 1e-4 \
    --lr-skt 1e-4
```
- The original baseline use ResNet50 backbone and Curvenet model.
`*-backbone=resnet50`.

The result of training process will be put inside folder `exps/pcl_exp_{num}`

### Ring-view method

**WARNING.** Have not checked for EfficientNet backbone yet (out of memory error).

The meaning and values of of arguments

```
python train_sketch_ringview.py --help
```

For example:

```
python train_sketch_ringview.py \
    --view-cnn-backbone resnet18 \
    --skt-cnn-backbone resnet18 \
    --rings-path data/SketchANIMAR2023/3D_Model_References/generated_models \
    --num-rings 6 \
    --skt-data-path data/SketchANIMAR2023/Train/SketchQuery_Train \
    --train-csv-path data/csv/train_skt.csv \
    --test-csv-path data/csv/test_skt.csv \
    --batch-size 2 \
    --epochs 50 \
    --latent-dim 256 \
    --output-path exps \
    --lr-obj 1e-4 \
    --lr-skt 1e-4 \
    --view-seq-embedder mha \
    --num-rings-mhas 2 \
    --num-heads 4
```

- The original baseline use ResNet50 backbone and LSTM + MHA (just a MHA layer, without any layernorm and skip connection): `*-backbone=resnet50`, `view-seq-embedder=bilstm`, `num-rings-mhas=1`.

The result of training process will be put inside folder `exps/ringview_exp_{num}`

## Training for text-based (track 2)

- Plan: Sentence-BERT instead of BERT

## Retrieval

- ...