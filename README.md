# SHREC23-ANIMAR

## 1. Install dependencies

Before installing the repo, we need to install the CUDA driver version >=11.6

If necessary, you could create a conda env:

```bash
$ conda env create -f animar.yml
$ conda activate animar
```
Build for PointMLP:

```bash
$ pip install utils/pointnet2_ops_lib/.
```

## 2. Dataset structure

Resources:

- Original data: [GDrive](https://drive.google.com/drive/folders/1lox1J_C3XXYpXeGdHh34QnXnvx9Vhj4Y?usp=share_link)

<!-- - Generated models (for RingView method, removed `depth` and `mask` folders): [GDrive](https://drive.google.com/file/d/1UsnawE_BLqK6vzJ0RGAa6KtXniJm3JkL/view?usp=sharing)

- Cropped sketches: [GDrive](https://drive.google.com/file/d/1AbaWwM0YP_7DLgOiP2U7tnV3OmsEfA0O/view?usp=share_link) -->

- Processed data: [Kaggle Dataset](https://kaggle.com/datasets/e1250d59a160e13c8c97d3d45006efe9f109ee338f09e664f5dc57f9625d616d)

```
SHREC2023-ANIMAR
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
|  |  |  ├─ generated_sketches/
│  │  ├─ Train/
│  │  │  ├─ SketchQuery_Train/
|  |  |  ├─ CroppedSketchQuery_Train/
│  │  │  ├─ *GT_Train.csv
│  │  │  ├─ *Train.csv
├─ ...
```

## 3. Training for sketch-based (track 2)

Current available models:

- CNN backbone: ResNet (`resnetXX`), EfficientNet (`efficientnet_bX`, `efficientnet_v2_X`)
- PCL model: CurveNet (`curvenet`), PointMLP (`pointmlp`), PointMLPElite (`pointmlpelite`)
- View sequence embedder: LSTM/BiLSTM (`bilstm`), Transformer Encoder (`mha`)
- Ring sequence embedder: Transformer Encoder (support multi MHA layers)

MLP for embedding to common space can be in the shinking (default) or expanding mode.

**NOTE**.
- Cross Batch Memory in training is currently disabled. If you want to use it, let's add the flag `--use-cbm` in training commands.
- Add the flag `--reduce-lr` to use the learning rate schedule.

### 3.1. Point-cloud Method

The meaning and default values of arguments:

```
python train_sketch_pcl.py --help
```

For example:

```
python train_sketch_pcl.py \
    --pcl-model curvenet \
    --cnn-backbone efficientnet_b2 \
    --obj-data-path data/SketchANIMAR2023/3D_Model_References/References \
    --skt-data-path data/SketchANIMAR2023/Train/CroppedSketchQuery_Train \
    --train-csv-path data/csv/train_skt.csv \
    --test-csv-path data/csv/test_skt.csv \
    --batch-size 4 \
    --epochs 50 \
    --latent-dim 256 \
    --output-path exps \
    --lr-obj 1e-4 \
    --lr-skt 1e-4
```
- The original baseline use ResNet50 backbone and Curvenet model:
`cnn-backbone=resnet50`, `pcl-model=curvenet`.

The result of training process will be put inside folder `exps/pcl_exp_{num}` (`num` is counted from 0)

### 3.2. Ring-view Method

The meaning and default values of arguments:

```
python train_sketch_ringview.py --help
```

We can use the processed ring-view images for training (`generated_sketches`), or use the default ring-view images (`generated_models`).

For example:

```
python train_sketch_ringview.py \
    --view-cnn-backbone efficientnet_v2_s \
    --skt-cnn-backbone efficientnet_v2_s \
    --rings-path data/SketchANIMAR2023/3D_Model_References/generated_sketches \
    --used-rings 2,3,4,5 \
    --skt-data-path data/SketchANIMAR2023/Train/CroppedSketchQuery_Train \
    --train-csv-path data/csv/train_skt.csv \
    --test-csv-path data/csv/test_skt.csv \
    --batch-size 2 \
    --epochs 50 \
    --latent-dim 256 \
    --output-path exps \
    --view-seq-embedder mha \
    --num-rings-mhas 2 \
    --num-heads 4 \
    --lr-obj 1e-4 \
    --lr-skt 1e-4
```

- The original baseline use ResNet50 backbone and LSTM + MHA (just a MHA layer, without any layernorm and skip connection): `*-backbone=resnet50`, `view-seq-embedder=bilstm`, `num-rings-mhas=1`.

The result of training process will be put inside folder `exps/ringview_exp_{num}` (`num` is counted from 0)

## 4. Training for text-based (track 3)

- Plan: Sentence-BERT instead of BERT

## 5. Retrieval

### 5.1. Sketch-based

- Point-cloud method:

```
python retrieve_sketch_pcl.py \
    --info-json ./exps/pcl_exp_0/args.json \
    --output-path predicts \
    --obj-data-path ./data/SketchANIMAR2023/3D_Model_References/References \
    --obj-csv-path ./data/SketchANIMAR2023/3D_Model_References/References.csv \
    --skt-data-path ./data/SketchANIMAR2023/Public\ Test/CroppedSketchQuery_Test \
    --skt-csv-path ./data/SketchANIMAR2023/Public\ Test/SketchQuery_Test.csv \
    --obj-weight ./exps/pcl_exp_0/weights/best_obj_embedder.pth \
    --skt-weight ./exps/pcl_exp_0/weights/best_query_embedder.pth
```

- Ring-view method

```
python retrieve_sketch_ringview.py \
    --info-json ./exps/ringview_exp_0/args.json \
    --rings-path data/SketchANIMAR2023/3D_Model_References/generated_sketches \
    --obj-csv-path ./data/SketchANIMAR2023/3D_Model_References/References.csv \
    --skt-data-path ./data/SketchANIMAR2023/Public\ Test/CroppedSketchQuery_Test \
    --skt-csv-path ./data/SketchANIMAR2023/Public\ Test/SketchQuery_Test.csv \
    --obj-weight ./exps/ringview_exp_0/weights/best_obj_embedder.pth \
    --skt-weight ./exps/ringview_exp_0/weights/best_query_embedder.pth \
    --output-path predicts
```
### 5.2. Text-based

...

## 6. Ensemble query results

For example, the folder `predicts/pointmlp` is currently in these structure:

```
├─ predicts/pointmlp/
│  ├─ a_rand_name/
│  │  ├─ query_results.json
│  │  ├─ ...
|
│  ├─ other_rand_name/
│  │  ├─ query_results.json
│  │  ├─ ...
|  |
|  ├─ ...
```

Ensemble results:

```
python utils/ensemble_results.py \
    --input-folder predicts/pointmlp \
    --output-folder predicts/pointmlp_ensembled
```

After running the above command, we get the result folder `predicts/pointmlp_ensembled` storing 2 files: `query_results.json` and `submission.csv`.