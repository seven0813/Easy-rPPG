# Easy-rPPG

Easy-rPPG 是一个基于 PyTorch 的轻量级 rPPG 训练与评估框架。当前开源版本以 **PhysNet** 为主要可运行模型，提供从视频预处理、H5 数据读取、模型训练、验证、测试到指标计算的完整流程。

项目采用 YAML 配置驱动。通常只需要准备数据清单、修改配置文件并运行一个命令，即可开始训练或推理。

更多features正在开发中，敬请期待。如果您有任何建议或问题，欢迎提交 Issue 或 PR。

## 项目特点

- **易于使用**：训练与推理共用统一入口 `run.py`，主要参数均通过 YAML 配置。
- **自动组织实验结果**：自动创建模型、训练状态、日志、TensorBoard 和可视化目录。
- **避免覆盖旧实验**：实验目录重名时，旧目录会自动添加时间戳并归档。
- **自动发现 Trainer 和 Dataset**：
  - `model/trainer/` 中以 `Trainer.py` 结尾的 Trainer 会被自动发现。
  - `dataloader/` 中以 `_dataset.py` 结尾的 Dataset 会被自动发现。
- **TensorBoard 日志**：训练时自动记录 batch loss、epoch loss、验证 loss 和学习率。
- **文本日志**：自动记录运行环境、完整配置、训练过程和最终评估指标。
- **灵活的评估粒度**：支持 `video`、`clip` 和 `clip_average` 三种评估方式。



## 当前支持

### 数据集预处理

- PURE
- UBFC-rPPG
- BUAA

### H5 Dataset

- `H5Dataset`：每次从一个 H5 文件中随机采样一个片段。
- `H5ClipOrderDataset`：按照时间顺序读取一个 H5 文件中的全部完整片段。

### 评估指标

- ME：Mean Error
- STD：误差标准差
- MAE：Mean Absolute Error
- RMSE：Root Mean Square Error
- MER：Mean Error Rate
- P：Pearson 相关系数

## 项目结构

```text
Easy-rPPG-open_source/
├── configs/
│   ├── train/                  # 训练配置
│   └── infer/                  # 推理配置
├── dataloader/                 # H5 Dataset 与 DataLoader
├── dataset/                    # H5 路径清单
├── evaluate/                   # 心率计算与评估指标
├── model/
│   ├── loss/                   # 损失函数
│   ├── network/                # 网络结构
│   ├── trainer/                # 训练、验证和测试流程
│   └── utils/                  # 日志、配置和分布式工具
├── preprocess/
│   ├── script/                 # OpenFace 关键点提取与 H5 预处理
│   ├── extract_landmarks.sh
│   └── preprocess.sh
├── run.py                      # 统一运行入口
├── run.sh                      # 训练命令示例
└── test.sh                     # 推理命令示例
```

## 1. 安装环境

建议使用 Linux、Python 3.10 或更高版本，以及支持 CUDA 的 PyTorch 环境。

```bash
conda create -n easy-rppg python=3.10 -y
conda activate easy-rppg
```

先根据本机 CUDA 版本安装 PyTorch，再安装项目使用的 Python 依赖：

```bash
pip install numpy scipy h5py pandas opencv-python tqdm matplotlib tensorboard
pip install imageio pycwt scikit-learn scikit-image pyyaml
```

验证 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果仅使用 CPU，请在 YAML 中设置：

```yaml
device: cpu
num_gpu: 0
```

## 2. 准备原始数据

请自行申请并下载对应数据集。建议将原始数据与项目代码分开存放，例如：

```text
/path/to/data/
├── PURE/
├── UBFC-rPPG/
└── BUAA/
```

预处理阶段需要使用 OpenFace 提取人脸关键点。请先安装 OpenFace，并确认 `FeatureExtraction` 可执行文件可以正常运行。

## 3. 提取 OpenFace 关键点

可以直接调用 Python 脚本。以下为 PURE 示例：

```bash
python preprocess/script/extract_openface_landmarks.py \
  --mode pure \
  --openface_bin /path/to/OpenFace/build/bin/FeatureExtraction \
  --input_root /path/to/data/PURE \
  --output_root /path/to/processed/PURE \
  --two_d_only \
  --skip_existing
```

普通视频数据集示例：

```bash
python preprocess/script/extract_openface_landmarks.py \
  --mode video \
  --openface_bin /path/to/OpenFace/build/bin/FeatureExtraction \
  --input_root /path/to/data/UBFC-rPPG \
  --output_root /path/to/processed/UBFC-rPPG \
  --pattern "vid.avi" \
  --recursive \
  --two_d_only \
  --skip_existing
```

常用参数：

- `--mode pure`：处理 PURE 图像序列。
- `--mode video`：处理普通视频文件。
- `--pattern`：视频文件名匹配规则。
- `--recursive`：递归搜索视频。
- `--two_d_only`：只导出 2D 人脸关键点。
- `--skip_existing`：跳过已生成的关键点文件。
- `--show_openface_output`：显示 OpenFace 输出，用于排查错误。

## 4. 生成 H5 数据

预处理脚本会根据视频、关键点和真实 BVP 信号裁剪人脸，并生成 H5 文件。

PURE 示例：

```bash
python preprocess/script/preprocess.py \
  --dataset_name PURE \
  --video_dir /path/to/data/PURE \
  --json_dir /path/to/data/PURE \
  --landmark_dir /path/to/processed/PURE \
  --h5_dir /path/to/h5/PURE \
  --store_size 128
```

UBFC-rPPG 示例：

```bash
python preprocess/script/preprocess.py \
  --dataset_name UBFC-rPPG \
  --video_dir /path/to/data/UBFC-rPPG \
  --json_dir /path/to/data/UBFC-rPPG \
  --landmark_dir /path/to/processed/UBFC-rPPG \
  --h5_dir /path/to/h5/UBFC-rPPG \
  --store_size 128
```

生成的 H5 文件至少需要包含：

```text
imgs    # 人脸帧，形状通常为 [T, H, W, C]
bvp     # BVP 波形，形状通常为 [T]
```

默认情况下，DataLoader 使用 `imgs` 和 `bvp`。使用其他键名时，可在数据集配置中增加：

```yaml
img_key: your_image_key
label_key: your_bvp_key
```

## 5. 创建 H5 路径清单

训练、验证和测试集分别使用一个文本文件。每行填写一个 H5 文件路径：

```text
/path/to/h5/PURE/01-01/01-01_s128.h5
/path/to/h5/PURE/01-02/01-02_s128.h5
/path/to/h5/PURE/02-01/02-01_s128.h5
```

建议创建：

```text
dataset/train_h5_paths.txt
dataset/val_h5_paths.txt
dataset/test_h5_paths.txt
```

清单支持绝对路径，也支持相对于清单文件所在目录的相对路径。空行和以 `#` 开头的注释行会被忽略。

## 6. 配置训练

复制训练模板：

```bash
cp configs/train/train_template.yaml configs/train/my_train.yaml
```

至少需要修改以下字段：

```yaml
name: physnet_experiment
device: cuda
num_gpu: 1

train:
  epochs: 30
  lr: 5.0e-4
  use_last_epoch: false

datasets:
  fs: 30

  train:
    h5_path: ./dataset/train_h5_paths.txt
    clip_length: 128

  val:
    h5_path: ./dataset/val_h5_paths.txt
    clip_length: 128

  test:
    h5_path: ./dataset/test_h5_paths.txt
    clip_length: 360

path:
  root: .
```

重要字段说明：

| 字段 | 说明 |
| --- | --- |
| `name` | 实验名称，同时用于生成输出目录 |
| `mode` | `train_and_test` 或 `only_test` |
| `device` | 例如 `cuda`、`cuda:0` 或 `cpu` |
| `num_gpu` | 训练 DataLoader 的 GPU 数量倍率；CPU 模式设为 `0` |
| `seed` | 随机种子 |
| `train.use_last_epoch` | `true` 使用最后 epoch；`false` 使用验证集选择最佳 epoch |
| `datasets.fs` | 数据采样率 |
| `type` | `H5Dataset` 或 `H5ClipOrderDataset` |
| `clip_length` | 每个输入片段的帧数 |
| `stride` | `H5ClipOrderDataset` 的滑动步长，默认等于 `clip_length` |
| `start_offset` | 从每个 H5 的第几帧开始读取，默认 `0` |
| `inference.eval_level` | `video`、`clip` 或 `clip_average` |
| `path.root` | 实验结果根目录 |

`H5Dataset` 适合训练时随机采样；`H5ClipOrderDataset` 会按照时间顺序生成所有完整片段，适合验证和测试。

## 7. 开始训练

从项目根目录运行：

```bash
python run.py --config_file configs/train/my_train.yaml
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --config_file configs/train/my_train.yaml
```

`mode: train_and_test` 会依次执行：

1. 创建 DataLoader。
2. 训练 PhysNet。
3. 每个 epoch 保存模型和训练状态。
4. 根据配置执行验证并选择最佳模型。
5. 训练完成后自动在测试集上评估。
6. 保存预测结果和评估指标。

## 8. 查看 TensorBoard

训练时会自动写入：

- `train/loss`
- `train/lr`
- `train/loss_epoch`
- `valid/loss`

TensorBoard 日志默认保存到：

```text
experiments/<实验名称>/tensorboard/
```

启动 TensorBoard：

```bash
tensorboard --logdir experiments
```

浏览器打开终端显示的地址即可查看所有实验。TensorBoard 会自动扫描 `experiments` 下的日志目录，无需逐个指定实验。

## 9. 单独运行推理

复制推理模板：

```bash
cp configs/infer/infer_template.yaml configs/infer/my_infer.yaml
```

修改权重和测试集清单：

```yaml
name: physnet_inference
mode: only_test

inference:
  pretrain_ckpt: ./pretrained_models/physnet.pth
  eval_level: clip_average

datasets:
  fs: 30
  test:
    dataset: PURE
    h5_path: ./dataset/test_h5_paths.txt
    clip_length: 360

path:
  root: .
```

运行：

```bash
python run.py --config_file configs/infer/my_infer.yaml
```

推理结果默认保存到：

```text
results/<实验名称>/
```

## 10. 评估粒度

通过 `inference.eval_level` 设置：

```yaml
inference:
  eval_level: clip_average
```

- `video`：拼接同一视频的全部预测片段，再计算一次心率。
- `clip`：每个片段分别计算心率，并使用全部片段计算指标。
- `clip_average`：每个片段分别计算心率，再对同一视频的片段心率取平均。

## 11. 输出目录

训练输出：

```text
experiments/<实验名称>/
├── models/                       # Epoch*.pth
├── training_states/              # Epoch*.state
├── tensorboard/                  # TensorBoard event 文件
├── visualization/
│   ├── loss_curve.png
│   ├── lr_curve.png
│   └── loss_lr_log.csv
├── outputs.pickle                # 预测波形和真实波形
└── <实验名称>_<时间>.log
```

仅推理输出：

```text
results/<实验名称>/
├── outputs.pickle
└── <实验名称>_<时间>.log
```

当输出目录已经存在时，框架会将旧目录重命名为：

```text
<原目录>_archived_<时间戳>
```

## 12. 断点恢复

每个 epoch 会保存：

- `models/EpochN.pth`：模型权重。
- `training_states/EpochN.state`：epoch、模型路径、optimizer、scheduler 和最佳验证信息。

当前版本的恢复逻辑同时读取 `path.resume_state` 和 `train.resume_state`。恢复训练时，请在训练 YAML 中将两处都设置为同一个 `.state` 文件：

```yaml
train:
  epochs: 50
  resume_state: ./experiments/physnet_train/training_states/Epoch9.state

path:
  root: .
  resume_state: ./experiments/physnet_train/training_states/Epoch9.state
```

然后正常运行训练命令：

```bash
python run.py --config_file configs/train/my_train.yaml
```

`train.epochs` 表示恢复后训练结束时的总 epoch 数，必须大于状态文件中的下一个 epoch。

## 13. 扩展自定义 Dataset

在 `dataloader/` 下创建以 `_dataset.py` 结尾的文件，例如：

```text
dataloader/custom_dataset.py
```

在文件中定义 Dataset 类，并在 YAML 中填写类名：

```yaml
datasets:
  train:
    type: CustomDataset
```

框架会自动扫描并实例化该 Dataset，无需手动修改注册表。

## 14. 扩展自定义 Trainer

在 `model/trainer/` 下创建以 `Trainer.py` 结尾的文件，例如：

```text
model/trainer/CustomTrainer.py
```

定义 `CustomTrainer` 类，并在 YAML 中设置：

```yaml
trainer_type: CustomTrainer
```

框架会自动扫描并创建对应 Trainer。



## 参考

- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
- [BasicSR](https://github.com/xpixelgroup/basicsr)


## License

本项目根目录采用 MIT License。PhysNet 网络代码来源于其原始实现，使用前请同时确认原始项目和相关数据集的许可证及使用限制。
