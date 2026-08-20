# Easy-rPPG 配置参数说明

Easy-rPPG 使用 YAML 配置文件统一控制训练、推理、数据加载、评测和输出目录。

- 训练模板：`configs/train/train_template.yaml`
- 推理模板：`configs/infer/infer_template.yaml`
- 运行入口：`python run.py --config_file <配置文件路径>`

新增或修改训练配置时，顶层 `name` 必须包含当天日期，格式为 `YYYYMMDD`，例如：

```yaml
name: physnet_train_20260820
```

## 1. 配置结构

```yaml
name: physnet_train_20260820
description: PhysNet training example
trainer_type: PhysnetTrainer
mode: train_and_test

device: cuda
dist: false
num_gpu: 1
seed: 42

train: {}
inference: {}
datasets: {}
path: {}
```

`train_and_test` 模式会执行训练和测试；`only_test` 模式只加载已有权重并测试。

## 2. 顶层参数

| 参数 | 类型 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `name` | string | 是 | 实验名称，也用于生成输出目录和日志文件名。训练配置必须包含 `YYYYMMDD` 日期。 |
| `description` | string | 否 | 配置用途说明，不参与程序逻辑。 |
| `trainer_type` | string | 是 | Trainer 类名。当前公开模板使用 `PhysnetTrainer`。框架会从 `model/trainer/` 自动发现对应类。 |
| `mode` | string | 是 | `train_and_test` 或 `only_test`。 |
| `device` | string | 是 | PyTorch 设备，例如 `cuda`、`cuda:0` 或 `cpu`。 |
| `dist` | bool | 是 | 是否按分布式模式构造训练 DataLoader。当前公开 PhysNet 流程建议保持 `false`。 |
| `num_gpu` | int | 是 | 非分布式训练时用于放大训练 batch size 和 worker 数量。单 GPU 通常为 `1`，CPU 为 `0`。 |
| `seed` | int | 是 | Python、NumPy、PyTorch 和 DataLoader 的随机种子。 |

CPU 配置需要同时设置：

```yaml
device: cpu
num_gpu: 0
```

## 3. `train`：训练参数

`train` 主要在 `mode: train_and_test` 时使用。

| 参数 | 类型 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `epochs` | int | 是 | 总训练 epoch 数。恢复训练时仍表示最终总 epoch 数，而不是额外训练轮数。 |
| `lr` | float | 是 | Adam 优化器学习率，同时作为 OneCycleLR 的最大学习率。推荐写成 `!!float 5.0e-4`。 |
| `use_last_epoch` | bool | 是 | `true`：训练结束后使用最后一个 epoch 测试；`false`：根据验证损失选择最佳 epoch。 |
| `plot_losses_and_lr` | bool | 是 | 是否输出训练/验证损失曲线、学习率曲线和 CSV 日志。 |
| `save_model_frequency` | int | 否 | 保留带编号 `EpochN` checkpoint 的周期，默认 `1`；必须大于 `0`。每个 epoch 都会覆盖保存 `latest`，最后一个 epoch 始终额外保留。 |
| `resume_state` | string/null | 否 | 恢复训练使用的 `.state` 文件。未恢复时设为 `~`。只使用该最新字段，不使用 `path.resume_state`。 |

当 `use_last_epoch: false` 时，必须提供 `datasets.val`；当其为 `true` 时，可以不配置验证集。

恢复训练示例：

```yaml
train:
  epochs: 50
  resume_state: ./experiments/physnet_train_20260820/training_states/latest.state
```

## 4. `inference`：推理与评测参数

训练完成后的测试和 `only_test` 都使用本节配置。

| 参数 | 类型 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `pretrain_ckpt` | string/null | `only_test` 必需 | 仅测试模式加载的 `.pth` 模型权重；训练模式下通常为 `~`。 |
| `eval_levels` | list[string] | 是 | 按列表顺序执行一个或多个评测层级。支持 `video`、`window`、`window-average`，不可重复。 |
| `eval_window` | int | 使用 window 时必需 | 每个评测窗口的帧数，必须为整数且不少于 9 帧。 |
| `eval_method` | string | 是 | 心率后处理方法：`Dai` 或 `Toolbox`。 |
| `toolbox_hr_method` | string | Toolbox 时使用 | Toolbox 的心率估计方式：`FFT` 或 `Peak`。 |

推荐写法：

```yaml
inference:
  pretrain_ckpt: ~
  eval_levels:
    - window-average
    - video
    - window
  eval_window: 300
  eval_method: Dai
  toolbox_hr_method: FFT
```

### 4.1 评测层级

- `video`：先按 clip 序号拼接同一视频的全部预测，再对整段视频计算一次心率。
- `window`：先拼接同一视频，再按 `eval_window` 切成不重叠窗口，每个窗口分别计算心率和指标。
- `window-average`：逐窗口计算心率，再对每个视频内部的窗口心率取平均，最后计算数据集指标。

不足一个完整 `eval_window` 的尾部仍会参与评测，只要其长度不少于 9 帧；更短的尾部会被忽略。

当前配置只使用 `eval_levels`，不兼容旧字段 `eval_level`、`clip` 或 `clip_average`。

### 4.2 评测方法

- `Dai`：使用 Dai 后处理计算心率。
- `Toolbox + FFT`：使用 rPPG-Toolbox 风格带通滤波和频域峰值计算心率。
- `Toolbox + Peak`：使用 rPPG-Toolbox 风格带通滤波和时域峰值间隔计算心率。

输出指标包括 ME、STD、MAE、RMSE、MER 和 Pearson 相关系数 P。

## 5. `datasets`：数据集公共参数

`datasets` 下不属于 `train`、`val`、`test` 的字段会作为公共默认值复制到每个 phase。phase 内同名字段优先，可覆盖公共配置。

```yaml
datasets:
  data_type: [Raw]
  label_type: Raw
  normalize: true
  fs: 30
  resize:
    height: 128
    width: 128
  dataset: PURE
  clip_length: 128

  train:
    # 自动继承以上公共字段
    name: train_set
    type: H5ClipOrderDataset
    h5_path: ./dataset/train_h5_paths.txt
```

公共参数如下：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `data_type` | string/list[string] | 未设置时保持旧行为 | 视频输入表示。支持 `Raw`、`Standardized`、`DiffNormalized`。列表中多种表示会按顺序沿通道维拼接。 |
| `label_type` | string | `Raw` | BVP 标签表示。支持 `Raw`、`Standardized`、`DiffNormalized`。 |
| `normalize` | bool | `true` | 读取图像后是否除以 255。该操作在 `data_type` 变换之前执行。 |
| `fs` | int/float | 无 | 视频和 BVP 的采样率，单位 Hz，用于心率评测。 |
| `resize` | mapping/null | `null` | 模型输入空间尺寸。支持 `height/width`，也兼容 `h/w`。不设置时保留 H5 原始尺寸。 |
| `dataset` | string | `unknown` | 数据集名称，例如 `PURE`、`UBFC-rPPG`、`BUAA`。用于生成 subject ID，也可填写自定义名称。 |
| `clip_length` | int | 无 | 每个模型输入 clip 的帧数，必须大于 0。 |

### 5.1 `data_type` 与 `label_type`

名称不区分大小写，并允许下划线或连字符，例如 `diff_normalized` 会转换为 `DiffNormalized`。

- `Raw`：保留读取后的信号。
- `Standardized`：减去均值并除以标准差；常量输入会安全转换为全 0。
- `DiffNormalized`：视频使用相邻帧相对差分，标签使用一阶差分，并用 0 补齐最后一帧以保持长度不变。

多输入表示示例：

```yaml
datasets:
  data_type: [Raw, DiffNormalized]
```

每种 RGB 表示包含 3 个通道，两个表示拼接后得到 6 通道。当前 PhysNet 默认接收 3 通道，因此使用 PhysNet 时通常只配置一种 `data_type`；多表示拼接需要模型本身支持相应输入通道数。

### 5.2 公共配置覆盖

```yaml
datasets:
  clip_length: 128
  dataset: PURE

  train:
    # 继承 clip_length=128、dataset=PURE
    name: train_set

  test:
    # 仅测试阶段覆盖为 300 帧
    clip_length: 300
    name: test_set
```

`phase` 字段由配置解析器根据 `train`、`val`、`test` 键自动添加，不需要手动填写。

## 6. `train`、`val`、`test` phase 参数

每个 phase 可以使用以下参数：

| 参数 | 类型 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `name` | string | 是 | DataLoader 日志中显示的数据集实例名称。 |
| `type` | string | 是 | Dataset 类名。当前支持 `H5Dataset` 和 `H5ClipOrderDataset`。 |
| `h5_path` | string | 是 | H5 路径清单文件；每行一个 H5 文件路径。 |
| `batch_size_per_gpu` | int | 是 | 每个 GPU 的 batch size。非分布式训练阶段会乘以 `num_gpu`；验证和测试阶段直接使用该值。 |
| `num_workers_per_gpu` | int | 是 | 每个 GPU 的 DataLoader worker 数。非分布式训练阶段会乘以 `num_gpu`；验证和测试阶段直接使用该值。 |
| `shuffle` | bool | train 必需 | 是否打乱训练数据。验证和测试固定为 `false`，无需配置。 |
| `clip_length` | int | 可继承 | 当前 phase 的 clip 长度，可覆盖 `datasets.clip_length`。 |
| `dataset` | string | 可继承 | 当前 phase 的数据集名称，可覆盖 `datasets.dataset`。 |
| `start_offset` | int | 否 | 从每个 H5 的第几帧开始取样，默认 `0`，必须大于等于 0。 |
| `img_key` | string | 否 | H5 中图像数组的键名，默认 `imgs`。 |
| `label_key` | string | 否 | H5 中 BVP 标签的键名，默认 `bvp`。 |
| `stride` | int | H5ClipOrderDataset 可选 | 相邻 clip 起点间隔，默认等于 `clip_length`，必须大于 0。 |

### 6.1 `H5Dataset`

每次访问一个 H5 文件时随机采样一个 clip，数据集长度等于 H5 文件数量，适合训练时随机取样。

### 6.2 `H5ClipOrderDataset`

按照时间顺序从每个 H5 中生成全部完整 clip。默认 `stride == clip_length`，clip 之间不重叠；设置较小 stride 可产生重叠 clip。

验证和测试通常使用 `H5ClipOrderDataset`，因为评测代码会按照 subject ID 和 clip 序号重新拼接完整视频。

### 6.3 H5 路径清单

```text
/absolute/path/to/video_01.h5
../relative/path/to/video_02.h5
# 注释行会被忽略
```

- `h5_path` 本身的相对路径相对于运行 `run.py` 时的当前目录解析，建议从项目根目录运行。
- 清单中的相对 H5 路径相对于清单文件所在目录解析。
- 空行和以 `#` 开头的行会被忽略。
- 每个 H5 至少需要包含图像和标签两个键，默认分别为 `imgs` 和 `bvp`。
- 图像数组预期形状为 `[T, H, W, C]`，标签数组预期形状为 `[T]`。
- 当图像与标签长度不一致时，使用二者的较短长度。
- H5 有效长度必须不少于 `start_offset + clip_length`。

## 7. `path`：输出目录

模板只需要设置：

```yaml
path:
  root: .
```

### 7.1 训练模式

当 `root: .` 时，框架自动生成：

```text
experiments/<name>/
├── models/
├── training_states/
├── tensorboard/
├── visualization/
├── outputs.pickle
└── <name>_<时间戳>.log
```

### 7.2 仅测试模式

当 `root: .` 时，框架自动生成：

```text
results/<name>/
├── visualization/
├── outputs.pickle
└── <name>_<时间戳>.log
```

如果 `path.root` 自身已经位于名为 `experiments` 或 `results` 的目录层级中，框架会直接使用 `<root>/<name>`，避免重复生成 `experiments/experiments` 或 `results/results`。

非恢复运行时，如果目标实验目录已存在，旧目录会被重命名为 `<原目录>_archived_<时间戳>`。框架还会把本次 YAML 配置备份到输出目录中。

## 8. 模式与必需数据集组合

### 8.1 训练并测试

```yaml
mode: train_and_test
```

- 必须配置 `datasets.train` 和 `datasets.test`。
- `train.use_last_epoch: false` 时还必须配置 `datasets.val`。
- `inference.pretrain_ckpt` 通常设为 `~`。

### 8.2 仅测试

```yaml
mode: only_test
```

- 只需要配置 `datasets.test`。
- 必须设置有效的 `inference.pretrain_ckpt`。
- 不需要 `train` 段。

## 9. 完整训练配置示例

```yaml
name: physnet_train_20260820
description: PhysNet training example
trainer_type: PhysnetTrainer
mode: train_and_test

device: cuda
dist: false
num_gpu: 1
seed: 42

train:
  epochs: 30
  lr: !!float 5.0e-4
  use_last_epoch: false
  plot_losses_and_lr: true
  save_model_frequency: 1
  resume_state: ~

inference:
  pretrain_ckpt: ~
  eval_levels: [window-average, video, window]
  eval_window: 300
  eval_method: Dai
  toolbox_hr_method: FFT

datasets:
  data_type: [Raw]
  label_type: Raw
  normalize: true
  fs: 30
  resize:
    height: 128
    width: 128
  dataset: PURE
  clip_length: 128

  train:
    name: train_set
    type: H5ClipOrderDataset
    h5_path: ./dataset/train_h5_paths.txt
    batch_size_per_gpu: 2
    num_workers_per_gpu: 4
    shuffle: true

  val:
    name: val_set
    type: H5ClipOrderDataset
    h5_path: ./dataset/val_h5_paths.txt
    batch_size_per_gpu: 2
    num_workers_per_gpu: 4

  test:
    name: test_set
    type: H5ClipOrderDataset
    h5_path: ./dataset/test_h5_paths.txt
    batch_size_per_gpu: 2
    num_workers_per_gpu: 4

path:
  root: .
```

## 10. 完整推理配置示例

```yaml
name: physnet_inference_20260820
description: PhysNet inference example
trainer_type: PhysnetTrainer
mode: only_test

device: cuda
dist: false
num_gpu: 1
seed: 42

inference:
  pretrain_ckpt: ./pretrained_models/physnet.pth
  eval_levels: [window-average, video, window]
  eval_window: 300
  eval_method: Dai
  toolbox_hr_method: FFT

datasets:
  data_type: [Raw]
  label_type: Raw
  normalize: true
  fs: 30
  resize:
    height: 128
    width: 128
  dataset: PURE
  clip_length: 128

  test:
    name: test_set
    type: H5ClipOrderDataset
    h5_path: ./dataset/test_h5_paths.txt
    batch_size_per_gpu: 2
    num_workers_per_gpu: 4

path:
  root: .
```

## 11. 常见错误

### `inference.eval_levels is required`

必须使用列表形式的新字段：

```yaml
eval_levels: [video, window]
```

### `H5 list file does not exist`

确认 `h5_path` 存在，并从项目根目录运行命令；清单内部的相对路径则相对于清单文件解析。

### `H5 files are shorter than start_offset + clip_length`

减小 `clip_length` 或 `start_offset`，或者检查预处理后的视频和标签长度。

### 模型输入通道数不匹配

检查 `data_type`。PhysNet 默认输入为 3 通道，配置多个表示会增加输入通道数。

### 恢复训练后立即报 epochs 已完成

`train.epochs` 必须大于恢复状态中的下一个 epoch。
