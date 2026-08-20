# Easy-rPPG

Easy-rPPG 是一个基于 PyTorch 的轻量级 rPPG 训练与推理框架，提供数据预处理、H5 数据加载、模型训练、断点恢复、推理和心率评测流程。

当前公开版本以 PhysNet 为主要示例模型，使用 YAML 文件统一管理实验配置。

## 主要功能

- 统一的训练与推理入口
- OpenFace 和 MediaPipe 两种预处理流程
- H5 视频与 BVP 数据加载
- Raw、Standardized、DiffNormalized 数据表示
- video、window、window-average 多层级评测
- Dai 和 Toolbox 心率后处理
- TensorBoard、日志、checkpoint 和结果保存

## 安装

建议使用 Linux 和 Python 3.10。

~~~bash
conda create -n easy-rppg python=3.10 -y
conda activate easy-rppg
~~~

先根据 CUDA 版本安装 PyTorch，再安装其他依赖：

~~~bash
pip install numpy scipy h5py pandas opencv-python tqdm matplotlib tensorboard
pip install imageio pycwt scikit-learn scikit-image pyyaml
~~~

使用 MediaPipe 预处理时还需要：

~~~bash
pip install mediapipe
~~~

使用 OpenFace 预处理时，需要另外安装 OpenFace，并准备 FeatureExtraction 可执行文件。

## 数据预处理

预处理脚本支持 PURE、UBFC-rPPG 和 BUAA。运行前请把 Bash 文件中的 /path/to/... 占位路径修改为本机路径。

### MediaPipe

~~~bash
NUM_WORKERS=4 bash preprocess/preprocess_mediapipe_masks.sh UBFC-rPPG
~~~

入口文件：

- preprocess/preprocess_mediapipe_masks.sh
- preprocess/script/preprocess_mediapipe_masks.py

### OpenFace

~~~bash
OPENFACE_BIN=/path/to/OpenFace/build/bin/FeatureExtraction \
NUM_WORKERS=4 \
bash preprocess/preprocess_openface.sh PURE
~~~

入口文件：

- preprocess/preprocess_openface.sh
- preprocess/script/preprocess_openface.py

两个 Bash 脚本都会在预处理完成后调用 preprocess/script/genetate_h5_txt.py，生成 H5 路径清单。

生成的 H5 文件至少应包含：

~~~text
imgs    # [T, H, W, C]
bvp     # [T]
~~~

## 准备 H5 清单

训练、验证和测试分别使用一个文本清单，每行填写一个 H5 文件路径：

~~~text
/path/to/h5/video_01.h5
/path/to/h5/video_02.h5
/path/to/h5/video_03.h5
~~~

清单支持绝对路径，也支持相对于清单文件所在目录的相对路径。

## 训练

复制并修改训练模板：

~~~bash
cp configs/train/train_template.yaml configs/train/my_train.yaml
~~~

至少需要修改实验名称和 train/val/test 的 H5 清单路径。训练配置的 name 必须包含 YYYYMMDD 日期。

运行：

~~~bash
python run.py --config_file configs/train/my_train.yaml
~~~

指定 GPU：

~~~bash
CUDA_VISIBLE_DEVICES=0 python run.py --config_file configs/train/my_train.yaml
~~~

## 推理

复制并修改推理模板：

~~~bash
cp configs/infer/infer_template.yaml configs/infer/my_infer.yaml
~~~

填写模型权重和测试集 H5 清单后运行：

~~~bash
python run.py --config_file configs/infer/my_infer.yaml
~~~

完整配置参数说明见 [configs/README.md](configs/README.md)。

## 输出目录

训练结果默认保存在：

~~~text
experiments/<name>/
├── models/
├── training_states/
├── tensorboard/
├── visualization/
├── outputs.pickle
└── <name>_<时间戳>.log
~~~

仅推理结果默认保存在：

~~~text
results/<name>/
├── visualization/
├── outputs.pickle
└── <name>_<时间戳>.log
~~~

## 项目结构

~~~text
Easy-rPPG/
├── configs/       # 训练、推理模板和参数说明
├── dataloader/    # H5 Dataset 与 DataLoader
├── evaluate/      # 心率后处理和评测指标
├── model/         # 网络、损失、Trainer 和工具
├── preprocess/    # OpenFace 与 MediaPipe 预处理
├── run.py         # 统一入口
├── run.sh
└── test.sh
~~~

## 参考

- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
- [BasicSR](https://github.com/xpixelgroup/basicsr)

## License

项目根目录采用 MIT License。模型代码和数据集可能具有各自的许可证，使用前请确认对应条款。
