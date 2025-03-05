# 基本环境

```bash
#  第一种办法
# Clone and checkout pytorch 2.6 release candidate
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.6.0-rc9
git submodule sync
git submodule update --init --recursive -j 8

# Install build dependencies (assumes you already have a system compiler)
pip install -r requirements.txt
pip install mkl-static mkl-include wheel

# Build PyTorch (will take a long time)
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=%CUDA_HOME%
set TORCH_CUDA_ARCH_LIST=Blackwell
python setup.py develop

# Optional, package build into a wheel to install on other machines.
python setup.py bdist_wheel
ls dist  # Wheel should be output in this directory
```

```bash
#  第二种办法
git clone https://github.com/triton-lang/triton.git
cd triton

pip install ninja cmake wheel pybind11 # build-time dependencies
pip install -e python
```

```bash
# 安装别人编译好的triton https://huggingface.co/madbuda/triton-windows-builds
pip install pytorch-triton --no-deps
```

```bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"

# 安装unsloth
pip install unsloth

# 升级unsloth
pip install --upgrade unsloth --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# 安装一些小工具

# 安装torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# bitsandbytes: 一个用于量化和压缩的库
pip install bitsandbytes

# unsloth_zoo: 一个用于unsloth的工具库
pip install unsloth_zoo
```

```bash
# 查看unsloth帮助
unsloth --help
```

## 预训练

下载预训练模型

```python

```
