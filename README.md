# 基本环境

```bash
# install wget 
sudo apt-get -y install wget -y

# 安装cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# 安装cudnn
sudo apt-get -y install libcudnn-dev

# 安装pytorch, 使用cuda12.4,需要2.5版本的
pip install torch===2.5.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装unsloth
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

pip install unsloth==2025.2.14 unsloth_zoo==2025.2.7

conda create --name unsloth_121 \
    python=3.11 -y
conda activate unsloth_121


   55  2025-03-07 15:43:10 conda env list
   56  2025-03-07 15:43:26 conda env remove unsloth_121 -y
   57  2025-03-07 15:43:33 conda env --help
   58  2025-03-07 15:43:41 conda env remove -n unsloth_121 -y
   59  2025-03-07 15:43:50 conda env --help
   60  2025-03-07 15:43:55 conda env list
   61  2025-03-07 15:44:01 conda env remove -n unsloth_env -y
   62  2025-03-07 15:44:09 conda env list
   63  2025-03-07 15:44:20 conda env remove -n unsloth_124 -y
   64  2025-03-07 15:44:29 conda env list
   65  2025-03-07 15:44:33 conda clean -a
   66  2025-03-07 15:44:47 conda clean -a -y
   67  2025-03-07 15:44:49 history

conda create --name unsloth_121 \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_121

pip install unsloth
```