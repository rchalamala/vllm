wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install -y \
    cuda-toolkit-12-8 \
    nvidia-cuda-toolkit \
    libnccl2 \
    libnccl-dev

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install -y nsight-systems

sudo sysctl -w kernel.perf_event_paranoid=0


echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export VLLM_WORKER_MULTIPROC_METHOD=spawn' >> ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh

echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

source ~/.bashrc

sudo apt install ccache -y

uv venv --python 3.12 --seed
source .venv/bin/activate

VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL=1 uv pip install -U -e . --torch-backend=auto

uv pip install -r requirements/build.txt --torch-backend=auto
python tools/generate_cmake_presets.py

cmake --preset release
cmake --build --preset release --target install

vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --served-model-name qwen3-next


nsys profile -o report.nsys-rep \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
vllm bench latency \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 8 \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 8 \
    --input-len 512 \
    --output-len 32 \
