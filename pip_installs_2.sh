pip install omegaconf
pip install Pillow
pip install imageio
pip install numpy
pip install pyarrow
pip install transformers
pip install safetensor
pip install einops
pip install torch==2.7.0
pip install torchvision==0.22.0

# flash attention
# pip install ~/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install ~/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash-linear-attention

pip install datasets==4.5.0
pip install tyro
pip install matplotlib
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .[longbench]
pip install -e .[niah]