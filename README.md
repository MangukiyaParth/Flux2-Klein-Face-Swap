---
title: Flux2 Klein Face Swap
emoji: 🦀
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.5.1
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Face Swap app using Flux.2 Klein 9B LoRA
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

rm -rf ~/.cache/huggingface
rm -rf /workspace/*

mkdir -p /workspace/huggingface-cache

mkdir flux-project
cd flux-project

export HF_HOME=/workspace/huggingface-cache
export TRANSFORMERS_CACHE=/workspace/huggingface-cache
export HF_DATASETS_CACHE=/workspace/huggingface-cache

echo 'export HF_HOME=/workspace/huggingface-cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/huggingface-cache' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/huggingface-cache' >> ~/.bashrc
source ~/.bashrc

git clone https://github.com/MangukiyaParth/Flux2-Klein-Face-Swap .

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1