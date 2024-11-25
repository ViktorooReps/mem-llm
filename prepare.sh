pip install requirements.txt

# as of 25/11/2024, to enable FlexAttention, you need a nightly version
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# download the datasets

python -m "from char_llm.dataset import load_dataset, TSConfig, FineWebEduConfig; load_dataset(TSConfig()); load_dataset(FineWebEduConfig())"