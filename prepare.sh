pip install -r requirements.txt

# as of 25/11/2024, to enable FlexAttention, you need a nightly version
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# download the datasets

python -c "from char_llm.dataset import load_dataset, TSConfig, FineWebEduConfig; load_dataset(TSConfig(target_length=1)); load_dataset(FineWebEduConfig(target_length=1))"