pip install -r requirements.txt

# as of 25/11/2024, to enable FlexAttention, you need a nightly version
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# download the datasets

python -c "from mem_llm.dataset import load_dataset, TSConfig, FineWebEduConfig; load_dataset(TSConfig(example_length=1)); load_dataset(FineWebEduConfig(example_length=1))"