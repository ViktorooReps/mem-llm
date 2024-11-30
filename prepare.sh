pip install -r requirements.txt

# as of 25/11/2024, to enable FlexAttention, you need a nightly version
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# download the datasets

python -c "
from mem_llm.dataset import load_dataset, TSConfig, FineWebEduConfig
from mem_llm.tokenizer import CharTokenizer, TikTokenTokenizer

arg1 = '$1'
if arg1 == 'char':
    tokenizer = CharTokenizer.from_config('configs/char_tokenizer.json')
elif arg1 == 'token':
    tokenizer = TikTokenTokenizer.from_config('configs/gpt2_tokenizer.json')
else:
    raise ValueError(f'Unknown tokenizer type: {arg1}')

load_dataset(TSConfig(example_length=1), tokenizer)
load_dataset(FineWebEduConfig(example_length=1), tokenizer)
"