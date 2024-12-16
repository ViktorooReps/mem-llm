from mem_llm.hf_models.modeling_llama import LlamaForCausalLM
from mem_llm.model import MemLLM, TestModel
from mem_llm.tokenizer import CharTokenizer, TikTokenTokenizer, HfTokenizer

TOKENIZERS = {
    CharTokenizer.TYPE: CharTokenizer,
    TikTokenTokenizer.TYPE: TikTokenTokenizer,
    HfTokenizer.TYPE: HfTokenizer,
}

MODELS = {
    'hf_llama': LlamaForCausalLM,
    'custom': MemLLM,
    'test': TestModel
}
