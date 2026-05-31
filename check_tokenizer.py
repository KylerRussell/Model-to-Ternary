#!/usr/bin/env python3
"""Quick check of Qwen3.6-27B tokenizer hash for GGUF conversion."""
import hashlib
from transformers import AutoTokenizer

MODEL_PATH = '/home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# This is the exact test string from convert_hf_to_gguf.py
chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

chktok = tok.encode(chktxt)
chkhsh = hashlib.sha256(str(chktok).encode()).hexdigest()
print(f'Qwen3.6-27B tokenizer hash: {chkhsh}')
print(f'Tokenizer class: {type(tok).__name__}')
print(f'Vocab size: {tok.vocab_size}')

# The existing qwen35 hash in the converter is for Qwen3.5-9B-Instruct
# Qwen3.6-27B likely uses a slightly different pre-tokenizer hence different hash
# But functionally it's the same "qwen35" tokenizer family
