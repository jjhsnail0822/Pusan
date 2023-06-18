import os
from dotenv import load_dotenv
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

load_dotenv()

PREFIX_AI = '### AI:'
PREFIX_USER = '### USER:'
EOS = '<|endoftext|>'
MODEL_ID = 'EleutherAI/polyglot-ko-5.8b'
PEFT_ID = 'pusan_qlora'
AI_NAME = os.environ.get('AI_NAME')
TXT_FILEPATH = os.environ.get('TXT_FILEPATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

if platform.system() == 'Darwin':
    TXT_FILEPATH = os.environ.get('TXT_FILEPATH_MAC')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = PeftConfig.from_pretrained(PEFT_ID)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map=device)
model = PeftModel.from_pretrained(model, PEFT_ID)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.to(device)
model.eval()
model.config.use_cache = True

def gen(x):
    gened = model.generate(
        **tokenizer(
            f"### USER:{x}\n### AI:", 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to(device),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))