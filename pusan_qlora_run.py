import os
from dotenv import load_dotenv
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

load_dotenv()

PREFIX_AI = '<|unused0|>'
PREFIX_USER = '<|unused1|>'
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

def generate_ai_chat(context, user_input):
    context = context + PREFIX_USER + user_input.strip() + '\n' + PREFIX_AI
    gened = model.generate(
        **tokenizer(
            context, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to(device),
        max_new_tokens=16,
        early_stopping=True,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    gened = tokenizer.decode(gened[0]).rstrip(EOS)
    gened = gened[len(context):]
    endidx = gened.find(PREFIX_USER)
    if endidx != -1:
        gened = gened[:endidx]
    return context + gened, gened.rstrip() + '\n'

def chat():
    context = ''
    while True:
        user_input = input('USER> ')
        if user_input == 'exit':
            break
        context, ai_output = generate_ai_chat(context, user_input)
        print('\nAI  > ' + ai_output)

chat()