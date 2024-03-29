import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

load_dotenv()

PREFIX = os.environ.get('PREFIX')
SUFFIX = os.environ.get('SUFFIX')
CHAT_DELIMITER = os.environ.get('CHAT_DELIMITER')
CHAT_TEMPLATE = os.environ.get('CHAT_TEMPLATE')
EOS = os.environ.get('EOS')
AI_NAME = os.environ.get('AI_NAME')
USER_NAME = os.environ.get('USER_NAME')
MODEL_ID = os.environ.get('MODEL_ID')
PEFT_ID = os.environ.get('PEFT_ID')
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS'))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = torch.device("cuda:0" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
config = PeftConfig.from_pretrained(PEFT_ID)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map=device)
model = PeftModel.from_pretrained(model, PEFT_ID)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.to(device)
model.eval()
model.config.use_cache = True

def generate_ai_chat(context, user_input):
    if user_input:
        context = context + PREFIX + USER_NAME + SUFFIX + user_input.strip() + CHAT_DELIMITER + PREFIX + AI_NAME + SUFFIX
    else:
        context = context + PREFIX + AI_NAME + SUFFIX
    gened = model.generate(
        **tokenizer(
            context, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to(device),
        max_new_tokens=MAX_NEW_TOKENS,
        early_stopping=True,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    gened = tokenizer.decode(gened[0]).rstrip(EOS)
    gened = gened[len(context):]
    endidx = gened.find(PREFIX[:2]) # find ## in the prompt
    if endidx != -1:
        gened = gened[:endidx]
    return context + gened.rstrip() + CHAT_DELIMITER, gened.rstrip()

def chat():
    context = CHAT_TEMPLATE
    while True:
        user_input = input('USER> ')
        if user_input == 'exit':
            break
        elif user_input == 'reset':
            print('-------------------------')
            context = CHAT_TEMPLATE
            continue
        context, ai_output = generate_ai_chat(context, user_input)
        print('AI  > ' + ai_output)

chat()