import os
from dotenv import load_dotenv
import kakaotalkparser
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

load_dotenv()

PREFIX_AI = '### AI:'
PREFIX_USER = '### USER:'
EOS = '<|endoftext|>'
MODEL_ID = 'EleutherAI/polyglot-ko-5.8b'
PEFT_ID = 'pusan_qlora'
AI_NAME = os.environ.get('AI_NAME')
TXT_FILEPATH = os.environ.get('TXT_FILEPATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

# for 12gb vram
BATCH_SIZE = 2
STEPS = 1000

if platform.system() == 'Darwin':
    TXT_FILEPATH = os.environ.get('TXT_FILEPATH_MAC')

p = kakaotalkparser.Parser()
plist = p.parse_lines(p.txtreadlines(TXT_FILEPATH))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map=device)
dataset = p.create_context_dataset(plist, AI_NAME, tokenizer)
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=STEPS,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir=PEFT_ID
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

trainer.save_model(PEFT_ID)