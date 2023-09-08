import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import pickle

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    load_dotenv()

    PREFIX_AI = os.environ.get('PREFIX_AI')
    PREFIX_USER = os.environ.get('PREFIX_USER')
    TEMPLATE = os.environ.get('TEMPLATE')
    EOS = os.environ.get('EOS')
    MODEL_ID = os.environ.get('MODEL_ID')
    PEFT_ID = os.environ.get('PEFT_ID')
    AI_NAME = os.environ.get('AI_NAME')
    PKL_PATH = os.environ.get('PKL_PATH')
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
    ACC_STEPS = int(os.environ.get('ACC_STEPS'))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE'))
    LOGGING_STEPS = int(os.environ.get('LOGGING_STEPS'))
    LR_SCHEDULER_TYPE = os.environ.get('LR_SCHEDULER_TYPE')
    STEPS = int(os.environ.get('STEPS'))
    LORA_R = int(os.environ.get('LORA_R'))
    LORA_ALPHA = int(os.environ.get('LORA_ALPHA'))
    LORA_DROPOUT = float(os.environ.get('LORA_DROPOUT'))
    SAVE_STEPS = int(os.environ.get('SAVE_STEPS'))
    NUM_WORKERS = int(os.environ.get('NUM_WORKERS'))
    CONTINUE_TRAINING = int(os.environ.get('CONTINUE_TRAINING'))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map=device)

    with open(PKL_PATH, 'rb') as f:
        dataset = pickle.load(f)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if CONTINUE_TRAINING == 1: # 1 is True, 0 is False
        model = PeftModel.from_pretrained(model, PEFT_ID, is_trainable=True)
    else:
        model = get_peft_model(model, config)

    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=ACC_STEPS,
            max_steps=STEPS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            logging_steps=LOGGING_STEPS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            save_steps=SAVE_STEPS,
            dataloader_num_workers=NUM_WORKERS,
            output_dir=PEFT_ID
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    if CONTINUE_TRAINING == 1:
        trainer.train(resume_from_checkpoint=PEFT_ID)
    else:
        trainer.train()

    trainer.save_model(PEFT_ID)