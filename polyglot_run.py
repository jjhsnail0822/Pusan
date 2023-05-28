import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/polyglot-ko-1.3b')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-1.3b')
model.eval()
model.to(device)

with torch.no_grad():
    c = ''
    while True:
        inp = input('USER> ')
        context = inp.strip()
        ids = tokenizer.encode(context, return_tensors='pt').to(device)
        gen = model.generate(ids, do_sample=False, num_beams=3, no_repeat_ngram_size=3, min_length=len(context), max_length=512)
        ans = tokenizer.decode(gen[0])
        c = ans
        print('BOT > ' + ans)
        print('')