import os
from dotenv import load_dotenv
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

load_dotenv()

Q_TKN = '<usr>'
A_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
UNK = '<unk>'
MODEL_PATH = os.environ.get('MODEL_PATH')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2', bos_token=BOS, eos_token=EOS, unk_token=UNK, pad_token=PAD, mask_token=MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model.to(device)

with torch.no_grad():
    c = ''
    while True:
        inp = input('USER> ')
        q = Q_TKN + ' ' + inp.strip() + A_TKN
        context = c + q
        ids = tokenizer.encode(context, return_tensors='pt').to(device)
        gen = model.generate(ids, do_sample=False, num_beams=3, no_repeat_ngram_size=3, min_length=len(context), max_length=128)
        ans = tokenizer.decode(gen[0]).rstrip(EOS)
        ans = ans[ans.rfind(A_TKN) + len(A_TKN):].lstrip()
        c = ans
        ans = ans.replace('\n','\nBOT > ')
        print('BOT > ' + ans)
        print('')