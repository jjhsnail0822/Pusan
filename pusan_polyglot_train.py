import os
from dotenv import load_dotenv
import kakaotalkparser
import platform
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import AdamW

load_dotenv()

Q_TKN = '<usr>'
A_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
UNK = '<unk>'
AI_NAME = os.environ.get('AI_NAME')
TXT_FILEPATH = os.environ.get('TXT_FILEPATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

# for ?gb vram
MAX_LEN = 64
BATCH_SIZE = 64
EPOCH = 20

if platform.system() == 'Darwin':
    TXT_FILEPATH = os.environ.get('TXT_FILEPATH_MAC')

p = kakaotalkparser.Parser()
plist = p.parse_lines(p.txtreadlines(TXT_FILEPATH))
dataset = p.create_merged_dataset(plist)

device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/polyglot-ko-1.3b', bos_token=BOS, eos_token=EOS, unk_token=UNK, pad_token=PAD, mask_token=MASK)
model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-1.3b')

# preprocess chatbot data for the dataset with the context
class ContextedDataset(Dataset):
    def __init__(self, dataset, ai_name, max_len=64):
        self.dataset = dataset
        self.chatlist = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.ai_name = ai_name
        halflen = int(self.max_len/2)
        self.newline = tokenizer.tokenize('\n')[0] # '▁\n'

        # very first contexts
        self.dataset.insert(0, {'speaker': '', 'text': ''})
        self.dataset.insert(1, {'speaker': '', 'text': ''})
        
        for idx, data in tqdm(enumerate(dataset[2:])):
            if data['speaker'] == self.ai_name:
                a_token = self.tokenizer.tokenize(A_TKN + data['text'] + EOS)
                a_len = len(a_token)
                if a_len > halflen:
                    a_token = a_token[:halflen]
                    newline_found = self.find_newline_pos(a_token, fromright=True)
                    if newline_found != -1:
                        a_token = a_token[:newline_found+1]
                    a_len = len(a_token)
                    a_token[-1] = EOS

                q_token = self.tokenizer.tokenize(Q_TKN + dataset[idx+1]['text'])
                q_len = len(q_token)
                if q_len > halflen:
                    q_token = q_token[-(halflen):]
                    newline_found = self.find_newline_pos(q_token, fromright=False)
                    if newline_found != -1:
                        q_token = q_token[newline_found:]
                    q_len = len(q_token)
                    q_token[0] = Q_TKN

                c_token = self.tokenizer.tokenize(dataset[idx]['text'])
                c_len = len(c_token)
                c_len_available = self.max_len - q_len - a_len
                if c_len_available == 0:
                    c_len = 0
                    c_token = []
                elif c_len > c_len_available:
                    c_token = c_token[-(c_len_available):]
                    newline_found = self.find_newline_pos(c_token, fromright=False)
                    if newline_found != -1:
                        c_token = c_token[newline_found:]
                    c_len = len(c_token)
                
                tokens_len = c_len + q_len + a_len
                input_ids = self.tokenizer.convert_tokens_to_ids(c_token + q_token + a_token)
                while len(input_ids) < self.max_len:
                    input_ids += [self.tokenizer.pad_token_id]
                attention_mask = [1] * (tokens_len) + [0] * (self.max_len - tokens_len)
                labels = input_ids
                self.chatlist.append((input_ids, attention_mask, labels))
        
        self.len = len(self.chatlist)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.chatlist[idx]
    
    def find_newline_pos(self, tokenlist, fromright=False):
        if '\n' in tokenlist:
            if fromright:
                tokenlist.reverse()
                i = tokenlist.index('\n')
                tokenlist.reverse()
                return len(tokenlist) - i - 1
            else:
                return tokenlist.index('\n')
        elif self.newline in tokenlist:
            if fromright:
                tokenlist.reverse()
                i = tokenlist.index(self.newline)
                tokenlist.reverse()
                return len(tokenlist) - i - 1
            else:
                return tokenlist.index(self.newline)
        else:
            return -1

def collate_batch(batch):
    input_ids = [i[0] for i in batch]
    attention_mask = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(labels)

# batch_size=48, max_len=64
train_set = ContextedDataset(dataset, AI_NAME, max_len=MAX_LEN)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=collate_batch)

class GenerationTask(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.learning_rate = 0.0001
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    
    def training_step(self, inputs, batch_idx):
        input_ids, attention_mask, labels = [t for t in inputs]
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return outputs.loss
    
accelerator = None
devices = None

if torch.cuda.is_available():
    accelerator = 'gpu'
    devices = 1
elif torch.backends.mps.is_available():
    accelerator = 'mps'
    devices = 1


task = GenerationTask(model)
trainer = Trainer(max_epochs=EPOCH, accelerator=accelerator, devices=devices)
trainer.fit(task, train_dataloaders=train_dataloader)

torch.save(model.state_dict(), MODEL_PATH)
