import re
import glob
import os
import natsort
import datetime as dt
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import pickle

# parse txt file(s) saved by kakaotalk exportation
class Parser():
    # concatenate txt files in the folder and save the result
    def mergetxt(self, path, mergedfilepath):
        if os.path.exists(mergedfilepath):
            os.remove(mergedfilepath)
        filelist = natsort.natsorted(glob.glob(os.path.join(path, '*.txt')))
        with open(mergedfilepath, 'w', encoding='UTF-8-sig') as mergedfile:
            for filename in filelist:
                with open(filename, encoding='UTF-8-sig') as file:
                    mergedfile.write(file.read())
                    mergedfile.write('\n')
                    print(filename)
        return

    # return a list containing all raw lines in a txt file
    # strip most right \n
    def txtreadlines(self, filepath):
        lines = []
        with open(filepath, 'r', encoding='UTF-8-sig') as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.strip(), lines))
        return lines

    # return parsed kakaotalk dialogs list
    def parse_lines(self, list):
        parsedlist = []
        # nickname is 3 hangul letters only
        MATCHSTR = r'^(\d{4})[.] (\d{1,2})[.] (\d{1,2})[.] (오[전후]) (\d{1,2}):(\d{2}), ([가-힣]{3}) :'
        SYSTEMSTRLIST = [
            r'^Talk_\d{4}[.]\d{1,2}[.]\d{1,2} \d{1,2}:\d{2}-\d{1,3}[.]txt$',
            r'^저장한 날짜 : \d{4}[.] \d{1,2}[.] \d{1,2}[.] 오[전후] \d{1,2}:\d{2}$',
            r'^\d{4}년 \d{1,2}월 \d{1,2}일 [월화수목금토일]요일$',
            r'^\d{4}[.] \d{1,2}[.] \d{1,2}[.] 오[전후] \d{1,2}:\d{2}: .+습니다[.]$'
        ]
        SYSTEMSTR = "|".join(SYSTEMSTRLIST)

        for line in list:
            if not line:
                continue
            matchedstr = re.match(MATCHSTR, line)
            if matchedstr:
                hour = int(matchedstr.group(5))
                if matchedstr.group(4) == '오전':
                    if hour == 12:
                        hour = 0
                else:
                    if hour != 12:
                        hour += 12
                chattime = dt.datetime(
                    int(matchedstr.group(1)), # year
                    int(matchedstr.group(2)), # month
                    int(matchedstr.group(3)), # day
                    hour, # hour
                    int(matchedstr.group(6)) # minute
                )
                chatspeaker = str(matchedstr.group(7))
                chattext = str(re.sub(MATCHSTR, '', line)).strip()
                if chattext == '사진' or chattext == '동영상' or chattext == '이모티콘' or chattext == '삭제된 메시지입니다.':
                    continue
                if '샵검색: #' in chattext:
                    continue
                parsedline = {'time': chattime, 'speaker': chatspeaker, 'text': chattext}
                parsedlist.append(parsedline)
            else:
                systemstr = re.match(SYSTEMSTR, line)
                if systemstr:
                    continue
                else:
                    if not parsedlist[-1]['text']:
                        parsedlist[-1]['text'] = line
                    else:
                        parsedlist[-1]['text'] = parsedlist[-1]['text'] + '\n' + line
        return parsedlist
    
    # merge txt files in the folder and save the result in a json file
    def parse(self, file_path, merged_file_path, result_file_path):
        self.mergetxt(file_path, merged_file_path)
        lines = self.txtreadlines(merged_file_path)
        parsedlist = self.parse_lines(lines)

        return parsedlist
    
    # merge text in same contexts for polyglot-ko qlora models (for kakotalk)
    # return a huggingface dataset of chat contexts
    def create_context_dataset_from_kakaotalk(self, parsedlist, ai_name, tokenizer, prefix, suffix, chat_template, context_len=2048, context_sec=1800):
        dataset = {'text': []}
        contextchat = ''
        contextchat_len = 0
        chat_template_len = len(tokenizer.tokenize(chat_template))
        timestamp_last = parsedlist[0]['time']
        isAIChatInSecond = False

        for data in tqdm(parsedlist):
            chat = prefix + data['speaker'] + suffix + data['text'] + '\n\n'
            chat_len = len(tokenizer.tokenize(chat))
            timestamp_now = data['time']
            if not isAIChatInSecond and data['speaker'] == ai_name:
                first_ai_chat_idx = contextchat.find(prefix + ai_name + suffix)
                cutting_point = contextchat[:first_ai_chat_idx].rfind(prefix)
                if cutting_point != -1:
                    isAIChatInSecond = True
                    contextchat = contextchat[cutting_point:]
                    contextchat_len = len(tokenizer.tokenize(contextchat))
            if (timestamp_now - timestamp_last).seconds < context_sec and chat_template_len + contextchat_len + chat_len < context_len - 2: # for \n and EOS
                contextchat = contextchat + chat
                contextchat_len = contextchat_len + chat_len
            else:
                if contextchat.count(prefix + ai_name + suffix) > 3:
                    last_ai_chat_idx = contextchat.rfind(prefix + ai_name + suffix)
                    cutting_point = contextchat[last_ai_chat_idx + 1:].find(prefix)
                    if cutting_point != -1:
                        contextchat = contextchat[:last_ai_chat_idx + cutting_point]
                    dataset['text'].append(chat_template + contextchat + tokenizer.eos_token)
                isAIChatInSecond = False
                contextchat = chat
                contextchat_len = chat_len
            timestamp_last = timestamp_now
        dataset = DatasetDict({'train': Dataset.from_dict(dataset)})
        return dataset

from dotenv import load_dotenv
import os
from transformers import AutoTokenizer
import platform

load_dotenv()

PREFIX = os.environ.get('PREFIX')
SUFFIX = os.environ.get('SUFFIX')
CHAT_TEMPLATE = os.environ.get('CHAT_TEMPLATE')
MODEL_ID = os.environ.get('MODEL_ID')
AI_NAME = os.environ.get('AI_NAME')
TXT_DIR_PATH = os.environ.get('TXT_DIR_PATH')
DATA_PATH = os.environ.get('DATA_PATH')
PKL_PATH = os.environ.get('PKL_PATH')
MERGED_FILE_PATH = os.environ.get('MERGED_FILE_PATH')

if platform.system() == 'Darwin':
    MERGED_FILE_PATH = os.environ.get('MERGED_FILE_PATH_MAC')

p = Parser()
plist = p.parse(TXT_DIR_PATH, MERGED_FILE_PATH, DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = p.create_context_dataset_from_kakaotalk(plist, AI_NAME, tokenizer, PREFIX, SUFFIX, CHAT_TEMPLATE)
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
with open(PKL_PATH, 'wb') as f:
    pickle.dump(dataset, f)