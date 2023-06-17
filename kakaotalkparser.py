import re
import glob
import os
import natsort
import datetime as dt
import copy

# parse txt file(s) saved by kakaotalk exportation
class Parser():
    # concatenate txt files in the folder and save the result
    def mergetxt(self, path):
        mergedfilepath = os.path.join(path, 'merge.txt')
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
                parsedline = {'time': chattime, 'speaker': chatspeaker, 'text': chattext}
                parsedlist.append(parsedline)
            else:
                systemstr = re.match(SYSTEMSTR, line)
                if systemstr:
                    continue
                else:
                    if not parsedlist[-1]['text']:
                        parsedlist[-1]['text'] = line
                    elif parsedlist[-1]['text'][-1] == ' ':
                        parsedlist[-1]['text'] = parsedlist[-1]['text'] + line
                    else:
                        parsedlist[-1]['text'] = parsedlist[-1]['text'] + ' ' + line
        return parsedlist
    
    # merge simultaneous texts by same speaker based on datetime
    # return a list of q:questions and a:answers of a given speaker
    def create_dataset(self, parsedlist, speaker):
        dataset = {'q': [], 'a': []}
        # lastchat['text'] has no meaning
        lastchat = {'time': dt.datetime(1970,1,1), 'speaker': '', 'text': ''}
        holdingchat = {'time': dt.datetime(1970,1,1), 'speaker': '', 'text': ''}
        question = ''
        answer = ''
        for chat in parsedlist:
            if lastchat['time'] == chat['time'] and lastchat['speaker'] == chat['speaker']:
                holdingchat['text'] = holdingchat['text'] + ' ' + chat['text']
            elif lastchat['speaker'] == chat['speaker']:
                if chat['speaker'] == speaker:
                    continue
                else:
                    holdingchat = lastchat = chat
            else:
                if chat['speaker'] == speaker:
                    question = holdingchat['text']
                elif lastchat['speaker'] == speaker:
                    answer = holdingchat['text']
                    dataset['q'].append(question)
                    dataset['a'].append(answer)
                    question = answer = ''
                holdingchat = lastchat = chat
        return dataset
    
    # merge text of same speakers
    # return a list of dictionary containing speaker, text information
    def create_merged_dataset(self, parsedlist):
        dataset = []
        lastchat = {'time': dt.datetime(1970,1,1), 'speaker': '', 'text': ''}
        nowchat = {'speaker': '', 'text': ''}

        for chat in parsedlist:
            if lastchat['speaker'] == chat['speaker']:
                nowchat['text'] = nowchat['text'] + '\n' + chat['text']
            else:
                if nowchat['speaker']:
                    dataset.append(copy.deepcopy(nowchat))
                nowchat['speaker'] = chat['speaker']
                nowchat['text'] = chat['text']
            lastchat = chat
        return dataset

# parsing test code

from dotenv import load_dotenv
import platform

load_dotenv()

TXT_DIR_PATH = os.environ.get('TXT_DIR_PATH')
TXT_FILEPATH = os.environ.get('TXT_FILEPATH')
if platform.system() == 'Darwin':
    TXT_FILEPATH = os.environ.get('TXT_FILEPATH_MAC')

p = Parser()

# p.mergetxt(TXT_DIR_PATH)

plist = p.parse_lines(p.txtreadlines(TXT_FILEPATH))
dataset = p.create_merged_dataset(plist)