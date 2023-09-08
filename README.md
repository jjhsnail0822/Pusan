# Pusan - AI chatbot generator from Kakaotalk conversation data

Pusan은 카카오톡에서 내보낸 대화 기록에 대해 주어진 언어 모델을 4-bit QLoRA를 통해 파인튜닝하여, 자유주제 챗봇 PyTorch 모델을 생성합니다.

필요한 환경 변수는 .env 파일에 넣어 사용합니다. .env 파일의 예시는 다음과 같습니다. 중괄호로 감싸진 텍스트는 개인화되어야 하며, 실제로 중괄호를 포함하는 것은 아닙니다.
```
PREFIX = "### "
SUFFIX = ":\n"
CHAT_DELIMITER = "\n\n"
CHAT_TEMPLATE = "아래는 여러 사람의 채팅 대화입니다. 이를 바탕으로 응답을 작성하세요.\n\n"
EOS = '<|endoftext|>'
MODEL_ID = 'nlpai-lab/kullm-polyglot-5.8b-v2'
PEFT_ID = 'pusan_qlora'
AI_NAME = '{AI 역할 사용자}'
USER_NAME = "{USER 역할 사용자}"
TXT_DIR_PATH = '{카카오톡 내보내기 폴더의 경로}'
DATA_PATH = 'Preprocessed.json'
PKL_PATH = 'Preprocessed.pkl'
MERGED_FILE_PATH = '{카카오톡 내보내기 폴더의 경로}\merge.txt'
MERGED_FILE_PATH_MAC = '{카카오톡 내보내기 폴더의 경로 (macOS)}/merge.txt'
BATCH_SIZE = 1
ACC_STEPS = 16
LEARNING_RATE = 3e-4
LOGGING_STEPS = 10
LR_SCHEDULER_TYPE = 'cosine'
STEPS = 1000
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_NEW_TOKENS = 32
SAVE_STEPS = 100
NUM_WORKERS = 0
CONTINUE_TRAINING = 0
```

# Prerequisites
```
pip install natsort tqdm python-dotenv torch transformers peft
```
Windows 사용자의 경우, bitsandbytes 패키지가 현재 Windows를 지원하고 있지 않으므로, 따로 수동으로 빌드하여 설치할 필요가 있습니다.

# Getting Started
```
python ./dataparser.py
python ./train.py
python ./run.py
```
