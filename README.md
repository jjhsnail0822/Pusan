# Pusan - AI chatbot generator from Kakaotalk conversation data

Pusan은 카카오톡에서 내보낸 대화 기록에 대해 Polyglot-ko 모델을 4-bit QLoRA를 통해 파인튜닝하여 자유주제 챗봇 PyTorch 모델을 생성합니다.

필요한 환경 변수는 .env 파일에 넣어 사용합니다. 환경 변수의 목록은 다음과 같습니다.
- PREFIX_AI: 답변 구분자. e.g., '### AI:'
- PREFIX_USER: 질문 구분자. e.g., '### USER:'
- EOS: EOS 토큰. e.g., '<|endoftext|>'
- MODEL_ID: 파인튜닝에 사용할 모델명. e.g., 'EleutherAI/polyglot-ko-5.8b'
- AI_NAME: 모델이 학습할 인물의 카카오톡 대화방 닉네임.
- TXT_FILEPATH: Parser.mergetxt()로 전처리된 카카오톡 대화 기록 파일명.
- TXT_FILEPATH_MAC: Mac, Linux 환경에서의 TXT_FILEPATH 값.
- MODEL_PATH: 모델을 저장하고 불러올 파일명.
- BATCH_SIZE: 학습 시의 배치 사이즈. e.g., 2
- ACC_STEPS: Gradient Accumulation 값. e.g., 8
- LEARNING_RATE: 학습 시의 학습률. e.g., 3e-4
- LR_SCHEDULER_TYPE: 학습 시 사용할 스케줄러 종류. e.g., 'cosine'
- STEPS: 학습할 총 스텝 수.
