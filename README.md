# Pusan - AI chatbot generator from Kakaotalk conversation data

Pusan은 카카오톡에서 내보낸 대화 기록을 KoGPT2-v2로 파인튜닝하여 자유주제 챗봇 PyTorch 모델을 생성합니다.

필요한 환경 변수는 .env 파일에 넣어 사용합니다. 환경 변수의 목록은 다음과 같습니다.
- AI_NAME: 모델이 학습할 인물의 카카오톡 대화방 닉네임.
- TXT_FILEPATH: Parser.mergetxt()로 전처리된 카카오톡 대화 기록 파일명.
- TXT_FILEPATH_MAC: Mac, Linux 환경에서의 TXT_FILEPATH 값.
- MODEL_PATH: 모델을 저장하고 불러올 파일명.

## Training Pusan model
약 70만 건의 대화 기록을 batch_size=64, epoch=20으로 학습했을 때의 loss입니다. 학습에 걸린 시간은 약 9시간입니다.
![image](https://user-images.githubusercontent.com/86543294/220026733-507e95d8-e0b1-4abd-8223-36d5b2605569.png)
