# TFTNet

2024-1 데이터실험 최종 프로젝트입니다. 머신러닝, 딥러닝 기법을 활용하여 TFT 메타덱을 예측하는 프로젝트입니다. 

## data_crawling.ipynb
자신의 Riot API Key를 붙여넣으면, 라이엇 게임즈에서 제공하는 다양한 데이터를 크롤링할 수 있습니다. 라이엇 데이터 구조는 계층식으로 이루어져 있기에, 자세한 사항은 라이엇 게임즈의 공식 api 문서를 참고하시는 것을 추천합니다. 저의 코드는 tft match 데이터를 크롤링하는 코드입니다. 

## TFTNet
3개의 FC layer로 구성된 단순한 신경망 모델입니다. 특이한 점은 loss function을 자체적으로 고안하여, MSE와 pointwise-pair loss가 결합된 hybrid loss function을 사용했습니다. 수식은 아래와 같습니다.
\\
<img width="684" alt="image" src="https://github.com/01tilinfinity/TFTNet/assets/137471403/3652a3aa-aab7-41a9-b8ec-44d497f9933f">

