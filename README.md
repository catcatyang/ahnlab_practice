# ahnlab_practice
ahnlab 실습용 및 과제 제출용

# 챗봇 제작 개요 : 양선영
  이용자료 : 1.프리랜서 가이드라인(출판본).pdf
            2.OutdoorClothingCatalog_1000.csv

# 최종 파일
 1. final_chatbot.py : langchain_C_chat.py 이용
  - 환경 변수 : openai API 키 적용
  - langchain 사용
  - 메모리 이용한 히스토리 관리
  - 질문 사항 적용
  - 채팅 구현 : 커멘드 라인에 입력

 2. final_fastapi.py : final_chatbot.py 이용
  - 환경 변수 : OpenAI API 키 적용
  - langchain 사용
  - 메모리 이용한 히스토리 관리
  - 프론트엔드 부분 ( index.html ) : 사용자가 데이터베이스(DB)를 선택 후, 메시지를 입력, 
                                    해당 메시지를 서버에 전송 후, 응답을 화면에 표시
 
# 참고
강사님께서 주신 수업자료와 기존 제출자들의 자료를 참고하여 최대한 이해해보고 구현해보려고 해보았으나
과제 내용을 구현하기에는 많이 미비합니다. 
차후 주신 자료로 더 공부해보겠습니다. 고생하신 강사님께 감사드립니다.
