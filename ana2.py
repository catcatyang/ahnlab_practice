import glob
import os
from typing import Any
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


def query_prompt(messages: []):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )

  return response

def get_countermeasures(content: str) -> Any:
  messages = [{
      "role": "system",
      "content": f"""
        다음의 [문장]의 부정적인 요인들을 극복할 대책방안의 초안을 작성하라.

        [문장]: {content}
      """
    }]
  response = query_prompt(messages)
  return response.choices[0].message["content"]


def get_review_result(content: str) -> Any:
  messages = [{
      "role": "system",
      "content": f"""
        다음의 [리뷰]를 긍정적인 요인과 부정적인 요인을 추출하여 ','로 구별하여 나열하라.

        [리뷰]: {content}
      """
    }]
  response = query_prompt(messages)
  return response.choices[0].message["content"]


def get_promotion_ments(content: str) -> Any:
  messages = [{
      "role": "system",
      "content": f"""
        다음의 [문장]의 긍정적인 요인들을 활용하여 100자 이내로 홍보 문구를 작성하라.

        [문장]: {content}
      """
    }]
  response = query_prompt(messages)
  return response.choices[0].message["content"]


def read_review_files():
    # 현재 디렉토리에서 review*.txt 파일 목록을 가져옵니다.
    review_files = glob.glob('./data/review*.txt')

    # 각 파일의 내용을 읽어서 yield로 반환합니다.
    for file_name in review_files:
        with open(file_name, 'r', encoding='utf-8') as file:
            yield file.read()

# 함수를 호출하여 각 파일의 내용을 가져옵니다.
result = ""
for review_content in read_review_files():
    result += get_review_result(review_content) + "\n\n"

ments = get_promotion_ments(result)
print(ments)
countermeasures = get_countermeasures(result)
print(countermeasures)

