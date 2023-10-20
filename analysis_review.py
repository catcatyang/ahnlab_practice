import glob
import os
from typing import Any
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")

def read_review_file_content(file_name):
    try:
        # 파일 열기
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            yield content
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")


def get_review_files_contents():
    # 현재 디렉토리에서 review*.txt 파일 목록을 가져옴
    review_files = glob.glob('./data/review*.txt')

    if not review_files:
        print("리뷰 파일이 없습니다.")
        return

    # 파일 내용을 yield로 반환
    for file_name in review_files:
        yield from read_review_file_content(file_name)


def query_prompt(messages: []):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )

  return response



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



# 함수 호출
def main():
  review_result = ""
  for idx, content in enumerate(get_review_files_contents(), start=1):
    if content is not None:
        result = get_review_result(content)
        print(f"리뷰분석 ({idx}) : {result}\n")
        review_result += "=======================================\n\n"
        review_result += result
    else:
        print(f"파일 {idx}을 읽을 수 없습니다.\n")

  ments = get_promotion_ments(review_result)
  countermeasures = get_countermeasures(review_result)

  print(f"홍보문구 :{ments}")
  print(f"대책방안 :{countermeasures}")




if __name__ == '__main__':
    main()