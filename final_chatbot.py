# 챗봇 제작 개요 : 양선영
# 1.서비스 선택 : 1.프리랜서 가이드라인(출판본).pdf
#              2.OutdoorClothingCatalog_1000.csv
# 2.선택된 DB에 맞춰 서비스 제공

#!/usr/bin/env python
# coding: utf-8

# # Question Answering


import os
import time
import json
import sys
from typing import Any, Iterable, List
import langchain
from langchain.docstore.document import Document

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))
llm_model = "gpt-3.5-turbo-1106"
PDF_FILE = "./data/프리랜서 가이드라인 (출판본).pdf"
CSV_FILE = "data/OutdoorClothingCatalog_1000.csv"

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils import (
  BusyIndicator,
  ConsoleInput,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path
  )


import re

def reduce_newlines(input_string):
  # 정규 표현식을 사용하여 연속된 '\n'을 하나로 치환
  reduced_string = re.sub(r'\n{3,}', '\n\n', input_string)
  return reduced_string


def print_documents(docs: List[Any]) -> None:
  if docs == None:
    return

  print(f"documents size: {len(docs)}")
  p = lambda meta, key: print(f"{key}: {meta[key]}") if key in meta else None
  for doc in docs:
    print(f"source : {doc.metadata['source']}")
    p(doc.metadata, 'row')
    p(doc.metadata, 'page')
    print(f"content: {reduce_newlines(doc.page_content)[0:500]}")
    print('-'*30)

def print_result(result: Any) -> None:
  p = lambda key: print(f"{key}: {result[key]}") if key in result else None
  p('query')
  p('question')
  print(f"result: {'-' * 22}" )
  p('result')
  p('answer')  
  print('-'*30)

#  if debug == 1 :
#    print("documents")
#    print_documents(result['source_documents'])

# 아래 출력 막음
  if 'source_documents' in result:
    print("documents")
    print_documents(result['source_documents'])


llm = ChatOpenAI(model_name=llm_model, temperature=0)


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_qa(vectordb) -> ConversationalRetrievalChain:
  memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key='answer',
    return_messages=True
  )
  retriever=vectordb.as_retriever()
  qa = ConversationalRetrievalChain.from_llm(
      llm,
      retriever=retriever,
      return_source_documents=True,
      return_generated_question=True,
      memory=memory
  )
  return qa




def test_memory(vectordb, questions, is_debug=False)-> None:
  langchain.is_debug = is_debug
  qa = get_qa(vectordb)
  for q in questions:
    result = qa({"question": q})
    print_result(result)
  langchain.is_debug = False




def chat_qa(vectordb, is_debug=False) -> None:
  console = ConsoleInput(basic_prompt='% ')
  qa = get_qa(vectordb)


  while True:  # 무한루프 시작
    t = console.input()[0].strip()

    if t == '':  # 빈 라인인 경우.
      continue

    if t == 'q' or t == 'Q' or t == 'ㅂ':
      break

    busy_indicator = BusyIndicator().busy(True)
    langchain.is_debug = is_debug
    result = qa({"question": t})
    langchain.is_debug = False
    busy_indicator.stop()
    console.out(result['answer'])
    if is_debug:
      print_result(result)

def input_select(menu: dict) -> (int, str):
  print(menu.get("title"))
  items = menu.get("items", None)

  if items == None or len(items) == 0:
     raise ValueError("menu에 items가 없습니다.")
  for idx, item in enumerate(items):
    print(f"{str(idx+1)}. {item}")

  size = len(items)
  select = -1

  while select < 0:
    try:
      select = int(input(">>선택 :"))
      if select <= 0 or size < select:
        select = -1
    except ValueError:
      select = -1

    if select < 0:
      print("잘못된 선택입니다.")

  return ( select, items[select-1] )




def main():
  test, _ = input_select({
    "title" : "테스트할 내용을 선택하세요.",
    "items" : [
      "test_memory()",
      "chat_qa()"
    ]
  })
  db, _ = input_select({
    "title" : "테스트할 db를 선택하세요.",
    "items" : [
      "PDF:프리랜서 가이드라인(출판본)",
      "CSV:OutdoorClothingCatalog_1000"
    ]
  })
#  debug, _ = input_select({
#    "title" : "debuging 모드로 하시겠습니까?",
#    "items" : [
#      "yes",
#      "no"
#    ]
#  })

  vectordb : FAISS = load_vectordb_from_file(PDF_FILE if db == 1 else CSV_FILE)
  is_debug = 0 #1
# is_debug = debug == 0 #1


  if test == 1:
    questions_pdf = [
     "다음 질문을 수행할때, 항상 해당하는 페이지를 다음 줄에 알려줘."
      "1.이 책을 전체적으로 요약해줘.",
      "2.전체 책의 총 페이지를 알려줘.",
      "3.이 책의 저자는 누구야?",
      "4.전체 글에 대한 예상 질문을 5가지만 예를 들어줘.",
      "5.전체 목차를 출력해줘.",
      "6.프리랜서의 미래는 몇 페이지(page)에 있지?"
    ]
    questions_csv = [
      "다음 질문을 수행하고 반드시 한글로 대답해줘"
      "Can you recommend a shirt that doesn't wrinkle and is breathable?",
      "Give me three other recommendations."
    ]
    test_memory(vectordb,
      questions_pdf if db == 1 else questions_csv,
      is_debug
    )
  else:


    chat_qa(vectordb, is_debug)



if __name__ == '__main__':
  main()
