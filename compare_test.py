import json
from tkinter.font import names
from get_talent_doc import is_same_talent, get_paper_doc, get_doc
import numpy as np
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse, StreamingResponse, Response
import httpx
from pydantic import BaseModel
import pandas as pd
from pypinyin import pinyin, Style
import re

# app = Flask(__name__)
app = FastAPI(debug=True, docs_url=None, redoc_url=None)


class CompareRequest(BaseModel):
    doc1_str: str
    doc2_str: str


class DocRequest(BaseModel):
    doc_str: str


def compare_function(doc1, doc2):
    if doc2 is None:
        return False

    name2 = doc2.get('name')
    if not name2 or name2 == '未找到':
        return False

    workplace1 = doc1.get('workplace')
    workplace2 = doc2.get('workplace', '')
    email1 = doc1.get('email', [])
    email2 = doc2.get('email', [])
    honor_track1 = doc1.get('honor_track', '[]')
    try:
        honor_track1 = json.loads(honor_track1)
    except:
        honor_track1 = None
    honor_track2 = json.loads(doc2.get('honor_track', '[]'))

    # 1. 比较 name + aminer_id
    aminer_id1 = doc1.get('aminer_id')
    aminer_id2 = doc2.get('aminer_id')
    if aminer_id1 and aminer_id2 and aminer_id1 == aminer_id2:
        return True

    # 2. 比较 name + google_scholar_url
    google_scholar_url1 = doc1.get('google_scholar_url')
    google_scholar_url2 = doc2.get('google_scholar_url')
    if google_scholar_url1 and google_scholar_url2 and google_scholar_url1 == google_scholar_url2:
        return True

    # 3. 比较 name + mainpage
    mainpage1 = doc1.get('mainpage')
    mainpage2 = doc2.get('mainpage')
    if mainpage1 and mainpage2 and mainpage1 == mainpage2:
        return True

    # 4. 比较 email
    if isinstance(email2, str):
        email2 = [email2]
    if any(e in email2 for e in email1):
        return True

    # 5. 比较 honor 字段
    four_titles = ["中国科学院院士", "中国工程院院士", "国家杰出青年科学基金获得者", "长江学者特聘教授",
                   "长江学者讲座教授"]
    if isinstance(honor_track1, list):
        if workplace1 and workplace2 and workplace1 == workplace2:
            for h1 in honor_track1:
                for h2 in honor_track2:
                    if h1.get('award') in four_titles and h2.get('award') in four_titles:
                        if h1.get('award') == h2.get('award') and h1.get('time') == h2.get('time'):
                            return True

    # 6. 比较 prize_relations
    if workplace1 and workplace2 and workplace1 == workplace2:
        prize_relations1 = doc1.get('prize_relations', [])
        prize_relations2 = doc2.get('prize_relations', [])
        if prize_relations1 is not None and prize_relations2 is not None:
            if set(prize_relations1) & set(prize_relations2):
                return True

    is_same = is_same_talent(doc1, doc2)
    if is_same and 'True' in is_same:
        return True

    return False


def process_doc(doc):
    for field in ['honor_track', 'education_track', 'professional_track']:
        try:
            doc[field] = json.loads(doc.get(field, 'null'))
            if not doc[field]:
                doc[field] = None
            if not isinstance(doc[field], list):
                doc[field] = None
        except (json.JSONDecodeError, TypeError):
            doc[field] = None
    for field in ['email', 'keywords']:
        if not doc.get(field):
            doc[field] = None
    return doc


def without_search(doc):
    if doc.get('education_track') is None and doc.get('professional_track') is None:
        return False
    return True


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(text))


def get_talent_doc(doc):
    name = doc['name']
    if contains_chinese(name) == False and not doc.get('honor_track'):
        updated_doc, candidates = get_paper_doc(doc)
    else:
        updated_doc, candidates = get_doc(doc)
    return updated_doc, candidates


def talent_doc(doc_str):
    try:
        doc = json.loads(doc_str)
    except:
        return {"code": 90000, "info": "输入的doc不符合json格式要求"}
    updated_doc, candidates = get_talent_doc(doc)

    if updated_doc is not None:
        return {"code": 91000, "candidates": updated_doc}
    else:
        if len(candidates) == 0:
            return {"code": 90002, "error": "未找到合适的候选人", "candidates": None}
        else:
            return {"code": 90002, "error": "找到多位候选人信息", "candidates": candidates}


def main_compare(doc1_str, doc2_str):
    try:
        doc1 = json.loads(doc1_str)
    except:
        return {"code": 90001, "info": "输入的doc1不符合json格式要求"}
    try:
        doc2 = json.loads(doc2_str)
    except:
        return {"code": 90001, "info": "输入的doc2不符合json格式要求"}
    doc1 = process_doc(doc1)
    if without_search(doc1) == True:
        if without_search(doc2) == True:
            flag = compare_function(doc1, doc2)
            return {"code": 10000, "result": flag}

        updated_doc2, candidates2 = get_talent_doc(doc2)
        if updated_doc2 is not None:
            flag = compare_function(doc1, updated_doc2)
            return {"code": 10000, "result": flag}
        else:
            return {"code": 90003, "result": False, "error": "doc2未找到合适的候选人"}
    else:
        updated_doc1, candidates1 = get_talent_doc(doc1)
        if without_search(doc2) == True:
            flag = compare_function(updated_doc1, doc2)
            return {"code": 10000, "result": flag}

        if updated_doc1 is not None:
            updated_doc2, candidates2 = get_talent_doc(doc2)
            if updated_doc2 is not None:
                flag = compare_function(updated_doc1, updated_doc2)
                return {"code": 10000, "result": flag}
            else:
                return {"code": 90003, "result": False, "error": "doc2未找到合适的候选人"}
        return {"code": 90004, "result": False, "error": "doc1未找到合适的候选人"}
