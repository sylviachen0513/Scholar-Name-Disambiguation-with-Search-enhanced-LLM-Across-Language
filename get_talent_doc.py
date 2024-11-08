import pandas as pd
import re
import json
import numpy as np
import csv
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import requests
from requests.auth import HTTPBasicAuth
from pypinyin import pinyin, Style
import logging

search_url = "http://101.226.141.241/search"
gpt_url = "http://101.226.141.241/gpt3"
chat_url = "http://101.226.141.241/chat"
headers = {
    "Content-Type": "application/json"
}


def processed_workplace(workplace):
    payload = {
        'text': f"请对输入的工作地点workplace {workplace} 进行以下处理："
                "1、如果输入的workplace为中文："
                "1.1 对输入的中文 workplace 进行补充和完善。例如：'上海高等研究院' 应补充为 '中国科学院上海高等研究院'。"
                "2、如果输入的workplace为英文："
                "2.1 请将workplace翻译为中文，并仅输出学校院系名称。例如：'Tsinghua Univ' 应翻译为 '清华大学'。"
                "请只输出答案workplace，忽略中间处理过程.",
        'model': 'gpt4o',
        'search': True
    }
    response = requests.post(gpt_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def is_school(workplace):
    payload = {
        'text': f"请判断以下输入的{workplace}是否为中国大陆境内的高等院校，不包括港澳台和海外地区。"
                f"如果{workplace}包含完整的高等院校名称，请返回True；否则，请返回False。"
                f"特别注意的是，当{workplace}包含'中国科学院','大学'子串，应返回True"
                "请只输出推断的结果，即True或False。",
        'model': 'hy',
    }
    response = requests.post(chat_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def search_info(text):
    # 搜索信息
    payload = {
        'text': f"{text}",
        'model': 'hy'
    }
    response = requests.post(search_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    return response_data


def search_info_google(text):
    payload = {
        'text': f"{text}",
        'model': 'gpt4o',
        'engine': 'google',
        'rewrite': True,
        'expand': True
    }
    response = requests.post(search_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    return response_data


def preprocess_sougou_data(datas):
    # 初筛并处理信息
    filtered_datas = [{key: d[key] for key in ['url', 'title', 'body'] if key in d} for d in datas]
    filtered_datas = [d for d in filtered_datas if 'body' in d]
    filtered_datas = [
        d for d in filtered_datas
        if 'zhaopin' not in d.get('url', '')
    ]
    priority_substrings = ['.edu', '.aminer', '.scholarmate', '.ac', '.org', '.cas', '.cae']

    def contains_keywords(body, keywords):
        for keyword in keywords:
            if re.search(keyword, body):
                return True
        return False

    def prioritize_urls(lst, priority_substrings):
        keywords = [
            r'(教育背景|学士|硕士|博士|导师|大学|学院|博士后|访问学者)',
            r'(工作经历|研究员|教授|博士后|公司|研究所)',
            r'(研究方向|研究领域|研究兴趣)',
            r'(论文|著作|出版|发表)',
            r'(奖项|荣誉|获奖|称号)'
        ]

        filtered_lst = [item for item in lst if contains_keywords(item['body'], keywords)]

        def url_priority(item):
            url = item.get('url', '')
            for substring in priority_substrings:
                if substring in url:
                    return 0
            return 1

        sorted_lst = sorted(filtered_lst, key=url_priority)
        return sorted_lst

    sorted_lst = prioritize_urls(filtered_datas, priority_substrings)
    return sorted_lst


def talent_search(text):
    payload = {
        'text': f"{text}",
        'model': 'gpt4o',
        'search': True

    }
    response = requests.post(gpt_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def get_mainpage_info(item):
    payload = {
        'text': f'Given the following information: item={item}. '
                'Format of item: {"url": url, "title": title, "body": body}. '
                'Determine if the item is a biography or personal homepage of an individual. '
                'The judging criteria are whether the body text contains relevant information such as educational background, work experience, research field etc.'
                'If the body contains information related to papers or research, it is more likely to be a personal homepage.'
                'Please return only "True" if it is a biography or personal homepage, otherwise return "False".',
        'model': 'gpt4o',
    }
    response = requests.post(gpt_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def filter_unrelated_info(item, query):
    payload = {
        'text': f'Given the following information: item={item},query={query}. '
                'Format of item: {"url": url, "title": title, "body": body}. '
                'Format of query: {"name": Chinese character or Pinyin format, "workplace": workplace}. '
                'Please determine if the item is related to the query based on the following criteria: '
                '1. Check if any expression of query["name"] appears in the item, including: '
                '   - Chinese name (e.g., 齐殿鹏) '
                '   - Pinyin representation (e.g., qi dianpeng) '
                '   - English name (e.g., dianpeng qi) '
                '   Note: The check is case-insensitive and ignores spaces. '
                '2. Check if the item mentions query["workplace"], considering variations such as Peking Univ and 北京大学 as equivalent. '
                'Please return only "True" If the item contains relevant information related to query,otherwise, return "False".',
        'model': 'gpt4o',
    }
    response = requests.post(gpt_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def filter_query(name, workplace=None):
    query_dict = {"name": name, "workplace": workplace}
    return query_dict


def deep_processed(datas):
    # 去重
    payload = {
        'text': f"您将得到字符串输入{datas}，其中包含多个字典记录，每个字典记录代表一位学者的信息，包含三个字段：url、title和body。body字段中包含网页的正文内容。"
                "请根据以下规则对学者信息进行去重操作："
                "1. 判断是否为重复学者信息的依据为字典中的信息均为学者个人生平介绍，可以根据学者的教育、工作经历、研究领域、工作单位等判断是否为同一学者。"
                "2. 对于重复的学者信息，优先保留title中含有高等院校或专业机构名称的字典记录；"
                "3. 如果有多个字典记录符合条件，优先保留body字段中包含论文相关信息的字典记录；"
                "4. 如果仍有多个字典记录符合条件，保留body内容最长的字典记录。"
                "输出应包含所有符合条件的字典记录,如果没有符合要求的字典，返回None。不输出任何中间处理和分析的过程。"
                "输出请确保满足格式要求：两个字典之间请务必用'||'进行分割，即{dict1}||{dict2}。",
        'model': 'gpt4o',
    }
    response = requests.post(chat_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def is_same_talent(doc1, doc2):
    if not isinstance(doc1, str):
        doc1_str = json.dumps(doc1, ensure_ascii=False)
    else:
        doc1_str = doc1

    if not isinstance(doc2, str):
        doc2_str = json.dumps(doc2, ensure_ascii=False)
    else:
        doc2_str = doc2
    # 判断两位学者是否相同
    payload = {
        'text': f"您将得到两个字典格式的json字符串输入{doc1_str}和{doc2_str}，请判断这两个字典记录的学者信息是否属于同一位学者，请按顺序特别关注以下字段："
                "workplace：若工作地点近似，得 2 分。如 “中国科学院上海高等研究院” 和 “上海高等研究院” 指代的是相同的地点。缺乏相关字段则记为 0 分。"
                "education_track：重点关注 school 和 scholar 字段。如果两位学者在同一个学校（school 字段）取得同一个学位（scholar 字段），每一条记录得 3 分。缺乏相关字段则记为 0 分。"
                "professional_track：重点关注 agency 字段。如果两位学者在同一家单位（agency 字段）得到同一个职称（title 字段），每一条记录得 3 分。缺乏相关字段则记为 0 分。"
                "keywords：对比学者的研究领域是否相似或相同，若研究领域相似度高的得 1-4 分，缺乏相关字段则记为 0 分。"
                "请根据得分判断两者是否为同一位学者，得分达到或超过 7 分可判定为同一位学者。请只输出最终的推断答案：True 或 False，不需要中间分析过程。",
        'model': 'gpt4o',
    }
    response = requests.post(gpt_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


url = 'http://stream-server-online-hyaide-app.turbotke.production.polaris:81/openapi/app_platform/app_create'
hy_headers = {
    'Authorization': 'Bearer 7auGXNATFSKl7dF',
    'Content-Type': 'application/json'
}


def summary_info(query):
    if not isinstance(query, str):
        query = json.dumps(query, ensure_ascii=False)
    data = {
        "query": query,
        "forward_service": "hyaide-application-4745",
        "query_id": "qid_123456"
    }
    response = requests.post(url, headers=hy_headers, data=json.dumps(data))
    response.raise_for_status()
    word = response.json().get('result', None)
    return word


def url_search(query):
    data = {
        "query": query,
        "forward_service": "hyaide-application-4748",
        "query_id": "qid_123456"
    }
    response = requests.post(url, headers=hy_headers, data=json.dumps(data))
    response.raise_for_status()
    word = response.json().get('result', None)
    return word


def process_email(email2):
    if isinstance(email2, str):
        return [email2]
    elif isinstance(email2, list):
        if len(email2) == 0:
            return None
        edu_emails = [email for email in email2 if isinstance(email, str) and ".edu" in email]
        non_edu_emails = [email for email in email2 if isinstance(email, str) and ".edu" not in email]
        return edu_emails + non_edu_emails
    else:
        return None


honors = ["中国科学院院士", "中国工程院院士", "长江学者特聘教授", "长江学者讲座教授", "国家杰出青年科学基金获得者",
          "国家“万人计划”科技创新领军人才", "35岁以下科技创新35人", "青年长江学者"]


def sort_honor_track(honor_track):
    if not honor_track or not isinstance(honor_track, list):
        return honor_track

    for item in honor_track:
        if not isinstance(item, dict):
            return honor_track

    honor_track = [item for item in honor_track if item.get('award') != '五年顶刊通信作者']
    if not honor_track:
        return None

    sorted_honor_track = sorted(
        honor_track,
        key=lambda x: (x.get('award') not in honors,
                       honors.index(x.get('award')) if x.get('award') in honors else float('inf'))
    )
    return sorted_honor_track


def extract_dict_url(s):
    pattern = r'{"url": "(.*?)", "title": "(.*?)", "body": "(.*?)"}'
    match = re.search(pattern, s)
    if match:
        return {
            "url": match.group(1),
            "title": match.group(2),
            "body": match.group(3)
        }
    return None

def process_single_candidate(candidate):
    url = candidate['url']
    doc2_extra = candidate['body']
    if len(doc2_extra) < 200:
        doc2_extra = url_search(url)
    doc2_extra_summary = summary_info(doc2_extra)
    return doc2_extra_summary


def search_candidate(text, query, candidates, key='sougou'):
    if key == 'sougou':
        data = search_info(text)
        info1 = preprocess_sougou_data(data)
    else:
        data = search_info_google(text)
        info1 = preprocess_google_data(data)
    if isinstance(info1,dict):
        return None,candidates

    info2, info3, info4, info5 = [], [], [], []
    for item in info1:  # 获取主页信息
        mainpage_info = get_mainpage_info(item)
        if mainpage_info is not None and 'True' in mainpage_info:
            info2.append(item)
    if len(info2) == 0:
        return None, candidates

    for item in info2:  # 删去不相关信息
        filtered_info = filter_unrelated_info(item, query)
        if filtered_info is not None and 'True' in filtered_info:
            info3.append(item)
    if len(info3) == 0:
        return None, candidates
    elif len(info3) == 1:
        return process_single_candidate(info3[0]), candidates

    info4 = deep_processed(json.dumps(info3, ensure_ascii=False))  # 删除重复信息
    if info4 is None or 'None' in info4 or info4 == '' or 'example.com' in info4 or 'python' in info4 or 'import' in info4:
        return None, candidates
    if '||' not in info4:
        candidate = extract_dict_url(info4)
        if candidate:
            return process_single_candidate(candidate), candidates

    info4_lst = info4.split('||')
    for word in info4_lst:
        if info5:
            is_same = is_same_talent(info5[-1], word)
            if is_same is not None and 'True' in is_same:
                continue
        info5.append(word)
    if len(info5) == 1:
        candidate = extract_dict_url(info5[0])
        if candidate:
            return process_single_candidate(candidate), candidates
    else:
        candidates.append(info5)

    return None, candidates


def construct_chat_text(name, email=None, workplace=None, honor=None):
    parts = []
    if name:
        parts.append(f"请搜索姓名为{name}")
    if workplace:
        parts.append(f"机构为{workplace}")
    if email:
        parts.append(f"{', '.join(email)}")
    if honor and 'award' in honor:
        parts.append(f"获得{honor['award']}奖项")

    if parts:
        return (f"{'，'.join(parts)}。请获取并汇总教育经历、工作经历、研究领域、工作地点等个人介绍信息。\n"
                "请确保获取到所有满足条件的学者信息，并提示学者数量，格式为'学者数量==X'，其中X为学者数量。不同学者信息之间请务必用'||'进行分割。")


def construct_search_text(name, email=None, workplace=None, honor=None, key='search'):
    parts = []
    if name:
        parts.append(f"{name}教师")
    if workplace:
        parts.append(f"{workplace}")
    if email:
        filtered_email = [e for e in email if e is not None]
        parts.append(f"{','.join(filtered_email)}")
    if honor and 'award' in honor:
        parts.append(f"{honor['award']}")

    if key == 'search':
        return (f"{'，'.join(parts)}，个人主页/简介 或 homepage 或 info")
    elif key == 'sougou':
        return (f"{','.join(parts)}")


def construct_paper_text(name, email=None, workplace=None):
    parts = []
    if name:
        parts.append(f"{name}")
    if workplace:
        parts.append(f"{workplace}")
    if email:
        filtered_email = [e for e in email if e is not None]
        parts.append(f"{','.join(filtered_email)}")

    return (f"{'，'.join(parts)}，info OR homepage OR profiles OR 个人主页")

def process_name(name):
    processed_name = re.sub(r'[^a-zA-Z\s]', '', name)
    processed_name = processed_name.title()
    return processed_name


def handle_search_result(doc2_extra, candidates):
    if doc2_extra is None:
        return None, candidates
    match = re.search(r'学者数量\s*==\s*(\d+)', doc2_extra)
    if match:
        scholar_count = int(match.group(1))
        if scholar_count == 0:
            return None, candidates
        elif scholar_count == 1:
            doc2_extra_summary = summary_info(doc2_extra)
            return doc2_extra_summary, candidates
        else:
            info = []
            doc2_extra_lst = doc2_extra.split('||')
            for word in doc2_extra_lst:
                if info:
                    is_same = is_same_talent(info[-1], word)
                    if is_same is not None and 'True' in is_same:
                        continue
                info.append(word)
            if len(info) == 1:
                doc2_extra_summary = summary_info(info[0])
                return doc2_extra_summary, candidates
            else:
                candidates.append(info)

    return None, candidates

def preprocess_google_data(data_list):
    filtered_data = []
    for item in data_list:
        if 'pagemap' in item and 'metatags' in item['pagemap'] and 'citation_keywords' in item['pagemap']['metatags'][
            0]:
            continue
        if isinstance(item, dict):
            filtered_item = {
                'url': item.get('link'),
                'title': item.get('title'),
                'body': item.get('body')
            }
            filtered_data.append(filtered_item)
    return filtered_data


def infer_name(item,query):
    payload = {
        'text':f'Given the following information: item={item} '
        'Format of item: {"url": url, "title": title, "body": body}. '
        f'Please identify the scholar\'s Chinese name matches the given conditions query={query}? '
        'Format of query: {"name": Pinyin format, "workplace": workplace}. '
        'Please return only the identified Chinese character name or "Not Found".'
        'If the name is in traditional Chinese, please convert it to simplified Chinese. '
        'Do not return any other information. Ignore intermediate processing.',
        'model':'gpt4o',
    }
    response = requests.post(gpt_url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    word = response_data.get('data', {}).get('gpt')
    return word


def infer_chinese_name(info, query):
    for item in info:
        response = infer_name(item, query)
        if response is not None and 'Not Found' not in response:
            return response
    return None


def get_school_name(affiliation):
    index = affiliation.find(',')
    if index != -1:
        processed_aff = affiliation[:index].strip()
    else:
        processed_aff = affiliation
    return processed_aff


def simple_workplace(workplace):
    comma_indices = [i for i, char in enumerate(workplace) if char == ',']
    if len(comma_indices) >= 2:
        workplace = workplace[:comma_indices[1]]
        return workplace.replace(',', '')
    else:
        return workplace.replace(',', '')


def get_chinese_name(doc2, key='google'):
    name = doc2.get('name')
    workplace = doc2.get('workplace', '')
    address = get_school_name(doc2.get('workplace', ''))
    email = doc2.get('email', '')

    if key == 'google':
        paper_text = construct_search_text(name, workplace=address,key='google')
        info = search_info_google(paper_text)
    else:
        address = processed_workplace(address)
        paper_text = construct_search_text(name, email=email, workplace=address,key='sougou')
        info = search_info(paper_text)

    if isinstance(info, dict):
        return None

    processed_info = preprocess_info(info, key)
    chinese_name = infer_chinese_name(processed_info, filter_query(name, address))
    return chinese_name


def preprocess_info(info, key):
    if key == 'google':
        return preprocess_google_data(info)
    else:
        return [{k: d[k] for k in ['url', 'title', 'body'] if k in d} for d in info]


def extract_fields_using_regex(doc2_summary):
    """使用正则表达式从doc2_summary中提取字段"""
    fields = ["name", "email", "workplace", "education_track", "professional_track", "honor_track", "keywords"]
    extracted_data = {}

    for field in fields:
        pattern = f'"{field}"\s*:\s*(\[.*?\]|".*?"|null)'
        match = re.search(pattern, doc2_summary)
        if match:
            value = match.group(1)
            if value == "null":
                extracted_data[field] = None
            elif value.startswith('['):
                try:
                    extracted_data[field] = json.loads(value)
                except:
                    extracted_data[field] = []
            else:
                extracted_data[field] = value.strip('"')
    return extracted_data

def is_dict_empty_or_null(d):
    """检查字典中的所有值是否都为 null"""
    return all(value in [None, "null"] for value in d.values())

def update_field(doc2_field, summary_field):
    """更新字段，去重并过滤'null'值"""
    if isinstance(doc2_field, str):
        try:
            doc2_field = json.loads(doc2_field)
        except json.JSONDecodeError:
            doc2_field = []

    doc2_field=[] if doc2_field is None else doc2_field
    summary_field=[] if summary_field is None else summary_field

    summary_field = [item for item in summary_field if item != "null"]
    summary_field = [item for item in summary_field if not isinstance(item, dict) or not is_dict_empty_or_null(item)]
    return json.dumps(doc2_field + summary_field, ensure_ascii=False)

def update_doc2_from_summary(doc2, doc2_summary):
    try:
        summary_data = json.loads(doc2_summary)
    except json.JSONDecodeError:
        summary_data = extract_fields_using_regex(doc2_summary)

    for col in ['name', 'workplace']:
        if summary_data.get(col) != 'null':
            doc2[col] = summary_data.get(col, doc2.get(col))

    for col in ['email', 'keywords']:
        if doc2.get(col) is None:
            doc2[col] = []
        if summary_data.get(col) is not None:
            doc2[col] = list(set(doc2[col] + [x for x in summary_data.get(col, []) if x != 'null']))

    for col in ['education_track', 'professional_track', 'honor_track']:
        if summary_data.get(col) is not None:
            doc2[col] = update_field(doc2.get(col, '[]'), summary_data.get(col, []))

    return doc2

def check(updated_doc2):
    edu,pro,key=updated_doc2.get('education_track'),updated_doc2.get('professional_track'),updated_doc2.get('keywords')
    if (edu in ['[]','[null]']) and (pro in ['[]','[null]']) and (key==[] or key==[None]):
        return None
    return updated_doc2

def get_doc(doc2):
    name2 = doc2.get('name')
    workplace2 = doc2.get('workplace')
    address = get_school_name(workplace2) if workplace2 is not None else None
    email2 = doc2.get('email', [])
    honor_track2 = doc2.get('honor_track', '[]')
    try:
        honor_track2 = json.loads(honor_track2)
        honor_track2 = sort_honor_track(honor_track2)
    except:
        honor_track2=[]

    email2 = process_email(email2)
    workplace2 = processed_workplace(workplace2)
    doc2['email']=email2
    honor = honor_track2[0] if honor_track2 and isinstance(honor_track2, list) else None
    if not isinstance(honor, dict) and honor != None:
        honor = None
    chat_text = construct_chat_text(name2, email2, workplace2, honor)
    search_text = construct_search_text(name2, email2, workplace2, honor,'search')
    query = filter_query(name2, workplace2)
    candidates=[]
    doc2_extra_summary, updated_doc2 = None, None
    if workplace2:
        flag = is_school(workplace2)
        if flag is not None and 'True' in flag:
            doc2_extra_summary, candidates = search_candidate(search_text, query, candidates, 'sougou')
            if doc2_extra_summary is None:
                doc2_extra = talent_search(chat_text)
                doc2_extra_summary, candidates = handle_search_result(doc2_extra, candidates)
        else:
            doc2_extra = talent_search(chat_text)
            doc2_extra_summary, candidates = handle_search_result(doc2_extra, candidates)

    else:
        doc2_extra = talent_search(chat_text)
        doc2_extra_summary, candidates = handle_search_result(doc2_extra, candidates)

    if doc2_extra_summary:
        updated_doc2 = update_doc2_from_summary(doc2, doc2_extra_summary)
        updated_doc2 = check(updated_doc2)

    return updated_doc2, candidates

def fetch_chinese_name(doc2,name):
    workplace=doc2.get('workplace','')
    chinese_name = get_chinese_name(doc2)
    if chinese_name is None:
        chinese_name = get_chinese_name(doc2, 'sougou')
    else:
        pinyin_format = name_to_pinyin(chinese_name)
        if 'Hong Kong' not in workplace and name not in pinyin_format:
            chinese_name = get_chinese_name(doc2, 'sougou')
    return chinese_name

def name_to_pinyin(name):
    pinyin_list = pinyin(name, style=Style.NORMAL)
    surname = pinyin_list[0][0].capitalize()
    given_name = ''.join([item[0] for item in pinyin_list[1:]]).capitalize()
    all_pinyin = ' '.join([item[0].capitalize() for item in pinyin_list])
    return [f"{surname} {given_name}",all_pinyin, f"{given_name} {surname}"]


def get_paper_doc(doc):
    name = process_name(doc['name'])
    doc['name'] = name
    chinese_name = fetch_chinese_name(doc,name)
    doc['workplace']=simple_workplace(doc['workplace'])
    if chinese_name is not None:
        doc['name'] = chinese_name
    updated_doc, candidates = get_doc(doc)
    if updated_doc is None:
        doc['name'] = name
        chinese_name = get_chinese_name(doc, 'sougou')
        if chinese_name is not None:
            doc['name'] = chinese_name
            updated_doc, candidates = get_doc(doc)
    if updated_doc is None:
        doc['name']=name
        updated_doc,candidates=get_doc(doc)

    return updated_doc,candidates

