# Copyright 2024 yanruibing@gmail.com All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import jieba
import opencc
from tqdm import tqdm
import json

def read_txt_files(dir: str) -> list:
    all_texts = []
    for root, _, files in os.walk(dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_texts.append(file.read())
    return all_texts

def extract_metadata(text):
    """提取书籍元数据"""
    metadata = {}
    book_info = re.search(r"<book>(.*?)</book>", text, re.DOTALL)
    if book_info:
        lines = book_info.group(1).splitlines()
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()
    return metadata

def extract_chapters(text):
    """提取章节和内容"""
    chapters = {}
    matches = re.split(r"(=====.*?=====)", text)
    current_chapter = None
    for match in matches:
        if re.match(r"^=====", match):
            current_chapter = match.strip("= \n")
            chapters[current_chapter] = []
        elif current_chapter:
            chapters[current_chapter].append(match.strip())
    # 合并每个章节的内容
    for chapter in chapters:
        chapters[chapter] = "\n".join(chapters[chapter])
    return chapters

def extract_prescriptions(text):
    """提取药方和内容"""
    prescriptions = {}
    matches = re.split(r"(__.*?__)", text)
    current_prescription = None
    for match in matches:
        if match.startswith("__") and match.endswith("__"):
            current_prescription = match.strip("__ ")
            prescriptions[current_prescription] = []
        elif current_prescription:
            prescriptions[current_prescription].append(match.strip())
    # 合并每个药方的内容
    for prescription in prescriptions:
        prescriptions[prescription] = "\n".join(prescriptions[prescription])
    return prescriptions

def clean_text(text):
    """清理文本数据"""
    # 去除元数据标签
    text = re.sub(r"<book>.*?</book>", "", text, flags=re.DOTALL)
    # 去除多余的分隔符和空行
    text = re.sub(r"={5,}", "", text)
    # 去掉多余空白字符，保留中文、英文、数字和常见标点
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：\n]", "", text)
    # 移除多余的空行
    text = re.sub(r"\n\s*\n", "\n", text).strip()
    return text

def segment_text(text) -> str:
    return ' '.join(jieba.cut(text))

def remove_stopwords(text):
    stopwords = set()
    with open("../cn_stopwords.txt", "r", encoding='utf-8') as file:
        for line in file:
            stopwords.add(line.strip())
    
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

if __name__ == "__main__":
    texts = read_txt_files("../data")

    all_metadata = []
    all_chapters = []
    all_prescriptions = []

    # 提取元数据、章节和药方内容
    for text in tqdm(texts, desc="Extracting Metadata, Chapters, Prescriptions"):
        metadata = extract_metadata(text)
        chapters = extract_chapters(text)
        prescriptions = extract_prescriptions(text)
        
        all_metadata.append(metadata)
        all_chapters.append(chapters)
        all_prescriptions.append(prescriptions)

    cleaned_chapters = []
    for chapter_texts in tqdm(all_chapters, desc="Cleaning Chapters Texts"):
        cleaned = {chapter: clean_text(content) for chapter, content in chapter_texts.items()}
        cleaned_chapters.append(cleaned)

    cleaned_prescriptions = []
    for prescription_texts in tqdm(all_prescriptions, desc="Cleaning Prescriptions Texts"):
        cleaned = {prescription: clean_text(content) for prescription, content in prescription_texts.items()}
        cleaned_prescriptions.append(cleaned)

    # 繁体转简体
    converter = opencc.OpenCC('t2s')

    simplified_chapters = []
    for chapter_texts in tqdm(cleaned_chapters, desc="Simplifying Chapters Texts"):
        simplified = {chapter: converter.convert(content) for chapter, content in chapter_texts.items()}
        simplified_chapters.append(simplified)

    simplified_prescriptions = []
    for prescription_texts in tqdm(cleaned_prescriptions, desc="Simplifying Prescriptions Texts"):
        simplified = {prescription: converter.convert(content) for prescription, content in prescription_texts.items()}
        simplified_prescriptions.append(simplified)

    # 分词
    segmented_chapters = []
    for chapter_texts in tqdm(simplified_chapters, desc="Segmenting Chapters Texts"):
        segmented = {chapter: segment_text(content) for chapter, content in chapter_texts.items()}
        segmented_chapters.append(segmented)

    segmented_prescriptions = []
    for prescription_texts in tqdm(simplified_prescriptions, desc="Segmenting Prescriptions Texts"):
        segmented = {prescription: segment_text(content) for prescription, content in prescription_texts.items()}
        segmented_prescriptions.append(segmented)

    # 移除停用词
    final_chapters = []
    for chapter_texts in tqdm(segmented_chapters, desc="Removing Stopwords in Chapters"):
        final = {chapter: remove_stopwords(content) for chapter, content in chapter_texts.items()}
        final_chapters.append(final)

    final_prescriptions = []
    for prescription_texts in tqdm(segmented_prescriptions, desc="Removing Stopwords in Prescriptions"):
        final = {prescription: remove_stopwords(content) for prescription, content in prescription_texts.items()}
        final_prescriptions.append(final)

    # 保存到文件
    with open('metadata.json', 'w', encoding='utf-8') as file:
        json.dump(all_metadata, file, ensure_ascii=False, indent=4)

    with open('chapters.json', 'w', encoding='utf-8') as file:
        json.dump(final_chapters, file, ensure_ascii=False, indent=4)

    with open('prescriptions.json', 'w', encoding='utf-8') as file:
        json.dump(final_prescriptions, file, ensure_ascii=False, indent=4)



