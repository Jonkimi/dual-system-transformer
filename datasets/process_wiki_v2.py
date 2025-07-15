import os
import re
import json
import time
from opencc import OpenCC
from tqdm import tqdm

def process_wikiextractor_output(extracted_dir, merged_corpus_path, sentences_output_path):
    """
    处理wikiextractor输出的完整流水线。
    步骤1: 合并所有JSON文件，提取文本并进行繁简转换。
    步骤2: 将合并后的文本切分成独立的句子。
    """
    
    # --- 步骤 1: 合并、提取、转换 ---
    
    if not os.path.exists(merged_corpus_path):
        print(f"步骤 1/2: 开始合并、提取和转换来自 '{extracted_dir}' 的文件...")
        start_time = time.time()
        
        cc = OpenCC('t2s')
        article_count = 0
        
        with open(merged_corpus_path, 'w', encoding='utf-8') as outfile:
            # 遍历wikiextractor生成的所有子目录和文件
            for root, _, files in os.walk(extracted_dir):
                for file in files:
                    if file.startswith('wiki_'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                # 每行是一个JSON对象，包含一篇文章
                                article = json.loads(line)
                                text = article['text']
                                
                                # 进行繁简转换并写入
                                simplified_text = cc.convert(text)
                                outfile.write(simplified_text + '\n')
                                article_count += 1

            print(f"  已合并和转换 {article_count} 篇文章。")

        end_time = time.time()
        print(f"步骤 1/2 完成！耗时 {(end_time - start_time) / 60:.2f} 分钟。")
        print(f"合并后的语料库已保存至: {merged_corpus_path}")
    else:
        print(f"步骤 1/2 已跳过，文件 '{merged_corpus_path}' 已存在。")

    # --- 步骤 2: 清洗与句子切分 (与之前的脚本逻辑相同) ---
    
    if not os.path.exists(sentences_output_path):
        print(f"\n步骤 2/2: 开始将文本切分成句子...")
        start_time = time.time()
        
        sentence_delimiters = re.compile('([。？！…])')
        sentence_count = 0
        
        with open(merged_corpus_path, 'r', encoding='utf-8') as infile, \
             open(sentences_output_path, 'w', encoding='utf-8') as outfile:
            
            # 使用tqdm来显示处理进度条
            for line in tqdm(infile, desc="正在切分句子"):
                line = line.strip()
                if not line:
                    continue
                
                # 去除维基百科中常见的空标题标记
                line = re.sub(r'={2,}\s*.*?\s*={2,}', '', line)
                line = line.strip()

                sentences = sentence_delimiters.split(line)
                
                temp_sentence = ""
                for part in sentences:
                    if part in '。？！…':
                        temp_sentence += part
                        clean_s = temp_sentence.strip().replace('\n', '')
                        
                        if len(clean_s) > 10: # 提高长度阈值，过滤掉更多噪声
                            outfile.write(clean_s + '\n')
                            sentence_count += 1
                        
                        temp_sentence = ""
                    else:
                        temp_sentence += part
        
        end_time = time.time()
        print(f"步骤 2/2 完成！共生成 {sentence_count} 条句子，耗时 {(end_time - start_time) / 60:.2f} 分钟。")
        print(f"最终的句子数据集已保存至: {sentences_output_path}")
    else:
        print(f"步骤 2/2 已跳过，文件 '{sentences_output_path}' 已存在。")


if __name__ == '__main__':
    # --- 参数配置 ---
    # wikiextractor的输出目录
    EXTRACTED_WIKI_DIR = 'extracted_wiki'
    
    # 中间文件：合并后的纯文本（带标点）
    MERGED_CORPUS_FILE = 'wiki_merged_simplified.txt'
    
    # 最终输出文件
    FINAL_SENTENCES_FILE = 'wiki_sentences_for_training_v2.txt'
    
    if not os.path.exists(EXTRACTED_WIKI_DIR):
        print(f"错误: 找不到wikiextractor的输出目录 '{EXTRACTED_WIKI_DIR}'。")
        print("请先运行 'wikiextractor' 命令进行提取。")
    else:
        process_wikiextractor_output(EXTRACTED_WIKI_DIR, MERGED_CORPUS_FILE, FINAL_SENTENCES_FILE)
        print("\n所有处理已完成！现在您可以使用 'wiki_sentences_for_training_v2.txt' 进行训练。")