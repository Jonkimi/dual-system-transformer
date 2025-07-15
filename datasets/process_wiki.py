import os
import re
import time
from gensim.corpora import WikiCorpus
from opencc import OpenCC

def process_wikipedia_dump(dump_file_path, processed_corpus_path, sentences_output_path):
    """
    一个完整处理维基百科XML Dump的流水线函数。
    步骤1: 从XML提取纯文本并转换为简体中文。
    步骤2: 将纯文本切分成独立的句子。
    """
    
    # --- 步骤 1: 提取纯文本并进行繁简转换 ---
    
    if not os.path.exists(processed_corpus_path):
        print("步骤 1/2: 开始从XML Dump中提取和转换文本...")
        start_time = time.time()
        
        # 创建一个WikiCorpus对象
        # lemmatize=False 表示不进行词形还原，保留原文
        wiki = WikiCorpus(dump_file_path, dictionary={})
        
        # 初始化繁转简转换器
        cc = OpenCC('t2s')
        
        article_count = 0
        with open(processed_corpus_path, 'w', encoding='utf-8') as output_file:
            # get_texts() 是一个生成器，逐篇返回文章内容
            for text in wiki.get_texts():
                # 将文章中的词语列表连接成一个字符串，并进行繁简转换
                article_content = cc.convert(' '.join(text))
                output_file.write(article_content + '\n')
                
                article_count += 1
                if article_count % 1000 == 0:
                    print(f"  已处理 {article_count} 篇文章...")
                    
        end_time = time.time()
        print(f"步骤 1/2 完成！共处理 {article_count} 篇文章，耗时 {(end_time - start_time) / 60:.2f} 分钟。")
        print(f"纯文本语料库已保存至: {processed_corpus_path}")
    else:
        print(f"步骤 1/2 已跳过，文件 '{processed_corpus_path}' 已存在。")

    # --- 步骤 2: 清洗文本并切分成句子 ---
    
    if not os.path.exists(sentences_output_path):
        print("\n步骤 2/2: 开始将纯文本切分成句子...")
        start_time = time.time()
        
        # 定义句子切分的分隔符：句号、问号、感叹号、省略号
        # 我们使用 re.compile 来提高正则表达式的效率
        sentence_delimiters = re.compile('([。？！…])')
        
        sentence_count = 0
        with open(processed_corpus_path, 'r', encoding='utf-8') as input_file, \
             open(sentences_output_path, 'w', encoding='utf-8') as output_file:
            
            for line in input_file:
                # 1. 初步清洗：去除一些维基百科特有的标记和多余空格
                # 去除 <doc ...> 和 </doc> 标签（gensim有时会留下这个）
                line = re.sub(r'</?doc.*>', '', line)
                # 去除多余的空格和换行符
                line = line.strip()
                
                if not line:
                    continue
                
                # 2. 切分句子
                # 我们在分隔符后也保留它，所以用()将分隔符括起来
                sentences = sentence_delimiters.split(line)
                
                # 3. 后处理和写入
                # split后，分隔符会成为独立的元素，需要和前面的句子合并
                # 例如 ['句子1', '。', '句子2', '。', '']
                temp_sentence = ""
                for part in sentences:
                    if part in '。？！…':
                        temp_sentence += part
                        
                        # 清洗最终的句子
                        clean_s = temp_sentence.strip()
                        
                        # 过滤掉太短的无意义片段 (例如只有标题或单个词)
                        if len(clean_s) > 8: # 长度阈值可以调整
                            output_file.write(clean_s + '\n')
                            sentence_count += 1
                        
                        temp_sentence = "" # 重置
                    else:
                        temp_sentence += part

                if sentence_count > 0 and sentence_count % 100000 == 0:
                     print(f"  已生成 {sentence_count} 条句子...")

        end_time = time.time()
        print(f"步骤 2/2 完成！共生成 {sentence_count} 条句子，耗时 {(end_time - start_time) / 60:.2f} 分钟。")
        print(f"最终的句子数据集已保存至: {sentences_output_path}")
    else:
        print(f"步骤 2/2 已跳过，文件 '{sentences_output_path}' 已存在。")


if __name__ == '__main__':
    # --- 参数配置 ---
    # 您下载的维基百科dump文件路径
    WIKI_DUMP_FILE = 'zhwiki-latest-pages-articles.xml.bz2'
    
    # 中间处理文件：保存从XML中提取出的纯文本
    PROCESSED_CORPUS_FILE = 'wiki_corpus_zh_simplified.txt'
    
    # 最终输出文件：每行一个干净的句子
    FINAL_SENTENCES_FILE = 'wiki_sentences_for_training.txt'
    
    if not os.path.exists(WIKI_DUMP_FILE):
        print(f"错误: 找不到维基百科Dump文件 '{WIKI_DUMP_FILE}'。")
        print("请先从 https://dumps.wikimedia.org/zhwiki/latest/ 下载，并放置在当前目录。")
    else:
        process_wikipedia_dump(WIKI_DUMP_FILE, PROCESSED_CORPUS_FILE, FINAL_SENTENCES_FILE)
        print("\n所有处理已完成！")