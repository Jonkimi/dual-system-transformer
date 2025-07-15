import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer
# from datasets import load_dataset
import os
from tqdm import tqdm
from model import CombinedConceptAutoencoderV2

# --- 2. 数据准备部分 ---

# from load_dataset import dataset
MODEL_SAVE_PATH = "advanced_concept_autoencoder.pth"


if __name__ == '__main__':
    NUM_CONCEPTS = 16  # 提取的局部概念数量
    BERT_MODEL_NAME='hfl/chinese-bert-wwm-ext'
    # BERT_MODEL_NAME='bert-base-chinese'
    MAX_SENTENCE_LEN=256
    # 步骤1: 准备数据
    # prepare_data(DATA_FILE)

    # 步骤2: 初始化模型、分词器和设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    MODEL = CombinedConceptAutoencoderV2(num_concepts=NUM_CONCEPTS).to(DEVICE)
    
    # 步骤6: 保存模型和验证
    # print("训练完成，正在保存模型...")
    MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH))
    MODEL.eval()

    print("\n--- 开始验证 ---")
    test_sentences = [
        "人类的未来将由人工智能决定。",
        "这辆红色的跑车在赛道上疾驰。",
        "机器学习模型需要大量数据进行训练。",
        "傍晚的日落给天空染上了一层金色。",
        "今天加班不",
    ]
    
    for sentence in test_sentences:
        reconstructed = MODEL.reconstruct(sentence, TOKENIZER, max_len=MAX_SENTENCE_LEN)
        print("-" * 20)
        print(f"原始句子: {sentence}")
        print(f"重建句子: {reconstructed}")

    while True:
        sentence = input("请输入一个句子 (输入'quit'退出): ")
        if sentence.lower() == 'quit':
            break
        reconstructed = MODEL.reconstruct(sentence, TOKENIZER, max_len=MAX_SENTENCE_LEN)
        print("-" * 20)
        print(f"原始句子: {sentence}")
        print(f"重建句子: {reconstructed}")
    
