import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer
from datasets import load_dataset
import os
from tqdm import tqdm
from model import ConceptPoolingHead
from model import CombinedConceptAutoencoder

# --- 2. 数据准备部分 ---

from load_dataset import dataset
MODEL_SAVE_PATH = "combined_concept_autoencoder.pth"
# --- 3. 训练主逻辑 ---

if __name__ == '__main__':
    # 超参数设置
    DATA_FILE = "sentences_nli_zh.txt"

    NUM_CONCEPTS = 8  # 提取的局部概念数量
    LEARNING_RATE = 1e-4 # 可以比之前稍高一些
    EPOCHS = 50 # 使用大数据集，少量epoch即可看到效果
    BATCH_SIZE = 256 # 根据你的GPU显存调整
    MAX_SENTENCE_LEN = 256 # 限制句子长度以节省显存
    BERT_MODEL_NAME='hfl/chinese-bert-wwm-ext'
    # BERT_MODEL_NAME='bert-base-chinese'

    # 步骤1: 准备数据
    # prepare_data(DATA_FILE)

    # 步骤2: 初始化模型、分词器和设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    MODEL = CombinedConceptAutoencoder(num_concepts=NUM_CONCEPTS).to(DEVICE)
    
    # 步骤3: 创建数据加载器
    # dataset = 
    
    def collate_fn(batch_sentences):
        inputs = TOKENIZER(
            batch_sentences, 
            padding=True, 
            truncation=True, 
            max_length=MAX_SENTENCE_LEN,
            return_tensors="pt"
        )
        return {
            "src_input_ids": inputs["input_ids"],
            "src_attention_mask": inputs["attention_mask"],
            "tgt_input_ids": inputs["input_ids"].clone(),
            "tgt_attention_mask": inputs["attention_mask"].clone()
        }

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 步骤4: 定义优化器和损失函数
    optimizer = AdamW(filter(lambda p: p.requires_grad, MODEL.parameters()), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)

    # 步骤5: 训练循环
    print(f"\n--- 开始在 {DEVICE} 上训练 ---")
    MODEL.train()
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            src_input_ids = batch["src_input_ids"].to(DEVICE)
            src_attention_mask = batch["src_attention_mask"].to(DEVICE)
            tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)
            tgt_attention_mask = batch["tgt_attention_mask"].to(DEVICE)

            labels = tgt_input_ids[:, 1:].contiguous()
            decoder_input_ids = tgt_input_ids[:, :-1].contiguous()
            decoder_attention_mask = tgt_attention_mask[:, :-1].contiguous()
            
            optimizer.zero_grad()
            
            logits = MODEL(
                src_input_ids,
                src_attention_mask,
                decoder_input_ids,
                decoder_attention_mask
            )
            
            loss = loss_fn(logits.view(-1, MODEL.vocab_size), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    # 步骤6: 保存模型和验证
    print("训练完成，正在保存模型...")
    torch.save(MODEL.state_dict(), MODEL_SAVE_PATH)

    print("\n--- 开始验证 ---")
    test_sentences = [
        "人类的未来将由人工智能决定。",
        "这辆红色的跑车在赛道上疾驰。",
        "机器学习模型需要大量数据进行训练。",
        "傍晚的日落给天空染上了一层金色。"
    ]
    
    for sentence in test_sentences:
        reconstructed = MODEL.reconstruct(sentence, TOKENIZER, max_len=MAX_SENTENCE_LEN)
        print("-" * 20)
        print(f"原始句子: {sentence}")
        print(f"重建句子: {reconstructed}")
