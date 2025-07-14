import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer

from model import ConceptualBottleneckAutoencoder # 确保上面的模型代码保存在 model.py
from model import CombinedConceptAutoencoder
from model import BERT_MODEL_NAME
from load_dataset import dataset

# 1. 初始化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
MODEL = ConceptualBottleneckAutoencoder().to(DEVICE)

LEARNING_RATE = 5e-5
EPOCHS = 100 # 对于小数据集，多训练几轮
BATCH_SIZE = 128



def collate_fn(batch_sentences):
    # 使用tokenizer对一个批次的句子进行编码和填充
    # 我们需要同时为编码器和解码器准备输入
    # 在这个任务中，源和目标是相同的
    inputs = TOKENIZER(
        batch_sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    return {
        "src_input_ids": inputs["input_ids"],
        "src_attention_mask": inputs["attention_mask"],
        "tgt_input_ids": inputs["input_ids"].clone(), # 目标也是它自己
        "tgt_attention_mask": inputs["attention_mask"].clone()
    }

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print("数据加载器准备完毕！")
# 3. 优化器和损失函数
# 只优化需要梯度的参数（即解码器的参数）
optimizer = AdamW(filter(lambda p: p.requires_grad, MODEL.parameters()), lr=LEARNING_RATE)
# 忽略padding token的损失计算
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)

# 4. 训练循环
print("开始训练...")
MODEL.train() # 设置为训练模式
for epoch in range(EPOCHS):
    total_loss = 0
    # 为了显示进度，可以使用tqdm
    from tqdm import tqdm
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        # 将数据移动到GPU/CPU
        src_input_ids = batch["src_input_ids"].to(DEVICE)
        src_attention_mask = batch["src_attention_mask"].to(DEVICE)
        tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)
        tgt_attention_mask = batch["tgt_attention_mask"].to(DEVICE)

        # 准备好解码器的目标输出（向左移动一位）
        labels = tgt_input_ids[:, 1:].contiguous()
        # 解码器的输入也需要去掉最后一个词，以匹配标签的长度
        decoder_input_ids = tgt_input_ids[:, :-1].contiguous()
        decoder_attention_mask = tgt_attention_mask[:, :-1].contiguous()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = MODEL(
            src_input_ids,
            src_attention_mask,
            decoder_input_ids, # 使用移位后的输入
            decoder_attention_mask
        )
        
        # 计算损失
        # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        # labels: (batch, seq_len) -> (batch * seq_len)
        loss = loss_fn(logits.view(-1, MODEL.vocab_size), labels.view(-1))

        # 反向传播和优化
        loss.backward()

        torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0) # 梯度裁剪，防止梯度爆炸

        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

MODEL_SAVE_PATH="conceptual_autoencoder.pth"

# 5. 保存模型和验证
print("训练完成，正在保存模型...")
torch.save(MODEL.state_dict(), MODEL_SAVE_PATH)

print("\n--- 开始验证 ---")
# 加载已保存的模型权重
MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 5. 验证结果
print("\n--- 开始验证 ---")
test_sentence = "机器学习是人工智能的分支。"
reconstructed = MODEL.reconstruct(test_sentence, TOKENIZER)
print(f"原始句子: {test_sentence}")
print(f"重建句子: {reconstructed}")

test_sentence = "我喜欢看科幻电影。"
reconstructed = MODEL.reconstruct(test_sentence, TOKENIZER)
print(f"原始句子: {test_sentence}")
print(f"重建句子: {reconstructed}")