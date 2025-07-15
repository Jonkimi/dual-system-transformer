import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
BERT_MODEL_NAME='hfl/chinese-bert-wwm-ext'
class ConceptualBottleneckAutoencoder(nn.Module):
    #bert-base-chinese
    def __init__(self, bert_model_name=BERT_MODEL_NAME, decoder_layers=3, hidden_dim=768, num_heads=8):
        super().__init__()
        
        # 1. 系统1：预训练的、冻结的BERT编码器
        self.encoder = BertModel.from_pretrained(bert_model_name)
        # 冻结BERT的所有参数，我们不训练它
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 获取词汇表大小，用于解码器的输出层
        self.vocab_size = self.encoder.config.vocab_size

        # 2. 系统2：待训练的Transformer解码器
        # 解码器的词嵌入层
        self.decoder_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        
        # 标准的Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            batch_first=True # 非常重要，让输入/输出的批次维度在前
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
        # 3. 输出层：将解码器的输出映射回词汇表
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        """
        前向传播函数
        src: 源句子，用于编码器 (BERT)
        tgt: 目标句子，用于解码器（在训练时是源句本身）
        """
        # --- 系统1：编码过程 ---
        # 编码源句子
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask)
        # 提取[CLS]向量作为“概念向量”，并将其作为解码器的“记忆”
        # BERT输出的last_hidden_state维度是 (batch_size, seq_len, hidden_dim)
        # 我们取所有句子的第一个token ([CLS]) 的向量
        concept_vector = encoder_outputs.last_hidden_state[:, 0, :].unsqueeze(0)
        # 解码器的memory需要 (seq_len, batch_size, hidden_dim) 的格式，所以我们调整一下维度
        # 在这里，源序列长度是1，因为它只有一个概念向量
        memory = concept_vector.permute(1, 0, 2)

        # --- 系统2：解码过程 ---
        # 准备解码器的输入（目标句子的词嵌入）
        tgt_emb = self.decoder_embedding(tgt_input_ids)
        
        # 创建解码器需要的未来信息遮罩 (causal mask)
        tgt_seq_len = tgt_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt_input_ids.device)

        # 将概念向量(memory)和目标嵌入输入解码器
        # tgt_key_padding_mask 用于忽略目标句子中的padding部分
        decoder_output = self.decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=~tgt_attention_mask.bool() # 注意取反
        )
        
        # 通过最后的线性层得到词汇表的logit分布
        logits = self.output_layer(decoder_output)
        
        return logits

    def reconstruct(self, sentence, tokenizer, max_len=50):
        """用于推理和验证的函数"""
        self.eval() # 设置为评估模式
        device = next(self.parameters()).device

        # 1. 编码输入句子，得到概念向量
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            encoder_outputs = self.encoder(**inputs)
            concept_vector = encoder_outputs.last_hidden_state[:, 0, :].unsqueeze(0)
            memory = concept_vector.permute(1, 0, 2)

        # 2. 准备开始解码
        # 从 [CLS] token 开始生成
        generated_ids = torch.LongTensor([[tokenizer.cls_token_id]]).to(device)

        # 3. 自回归生成
        for _ in range(max_len - 1):
            tgt_emb = self.decoder_embedding(generated_ids)
            
            # 不需要mask，因为我们是一步步生成的
            decoder_output = self.decoder(tgt=tgt_emb, memory=memory)
            
            # 只取最后一个时间步的输出，通过输出层得到logits
            last_token_logits = self.output_layer(decoder_output[:, -1, :])
            
            # 使用argmax进行贪心解码，选出最可能的下一个token
            next_token_id = torch.argmax(last_token_logits, dim=-1)
            
            # 将新生成的token添加到序列中
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

            # 如果生成了[SEP] token，则停止
            if next_token_id.item() == tokenizer.sep_token_id:
                break
        
        # 将生成的ID序列转换回文本
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    

# --- 1. 模型架构定义 ---

class ConceptPoolingHead(nn.Module):
    """
    通过可学习的“查询向量”从BERT输出中提取多个概念向量。
    """
    def __init__(self, num_concepts=5, hidden_dim=768, num_heads=8):
        super().__init__()
        # 1. 定义 k 个可学习的“概念查询向量”
        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, hidden_dim))
        
        # 2. 定义一个交叉注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, bert_outputs):
        # bert_outputs 的维度: (batch_size, seq_len, hidden_dim)
        # self.concept_queries 的维度: (1, num_concepts, hidden_dim)
        queries = self.concept_queries.repeat(bert_outputs.size(0), 1, 1)
        
        # Q=queries, K=bert_outputs, V=bert_outputs
        # 输出的维度: (batch_size, num_concepts, hidden_dim)
        concept_vectors, _ = self.attention(query=queries, key=bert_outputs, value=bert_outputs)
        return concept_vectors

class CombinedConceptAutoencoder(nn.Module):
    """
    结合了全局[CLS]向量和多个局部概念向量的自编码器模型。
    """
    def __init__(self, num_concepts=5, bert_model_name=BERT_MODEL_NAME, decoder_layers=6, hidden_dim=768, num_heads=8):
        super().__init__()
        # 系统1: BERT 编码器
        print("正在初始化BERT编码器...")
        self.encoder = BertModel.from_pretrained(bert_model_name)
        # 冻结BERT的所有参数
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.vocab_size = self.encoder.config.vocab_size
        
        # 多概念提取头
        print(f"正在初始化概念池化头 (提取 {num_concepts} 个概念)...")
        self.concept_pooling = ConceptPoolingHead(num_concepts=num_concepts, hidden_dim=hidden_dim, num_heads=num_heads)
        
        # 系统2: Transformer 解码器
        print(f"正在初始化Transformer解码器 ({decoder_layers} 层)...")
        self.decoder_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)
        print("模型初始化完成。")

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        # 1. BERT编码
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask)
        bert_hidden_states = encoder_outputs.last_hidden_state
        
        # 2. 提取两种概念向量并拼接
        # a. 全局概念向量 ([CLS] vector)
        global_concept_vector = bert_hidden_states[:, 0, :].unsqueeze(1)
        
        # b. 多个局部概念向量
        local_concept_vectors = self.concept_pooling(bert_hidden_states)
        
        # c. 在序列维度上拼接，形成统一的记忆区
        memory = torch.cat([global_concept_vector, local_concept_vectors], dim=1)
        
        # 3. GPT解码
        tgt_emb = self.decoder_embedding(tgt_input_ids)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input_ids.size(1)).to(tgt_input_ids.device)
        
        decoder_output = self.decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~tgt_attention_mask.bool()
        )
        
        logits = self.output_layer(decoder_output)
        return logits

    def reconstruct(self, sentence, tokenizer, max_len=50):
        """用于推理和验证的函数"""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # 1. 编码并提取组合概念
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            encoder_outputs = self.encoder(**inputs)
            bert_hidden_states = encoder_outputs.last_hidden_state
            
            global_concept_vector = bert_hidden_states[:, 0, :].unsqueeze(1)
            local_concept_vectors = self.concept_pooling(bert_hidden_states)
            memory = torch.cat([global_concept_vector, local_concept_vectors], dim=1)

            # 2. 自回归生成
            generated_ids = torch.LongTensor([[tokenizer.cls_token_id]]).to(device)

            for _ in range(max_len - 1):
                tgt_emb = self.decoder_embedding(generated_ids)
                decoder_output = self.decoder(tgt=tgt_emb, memory=memory)
                last_token_logits = self.output_layer(decoder_output[:, -1, :])
                next_token_id = torch.argmax(last_token_logits, dim=-1)
                
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

                if next_token_id.item() == tokenizer.sep_token_id:
                    break
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


class CombinedConceptAutoencoderV2(nn.Module):
    """
    结合了全局[CLS]向量和多个局部概念向量的自编码器模型。
    """
    def __init__(self, num_concepts=16, concept_encoder_layers=6, bert_model_name=BERT_MODEL_NAME, decoder_layers=6, hidden_dim=768, num_heads=8):
        super().__init__()
        # 系统1: BERT 编码器
        print("正在初始化BERT编码器...")
        self.encoder = BertModel.from_pretrained(bert_model_name)
        # 冻结BERT的所有参数
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.vocab_size = self.encoder.config.vocab_size
        
        # 多概念提取头
    # 系统1b: 高级概念编码器 (核心创新)
        print(f"初始化高级ConceptEncoder ({concept_encoder_layers}层, {num_concepts}类概念)...")
        self.concept_encoder = ConceptEncoder(
            num_concepts=num_concepts, 
            num_layers=concept_encoder_layers, # 例如，使用4层
            hidden_dim=768, 
            num_heads=8
        )
        # 系统2: Transformer 解码器
        print(f"正在初始化Transformer解码器 ({decoder_layers} 层)...")
        self.decoder_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)
        print("模型初始化完成。")

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        # 1. BERT编码
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask)
        bert_hidden_states = encoder_outputs.last_hidden_state
        
        # 2. 提取两种概念向量并拼接
#         # memory 的形状是 (batch_size, num_concepts, 768)
        memory = self.concept_encoder(bert_hidden_states)
        
        # 3. GPT解码
        tgt_emb = self.decoder_embedding(tgt_input_ids)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input_ids.size(1)).to(tgt_input_ids.device)
        
        decoder_output = self.decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~tgt_attention_mask.bool()
        )
        
        logits = self.output_layer(decoder_output)
        return logits

    def reconstruct(self, sentence, tokenizer, max_len=50):
        """用于推理和验证的函数"""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # 1. 编码并提取组合概念
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            encoder_outputs = self.encoder(**inputs)
            bert_hidden_states = encoder_outputs.last_hidden_state
            
            # global_concept_vector = bert_hidden_states[:, 0, :].unsqueeze(1)
            # local_concept_vectors = self.concept_pooling(bert_hidden_states)
            # memory = torch.cat([global_concept_vector, local_concept_vectors], dim=1)
            memory = self.concept_encoder(bert_hidden_states)

            # 2. 自回归生成
            generated_ids = torch.LongTensor([[tokenizer.cls_token_id]]).to(device)

            for _ in range(max_len - 1):
                tgt_emb = self.decoder_embedding(generated_ids)
                decoder_output = self.decoder(tgt=tgt_emb, memory=memory)
                last_token_logits = self.output_layer(decoder_output[:, -1, :])
                next_token_id = torch.argmax(last_token_logits, dim=-1)
                
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

                if next_token_id.item() == tokenizer.sep_token_id:
                    break
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
class ConceptEncoder(nn.Module):
    def __init__(self, num_concepts=8, num_layers=4, hidden_dim=768, num_heads=8):
        """
        一个多层的Transformer概念编码器。
        
        Args:
            num_concepts (int): 您希望提取的固定概念数量 (k)。
            num_layers (int): “研讨会”的轮数，即Encoder的层数。
            hidden_dim (int): 模型的隐藏维度，应与BERT匹配。
            num_heads (int): 多头注意力的头数。
        """
        super().__init__()
        self.num_concepts = num_concepts

        # 1. 初始化 k 个可学习的“专家”查询向量
        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, hidden_dim))

        # 2. 定义“研讨会”的核心流程
        # 我们使用 TransformerDecoderLayer 因为它同时包含 self-attn 和 cross-attn
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            batch_first=True,
            dim_feedforward=hidden_dim * 4 # 通常是隐藏维度的4倍
        )
        
        # 3. 将单层流程堆叠成多层
        self.transformer_encoder = nn.TransformerDecoder(
            decoder_layer=encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, bert_hidden_states):
        """
        前向传播。
        
        Args:
            bert_hidden_states (Tensor): 来自BERT的输出, 形状 (batch_size, seq_len, hidden_dim)。
        
        Returns:
            Tensor: 精炼后的概念向量集, 形状 (batch_size, num_concepts, hidden_dim)。
        """
        batch_size = bert_hidden_states.size(0)
        
        # 为批次中的每个样本准备查询向量
        # 形状: (batch_size, num_concepts, hidden_dim)
        queries = self.concept_queries.repeat(batch_size, 1, 1)

        # 执行“研讨会”
        # tgt (Target): 我们的查询向量，它将进行自我注意并被更新。
        # memory (Memory): BERT的输出，作为只读的参考资料。
        refined_concepts = self.transformer_encoder(tgt=queries, memory=bert_hidden_states)
        
        return refined_concepts
