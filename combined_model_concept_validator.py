import torch
import numpy as np
import os
from scipy.spatial.distance import cosine
from transformers import BertTokenizer
from model import ConceptualBottleneckAutoencoder
from model import BERT_MODEL_NAME
class ConceptValidator:
    def __init__(self, model_path, device):
        self.device = device
        print(f"正在加载模型和分词器到 {self.device}...")
        
        # 1. 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        # 2. 加载您训练好的模型结构和权重
        self.model = ConceptualBottleneckAutoencoder().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # 切换到评估模式，这会禁用dropout等
        print("模型加载成功！")

    def get_concept_vector(self, sentence: str) -> np.ndarray:
        """提取单个句子的[CLS]概念向量。"""
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
            # 只需要编码器部分，所以我们直接调用 self.model.encoder
            encoder_outputs = self.model.encoder(**inputs)
            # 提取 [CLS] 向量
            cls_vector = encoder_outputs.last_hidden_state[:, 0, :].squeeze()
            return cls_vector.cpu().numpy()

    def get_bow_vector(self, sentence: str) -> np.ndarray:
        """
        计算一个基线向量：词袋向量（Bag-of-Words）。
        它只关心句子里有哪些词，忽略顺序和语法。
        """
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
            encoder_outputs = self.model.encoder(**inputs)
            last_hidden_states = encoder_outputs.last_hidden_state.squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
            
            # 我们只对非 [PAD] 的词元进行平均
            masked_hidden_states = last_hidden_states[attention_mask.bool()]
            bow_vector = torch.mean(masked_hidden_states, dim=0)
            return bow_vector.cpu().numpy()

    @staticmethod
    def calculate_similarity(vec1, vec2):
        """计算余弦相似度。注意 scipy 的 cosine 计算的是距离，所以要用 1 减去它。"""
        return 1 - cosine(vec1, vec2)

    def run_experiment_1_similarity_test(self):
        print("\n" + "="*50)
        print("实验一：同义/释义句相似度测试")
        print("="*50)
        
        # 定义测试用例
        a1 = "这台电脑的性能非常强大。"
        a2 = "这台计算机的运行速度很快。"
        a3 = "这部机器处理任务的能力很强。"
        b = "这台电脑的性能非常糟糕。"
        c = "今天的天气非常不错。"

        # 提取概念向量
        vec_a1 = self.get_concept_vector(a1)
        vec_a2 = self.get_concept_vector(a2)
        vec_a3 = self.get_concept_vector(a3)
        vec_b = self.get_concept_vector(b)
        vec_c = self.get_concept_vector(c)

        print(f"A1: {a1}")
        print(f"A2: {a2}")
        print(f"A3: {a3}")
        print(f"B:  {b}")
        print(f"C:  {c}\n")

        # 计算并打印相似度
        sim_a1_a2 = self.calculate_similarity(vec_a1, vec_a2)
        sim_a1_a3 = self.calculate_similarity(vec_a1, vec_a3)
        sim_a1_b = self.calculate_similarity(vec_a1, vec_b)
        sim_a1_c = self.calculate_similarity(vec_a1, vec_c)

        print(f"同义句相似度 Sim(A1, A2) (期望值: 非常高): {sim_a1_a2:.4f}")
        print(f"释义句相似度 Sim(A1, A3) (期望值: 非常高): {sim_a1_a3:.4f}")
        print(f"反义句相似度 Sim(A1, B) (期望值: 较低):   {sim_a1_b:.4f}")
        print(f"无关句相似度 Sim(A1, C) (期望值: 非常低): {sim_a1_c:.4f}")

    def run_experiment_2_neighbor_search(self):
        print("\n" + "="*50)
        print("实验二：概念空间邻近搜索")
        print("="*50)

        # 建立一个小的“概念数据库”
        corpus = [
            "如何才能有效地学习一门新的编程语言？",
            "学习Python的最佳方法是什么？",
            "我想快速掌握Java编程技巧。",
            "金融市场最近的趋势如何？",
            "语言是人类交流的工具。",
            "我应该如何开始学习Go语言？",
            "投资股票有什么风险？",
            "我喜欢在图书馆学习。"
        ]
        
        corpus_vectors = [self.get_concept_vector(s) for s in corpus]
        
        query_sentence = "有没有掌握新编程技能的诀窍？"
        print(f"查询语句: {query_sentence}\n")
        query_vector = self.get_concept_vector(query_sentence)

        # 计算查询向量与数据库中所有向量的相似度
        similarities = [self.calculate_similarity(query_vector, v) for v in corpus_vectors]
        
        # 排序并找到最相似的句子
        sorted_results = sorted(zip(similarities, corpus), key=lambda item: item[0], reverse=True)
        
        print("Top 3 相似的句子 (期望是关于学习编程的):")
        for i in range(3):
            print(f"  {i+1}. “{sorted_results[i][1]}” (相似度: {sorted_results[i][0]:.4f})")

    def run_experiment_3_bow_comparison(self):
        print("\n" + "="*50)
        print("实验三：与词袋模型(BoW)的关键对比")
        print("="*50)
        
        s1 = "猫追着老鼠跑。"
        s2 = "老鼠追着猫跑。"
        print(f"S1: {s1}")
        print(f"S2: {s2}")
        print("这两个句子的词汇完全相同，但概念完全相反。\n")

        # 使用 [CLS] 概念向量
        concept_vec_s1 = self.get_concept_vector(s1)
        concept_vec_s2 = self.get_concept_vector(s2)
        concept_sim = self.calculate_similarity(concept_vec_s1, concept_vec_s2)
        print(f"使用 [CLS] 概念向量的相似度 (期望值: 低): {concept_sim:.4f}")
        print("  -> 这表明模型理解了句子的结构和主谓宾关系。")

        # 使用词袋 (BoW) 基线向量
        bow_vec_s1 = self.get_bow_vector(s1)
        bow_vec_s2 = self.get_bow_vector(s2)
        bow_sim = self.calculate_similarity(bow_vec_s1, bow_vec_s2)
        print(f"使用 BoW (词袋) 向量的相似度 (期望值: 非常高): {bow_sim:.4f}")
        print("  -> 这表明BoW模型只看到了相同的词，忽略了它们之间的关系。")


from train_combined_model import MODEL_SAVE_PATH
MODEL_FILE = MODEL_SAVE_PATH

if not os.path.exists(MODEL_FILE):
    print(f"错误: 找不到模型文件 '{MODEL_FILE}'。请先运行 train.py 进行训练。")
else:
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建验证器实例
    validator = ConceptValidator(model_path=MODEL_FILE, device=device)
    
    # 依次运行所有实验
    validator.run_experiment_1_similarity_test()
    validator.run_experiment_2_neighbor_search()
    validator.run_experiment_3_bow_comparison()