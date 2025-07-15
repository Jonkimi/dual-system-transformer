import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist # 用于计算成对距离
from scipy.optimize import linear_sum_assignment # 用于匈牙利算法匹配
from transformers import BertTokenizer
from train_combined_model import CombinedConceptAutoencoder # 从您的训练脚本中导入模型
from model import BERT_MODEL_NAME
class MultiConceptValidator:
    def __init__(self, model_path, device):
        self.device = device
        print(f"正在加载模型和分词器到 {self.device}...")
        
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        # 加载我们训练好的组合概念模型
        # 注意：这里的num_concepts需要和训练时一致
        self.model = CombinedConceptAutoencoder(num_concepts=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("模型加载成功！")

    def get_all_concepts(self, sentence: str):
        """提取一个句子的所有概念向量（全局+局部）。"""
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
            encoder_outputs = self.model.encoder(**inputs)
            bert_hidden_states = encoder_outputs.last_hidden_state
            
            global_vec = bert_hidden_states[:, 0, :].unsqueeze(1)
            local_vecs = self.model.concept_pooling(bert_hidden_states)
            
            combined_vecs = torch.cat([global_vec, local_vecs], dim=1)
            return combined_vecs.squeeze(0).cpu().numpy()

    @staticmethod
    def calculate_set_similarity(set_A, set_B):
        """
        计算两组向量集合之间的最佳匹配相似度。
        使用匈牙利算法找到最佳匹配，避免顺序问题。
        """
        # 计算两个集合中所有向量对之间的余弦距离
        # cdist 输出的是距离(0-2)，所以相似度是 1 - 距离
        cosine_dist_matrix = cdist(set_A, set_B, 'cosine')
        
        # 使用匈牙利算法找到最小代价匹配（即最大相似度匹配）
        row_ind, col_ind = linear_sum_assignment(cosine_dist_matrix)
        
        # 计算最佳匹配的平均相似度
        min_distance_sum = cosine_dist_matrix[row_ind, col_ind].sum()
        num_matches = len(row_ind)
        avg_distance = min_distance_sum / num_matches
        avg_similarity = 1 - avg_distance
        
        return avg_similarity

    def run_experiment_1_synonym_set_test(self):
        print("\n" + "="*50)
        print("实验一：同义句的“概念集合”相似度测试")
        print("="*50)
        
        a1 = "这台电脑的性能非常强大。"
        a2 = "这部计算机的运行效率极高。"
        b = "这台电脑的外观非常漂亮。"
        c = "今天的天气非常不错。"

        concepts_a1 = self.get_all_concepts(a1)
        concepts_a2 = self.get_all_concepts(a2)
        concepts_b = self.get_all_concepts(b)
        concepts_c = self.get_all_concepts(c)

        sim_a1_a2 = self.calculate_set_similarity(concepts_a1, concepts_a2)
        sim_a1_b = self.calculate_set_similarity(concepts_a1, concepts_b)
        sim_a1_c = self.calculate_set_similarity(concepts_a1, concepts_c)

        print(f"同义句集相似度 Sim(A1, A2) (期望值: 非常高): {sim_a1_a2:.4f}")
        print(f"主题相关句集相似度 Sim(A1, B) (期望值: 中等): {sim_a1_b:.4f}")
        print(f"无关句集相似度 Sim(A1, C) (期望值: 非常低): {sim_a1_c:.4f}")

    def run_experiment_2_attention_visualization(self):
        """可视化一个概念向量到底关注了哪些词。"""
        print("\n" + "="*50)
        print("实验二：概念向量的注意力可视化")
        print("="*50)

        sentence = "在暴风雨的夜晚，那位戴着礼帽的老侦探仔细检查着一把银色的匕首。"
        print(f"分析句子: “{sentence}”")

        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            # 我们需要获取注意力权重，所以需要修改一下模型的前向传播逻辑
            # 这里我们直接访问底层的 attention 模块
            encoder_outputs = self.model.encoder(**inputs).last_hidden_state
            queries = self.model.concept_pooling.concept_queries.repeat(encoder_outputs.size(0), 1, 1)
            
            # MultiheadAttention模块可以返回注意力权重
            _, attn_weights = self.model.concept_pooling.attention(
                query=queries, 
                key=encoder_outputs, 
                value=encoder_outputs,
                attn_mask=None
            )
            # attn_weights 的形状是 (batch_size, num_concepts, seq_len)
            
        attn_weights = attn_weights.squeeze(0).cpu().numpy()

        # 绘制热力图
        plt.figure(figsize=(12, 6))
        # 使用中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False

        sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=[f"概念_{i+1}" for i in range(attn_weights.shape[0])], cmap="viridis")
        plt.title("概念向量对输入词元的注意力热力图")
        plt.xlabel("输入句子的词元 (Tokens)")
        plt.ylabel("概念查询向量 (Concept Queries)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存或显示图像
        plt.savefig("attention_visualization.png")
        print("注意力热力图已保存为 'attention_visualization.png'。请查看该图片。")
        print("观察图表：一个好的概念向量应该将注意力集中在语义相关的词组上。")

    def run_experiment_3_structure_sensitivity_test(self):
        print("\n" + "="*50)
        print("实验三：句子结构敏感性测试")
        print("="*50)
        
        s1 = "猫追着老鼠跑。"
        s2 = "老鼠追着猫跑。"
        
        concepts_s1 = self.get_all_concepts(s1)
        concepts_s2 = self.get_all_concepts(s2)

        similarity = self.calculate_set_similarity(concepts_s1, concepts_s2)

        print(f"“猫追老鼠” vs “老鼠追猫” 的概念集相似度 (期望值: 不会非常高): {similarity:.4f}")
        print("  -> 如果相似度远低于1.0，说明模型区分了这两个不同的场景。")


if __name__ == '__main__':
    MODEL_FILE = "combined_concept_autoencoder.pth"
    
    if not os.path.exists(MODEL_FILE):
        print(f"错误: 找不到模型文件 '{MODEL_FILE}'。请先运行 train_combined_model.py 进行训练。")
    else:
        # 安装必要的库
        try:
            import matplotlib
            import seaborn
        except ImportError:
            print("请先安装可视化库: pip install matplotlib seaborn")
            exit()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validator = MultiConceptValidator(model_path=MODEL_FILE, device=device)
        
        validator.run_experiment_1_synonym_set_test()
        validator.run_experiment_2_attention_visualization()
        validator.run_experiment_3_structure_sensitivity_test()
