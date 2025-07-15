import torch
import numpy as np
import os
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
# -- 确保这些行存在 --
from scipy.spatial.distance import cdist 
from scipy.optimize import linear_sum_assignment
from transformers import BertTokenizer
from model import CombinedConceptAutoencoderV2
from model import BERT_MODEL_NAME
# validate_advanced_model.py (FIXED VERSION)


class AdvancedConceptValidator:
    def __init__(self, model_path, num_concepts, concept_encoder_layers, device):
        self.device = device
        print(f"正在加载模型和分词器到 {self.device}...")
        
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        self.model = CombinedConceptAutoencoderV2(
            num_concepts=num_concepts,
            concept_encoder_layers=concept_encoder_layers
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("模型加载成功！")

    def get_concepts(self, sentence: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
            encoder_outputs = self.model.encoder(**inputs)
            bert_hidden_states = encoder_outputs.last_hidden_state
            concept_vectors = self.model.concept_encoder(bert_hidden_states)
            return concept_vectors.squeeze(0).cpu().numpy()

    @staticmethod
    def calculate_set_similarity(set_A, set_B):
        cosine_dist_matrix = cdist(set_A, set_B, 'cosine')
        row_ind, col_ind = linear_sum_assignment(cosine_dist_matrix)
        min_distance_sum = cosine_dist_matrix[row_ind, col_ind].sum()
        avg_similarity = 1 - (min_distance_sum / len(row_ind))
        return avg_similarity

    def run_experiment_1_synonym_set_test(self):
        print("\n" + "="*50)
        print("实验一：同义句的“概念集合”相似度测试")
        print("="*50)
        a1 = "这台电脑的性能非常强大。"
        a2 = "这部计算机的运行效率极高。"
        b = "这台电脑的外观非常漂亮。"
        c = "今天的天气非常不错。"
        concepts_a1, concepts_a2, concepts_b, concepts_c = map(self.get_concepts, [a1, a2, b, c])
        sim_a1_a2 = self.calculate_set_similarity(concepts_a1, concepts_a2)
        sim_a1_b = self.calculate_set_similarity(concepts_a1, concepts_b)
        sim_a1_c = self.calculate_set_similarity(concepts_a1, concepts_c)
        print(f"同义句集相似度 Sim(A1, A2) (期望值: 非常高): {sim_a1_a2:.4f}")
        print(f"主题相关句集相似度 Sim(A1, B) (期望值: 中等): {sim_a1_b:.4f}")
        print(f"无关句集相似度 Sim(A1, C) (期望值: 非常低): {sim_a1_c:.4f}")

    def run_experiment_2_attention_visualization(self):
        print("\n" + "="*50)
        print("实验二：高级概念的注意力可视化（最后一层）")
        print("="*50)
        sentence = "在暴风雨的夜晚，那位戴着礼帽的老侦探仔细检查着一把银色的匕首。"
        print(f"分析句子: “{sentence}”")
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        with torch.no_grad():
            bert_hidden_states = self.model.encoder(**inputs).last_hidden_state
            queries = self.model.concept_encoder.concept_queries.repeat(bert_hidden_states.size(0), 1, 1)
            
            # --- 修正后的代码块在这里 ---
            for layer in self.model.concept_encoder.transformer_encoder.layers:
                # 1. Self-Attention Block
                q_self_attn = layer.self_attn(queries, queries, queries)[0]
                queries = queries + layer.dropout1(q_self_attn)
                queries = layer.norm1(queries)
                # 2. Cross-Attention Block (我们从这里捕获注意力权重)
                q_cross_attn, attn_weights = layer.multihead_attn(queries, bert_hidden_states, bert_hidden_states)
                queries = queries + layer.dropout2(q_cross_attn)
                queries = layer.norm2(queries)
                # 3. Feed-Forward Block
                q_ffn = layer.linear2(layer.dropout(layer.activation(layer.linear1(queries))))
                queries = queries + layer.dropout3(q_ffn)
                queries = layer.norm3(queries)
            # --- 修正结束 ---
            
            attn_weights = attn_weights.squeeze(0).cpu().numpy()

        plt.figure(figsize=(12, 6))
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=[f"概念_{i+1}" for i in range(attn_weights.shape[0])], cmap="viridis")
        plt.title("高级概念对输入词元的最终注意力热力图"); plt.xlabel("输入句子的词元 (Tokens)"); plt.ylabel("精炼后的概念 (Refined Concepts)"); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig("advanced_attention_visualization.png")
        print("注意力热力图已保存为 'advanced_attention_visualization.png'。")
        print("请观察图表：经过多层“研讨”，概念的注意力应该更集中、分工更明确。")

    def run_experiment_3_structure_sensitivity_test(self):
        print("\n" + "="*50)
        print("实验三：句子结构敏感性测试")
        print("="*50)
        s1 = "猫追着老鼠跑。"; s2 = "老鼠追着猫跑。"
        concepts_s1 = self.get_concepts(s1)
        concepts_s2 = self.get_concepts(s2)
        similarity = self.calculate_set_similarity(concepts_s1, concepts_s2)
        print(f"“猫追老鼠” vs “老鼠追猫” 的概念集相似度 (期望值: 不会非常高): {similarity:.4f}")

if __name__ == '__main__':
    MODEL_FILE = "advanced_concept_autoencoder.pth"
    NUM_CONCEPTS = 16; CONCEPT_ENCODER_LAYERS = 6
    if not os.path.exists(MODEL_FILE):
        print(f"错误: 找不到模型文件 '{MODEL_FILE}'。请先运行 train_advanced_model.py 进行训练。")
    else:
        try:
            import matplotlib; import seaborn; from scipy.spatial.distance import cdist; from scipy.optimize import linear_sum_assignment
        except ImportError:
            print("请先安装可视化库: pip install matplotlib seaborn scipy"); exit()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validator = AdvancedConceptValidator(
            model_path=MODEL_FILE, 
            num_concepts=NUM_CONCEPTS,
            concept_encoder_layers=CONCEPT_ENCODER_LAYERS,
            device=device
        )
        validator.run_experiment_1_synonym_set_test()
        validator.run_experiment_2_attention_visualization()
        validator.run_experiment_3_structure_sensitivity_test()