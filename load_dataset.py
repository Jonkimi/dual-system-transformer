
from torch.utils.data import Dataset, DataLoader
# 0. 准备一个简单的数据集
# 在实际应用中，这里应该是一个文件路径
# sentences = [
#     "一只敏捷的棕色狐狸跳过了懒狗。",
#     "机器学习是人工智能的一个分支。",
#     "今天天气真好，我们去公园散步吧。",
#     "北京是中国的首都，拥有悠久的历史。",
#     "我喜欢阅读科幻小说和看电影。",
#     "这个模型的想法非常具有创新性。",
# ]

# --- 主要修改部分在这里 ---
def load_sentences_from_file(filepath="./datasets/train/sentences_for_training.txt"):
    """从文件加载句子列表"""
    print(f"正在从 '{filepath}' 加载数据集...")
    with open(filepath, 'r', encoding='utf-8') as f:
        # 读取所有行，去除首尾空格，并过滤掉空行，限制读取一万行
        sentences = [line.strip() for line in f if line.strip()]

    return sentences[:2000]

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    
# 2. 数据加载器
sentences = load_sentences_from_file()

sentences +=[
    "一只敏捷的棕色狐狸跳过了懒狗。",
    "机器学习是人工智能的一个分支。",
    "今天天气真好，我们去公园散步吧。",
    "北京是中国的首都，拥有悠久的历史。",
    "我喜欢阅读科幻小说和看电影。",
    "这个模型的想法非常具有创新性。",
]

print(f"加载了 {len(sentences)} 条句子用于训练。")
dataset = SentenceDataset(sentences)