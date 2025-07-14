from datasets import load_dataset
import os

def prepare_nli_data(output_filename="sentences_for_training.txt"):
    """
    下载 shibing624/nli_zh 数据集，提取所有独立句子，并保存到文件。
    """
    if os.path.exists(output_filename):
        print(f"文件 '{output_filename}' 已存在，跳过数据准备。")
        return

    print("正在从 Hugging Face Hub 下载 shibing624/nli_zh 数据集...")
    # NLI 数据集通常包含 premise 和 hypothesis，这里是 sentence1 和 sentence2
    # 我们需要设置 trust_remote_code=True 来加载这个特定的数据集
    try:
        dataset = load_dataset("shibing624/nli_zh", "LCQMC")
        print("数据集加载完毕！")
    except Exception as e:
        print(f"加载数据集失败，请检查网络连接或依赖库。错误: {e}")
        return

    # 使用集合（set）来自动处理重复的句子
    unique_sentences = set()

    print("正在提取句子...")
    # 遍历所有可用部分（train, dev, test）
    for split in dataset.keys():
        print(f"  - 处理 {split} 部分...")
        for example in dataset[split]:
            # 提取 sentence1 和 sentence2，并去除首尾空格
            s1 = example['sentence1'].strip()
            s2 = example['sentence2'].strip()
            
            if s1:
                unique_sentences.add(s1)
            if s2:
                unique_sentences.add(s2)

    print(f"提取完成！共找到 {len(unique_sentences)} 条独立句子。")

    # 将所有独立句子写入文件，每行一句
    print(f"正在将句子写入到文件 '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        for sentence in unique_sentences:
            f.write(sentence + '\n')
            
    print("数据准备完成！")


prepare_nli_data()