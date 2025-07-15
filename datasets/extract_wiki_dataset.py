import random
import sys
import os

def extract_random_lines(input_path, output_path, num_to_extract=500000, min_len=20):
    """
    从大文本文件中随机抽取指定数量的行，并过滤掉长度过短的行。
    使用蓄水池抽样算法，这允许我们只遍历文件一次。

    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param num_to_extract: 要抽取的行数
    :param min_len: 最小行长度
    """
    print(f"开始从 '{input_path}' 采样 {num_to_extract} 行...")
    print(f"过滤掉长度小于 {min_len} 的行。")

    reservoir = []
    lines_seen = 0
    total_lines_processed = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines_processed += 1
                if total_lines_processed % 500000 == 0:
                    print(f"已处理 {total_lines_processed} 行...")

                # 过滤掉过短的行
                if len(line.strip()) < min_len:
                    continue

                lines_seen += 1
                if len(reservoir) < num_to_extract:
                    reservoir.append(line)
                else:
                    # 蓄水池已满，以一定概率替换
                    j = random.randint(0, lines_seen - 1)
                    if j < num_to_extract:
                        reservoir[j] = line
    except FileNotFoundError:
        print(f"错误：输入文件未找到 '{input_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

    print(f"总共处理了 {total_lines_processed} 行。")
    if len(reservoir) < num_to_extract:
        print(f"警告：只找到 {len(reservoir)} 行符合条件，少于所要求的 {num_to_extract} 行。")
    else:
        print(f"成功采样 {len(reservoir)} 行。")

    # 将结果写入输出文件
    print(f"正在将采样行写入 '{output_path}'...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        # 打乱顺序，因为蓄水池抽样最后的结果中，前面的元素被替换的概率较小
        random.shuffle(reservoir)
        for line in reservoir:
            f.write(line)

    print("完成。")

if __name__ == '__main__':
    # 假设脚本从项目根目录运行
    input_file = 'datasets/wiki_sentences_for_training_v2.txt'
    output_file = 'datasets/wiki_sentences_for_training_500k.txt'
    
    # 执行抽取函数
    extract_random_lines(input_file, output_file, num_to_extract=500000, min_len=20)