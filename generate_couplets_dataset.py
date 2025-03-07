import json
import random
from typing import List, Dict

# 扩展词汇库
NOUNS = [
    '春', '秋', '风', '雨', '山', '水', '花', '鸟', '月', '日', '天', '地', 
    '云', '雾', '江', '湖', '海', '河', '松', '竹', '梅', '兰', '草', '树',
    '星', '雪', '霜', '露', '龙', '凤', '虎', '豹', '琴', '棋', '书', '画',
    '亭', '台', '楼', '阁', '庭', '院', '门', '窗', '桥', '路', '溪', '涧'
]

VERBS = [
    '飞', '落', '升', '沉', '来', '去', '进', '退', '开', '合', '起', '落',
    '游', '走', '跑', '跳', '唱', '和', '吟', '咏', '望', '看', '思', '忆',
    '笑', '哭', '醉', '醒', '栖', '居', '寄', '托', '种', '植', '采', '摘'
]

ADJECTIVES = [
    '红', '绿', '青', '白', '高', '低', '远', '近', '深', '浅', '明', '暗',
    '古', '今', '雅', '俗', '清', '浊', '善', '恶', '美', '丑', '真', '假',
    '冷', '热', '干', '湿', '轻', '重', '快', '慢', '硬', '软', '苦', '甜'
]

def generate_word_pairs() -> List[tuple]:
    """生成对仗词对"""
    pairs = [
        ('天', '地'), ('山', '水'), ('云', '雾'), ('风', '雨'),
        ('春', '秋'), ('花', '鸟'), ('日', '月'), ('江', '海'),
        ('红', '绿'), ('高', '低'), ('远', '近'), ('深', '浅'),
        ('来', '去'), ('进', '退'), ('开', '合'), ('升', '落'),
        ('松', '竹'), ('梅', '兰'), ('琴', '棋'), ('书', '画'),
        ('亭', '台'), ('楼', '阁'), ('古', '今'), ('雅', '俗'),
        ('龙', '凤'), ('虎', '豹'), ('星', '月'), ('霜', '雪'),
        ('清', '浊'), ('善', '恶'), ('美', '丑'), ('真', '假')
    ]
    return pairs

def generate_pattern() -> str:
    """生成对联的基本模式"""
    patterns = [
        "AABB", "ABAB", "ABBA", "AABB",
        "ABCC", "AABB_CC", "ABC_ABC",
        "ABCD_ABCD", "AAB_CCB", "ABCD_DCBA"
    ]
    return random.choice(patterns)

def generate_thinking_process(first_line: str, second_line: str) -> str:
    """生成思考过程"""
    templates = [
        f"分析上联'{first_line}'：\n1. 字数结构：{len(first_line)}字\n2. 意境特点：{random.choice(['自然景物', '人文风情', '哲理寓意', '生活场景'])}\n3. 对仗要求：需要在平仄、意象上对应\n\n思考过程：\n1. 确定对应意象\n2. 调整平仄关系\n3. 优化语言表达\n\n最终选定下联：'{second_line}'",
        
        f"创作思路：\n1. 上联'{first_line}'展现了{random.choice(['空间之美', '时间流转', '自然变化', '人文气息'])}\n2. 下联需要在意境上相对，在声调上相配\n3. 通过细致推敲，选择'{second_line}'作为下联\n\n对仗分析：\n- 字字相对\n- 意境呼应\n- 韵律和谐",
        
        f"创作要点：\n1. 上联'{first_line}'的特点：\n   - 用字精炼\n   - 意境深远\n   - 韵律优美\n2. 下联构思：\n   - 保持相同节奏\n   - 选用对应意象\n   - 追求意境统一\n\n最终采用'{second_line}'，既工整对仗，又意境深远。"
    ]
    return random.choice(templates)

def generate_couplet() -> Dict:
    """生成一副完整的对联"""
    pairs = generate_word_pairs()
    pattern = generate_pattern()
    
    # 生成上联
    first_line = ""
    second_line = ""
    used_pairs = random.sample(pairs, 2)
    
    # 根据不同模式生成对联
    if pattern in ["AABB", "ABAB"]:
        first_line = used_pairs[0][0] + used_pairs[1][0] + random.choice(VERBS) + random.choice(NOUNS)
        second_line = used_pairs[0][1] + used_pairs[1][1] + random.choice(VERBS) + random.choice(NOUNS)
    else:
        first_line = used_pairs[0][0] + random.choice(VERBS) + used_pairs[1][0] + random.choice(NOUNS)
        second_line = used_pairs[0][1] + random.choice(VERBS) + used_pairs[1][1] + random.choice(NOUNS)
    
    thinking = generate_thinking_process(first_line, second_line)
    
    return {
        "question": first_line,
        "answer": second_line,
        "thinking_process": thinking,
        "pattern": pattern
    }

def generate_dataset(num_samples: int = 6000) -> List[Dict]:
    """生成完整的数据集"""
    dataset = []
    for _ in range(num_samples):
        couplet = generate_couplet()
        dataset.append(couplet)
    return dataset

if __name__ == "__main__":
    # 生成数据集
    dataset = generate_dataset()
    
    # 保存到JSON文件
    with open('couplets_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 打印示例
    print(f"已生成{len(dataset)}条对联数据集，并保存到couplets_dataset.json文件中。")
    print("\n示例数据：")
    for i in range(3):
        print(f"\n示例 {i+1}:")
        print(json.dumps(dataset[i], ensure_ascii=False, indent=2)) 