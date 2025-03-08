from openai import OpenAI
import os
import random
import requests
import json


SiliconflowToken = home = os.environ.get("SiliconflowToken")

def get_user_message(message:str):

    url = "https://api.siliconflow.cn/v1/chat/completions"

    base_prompt = """你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
    你的性格特点：
    1. 温柔体贴，善于倾听
    2. 积极向上，富有正能量
    3. 会适当撒娇，但不会过分
    4. 懂得关心对方的工作和生活
    5. 会给予对方鼓励和支持
    
    需要分析用户当前的情绪状态，并结合性格特点，组织语言"""

    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {
                "role": "system",
                "content": base_prompt
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": False,
        "max_tokens": 1200,
        "stop": None,
        "n": 1, # 生成数量
        "response_format": {"type": "text"}, # 响应格式
    }
    headers = {
        "Authorization": f"Bearer {SiliconflowToken}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text

def split_text(response:str):
    try:
        response_json = json.loads(response)
        reasoning_content = response_json['choices'][0]['message']['reasoning_content']
        content = response_json['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return None,None
    
    return reasoning_content,content


def main():
    # 准备一些常见的用户消息场景
    user_messages = [
        "今天工作好累啊，想找你聊聊天",
        "最近在学编程，感觉好难，需要鼓励",
        "想你了，不知道你在干什么",
        "考试成绩不太理想，有点沮丧",
        "周末有空吗？想约你出去玩",
        "工作上遇到难题，想听听你的建议",
        "最近状态不太好，需要你的安慰",
        "有个好消息想第一个告诉你",
        "天气这么好，好想和你一起散步",
        "项目遇到瓶颈，需要一些鼓励",
        "今天特别想见你一面",
        "刚见完你，心情超级好",
        "加班到现在，好累",
        "终于完成了一个大项目",
        "和同事有点不愉快",
        "在考虑要不要换工作",
        "这段时间睡得不太好",
        "想学个新技能，你觉得怎么样",
        "今天吃了家不错的餐厅，下次带你去",
        "刚健完身，感觉充满活力",
        "今天差点迟到了",
        "被领导表扬了，想和你分享",
        "这周末想在家打游戏放松一下",
        "最近体重上升了，有点担心",
        "路上看到只可爱的猫，拍给你看",
        "和朋友聚完会了",
        "明天要面试，有点紧张",
        "买了新衣服，想听听你的意见",
        "好想尝尝你的手艺",
        "刚下班，好想见你",
        "下雨天心情不太好",
        "看了部很感人的电影",
        "想给父母买礼物，需要你帮我参考",
        "看到别人秀恩爱，想你了",
        "在考虑要不要考研",
        "刚和家里通完电话",
        "做了个有你的梦",
        "好想和你一起去旅行",
        "存了点钱，想买点什么好",
        "今天遇到一件暖心的事",
        "学会了一道新菜，想给你尝尝",
        "今天工作特别顺利",
        "忘带伞被淋湿了",
        "想换个发型，你觉得适合我吗",
        "今天起晚了",
        "和朋友闹矛盾了",
        "最近在追的剧好看",
        "晚霞特别美，想和你一起看",
        "收到一份惊喜礼物",
        "做了个重要决定想和你商量",
        "一个人感觉有点孤单",
        "刚做完演讲，紧张死了",
        "认识了个有趣的人，和你说说",
        "在学做饭，想为你做一顿",
        "好想听你唱歌",
        "参加了个有意思的活动",
        "买了本好书，推荐给你",
        "想养只宠物，你觉得呢",
        "今天上班好无聊",
        "见了老同学，聊到了你",
        "在减肥，好难坚持下去",
        "今天帮助了别人，很开心",
        "怀念小时候吃的零食",
        "在学新语言，进步很慢",
        "看别人秀恩爱有点羡慕",
        "刚做完工作汇报",
        "在想未来的规划",
        "收到重要邮件了",
        "开完重要会议",
        "在考虑买房的事",
        "遇到以前的朋友了",
        "掌握了新技能",
        "为目标在努力",
        "今天工作很有挑战性",
        "和家人视频了",
        "在看一本不错的书",
        "鼓起勇气做了决定",
        "考试通过了",
        "最近在思考人生",
        "遇到点小困难",
        "参加了个培训",
        "在学理财",
        "今天心情很复杂",
        "和朋友谈心",
        "在准备重要项目",
        "坚持运动第一天",
        "有个好消息要告诉你",
        "开始学摄影了",
        "工作不太顺",
        "和家人商量了事情",
        "想去国外发展",
        "遇到个难题想和你说",
        "达成了小目标",
        "开始学画画了",
        "特别特别想你",
        "参加了个派对",
        "在想职业方向",
        "生活有了新变化",
        "完成了团队项目",
        "开始学吉他了",
        "今天心情特别好",
        "完成一次演讲",
        "在规划职业发展",
        "遇到点困难，想找你聊聊"
    ]

    user_message = random.choice(user_messages)
    print(user_message)
    response = get_user_message(user_message)    
    reasoning_content,content = split_text(response)
    print(reasoning_content)
    print(content)

    # 生成数据集
    num_samples = 10000
    output_file = "data_create_dianzinvyou/output/girlfriend.jsonl"

    for _ in range(num_samples):
        user_message = random.choice(user_messages)
        response = get_user_message(user_message)    
        reasoning_content,content = split_text(response)

        if reasoning_content is None or content is None:
            continue
        
        sample = {
            "instruction": "你是一个温柔体贴的女友，请以女友的身份回复以下消息",
            "input": user_message,
            "reasoning_content": reasoning_content,
            "output": content
        }

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()