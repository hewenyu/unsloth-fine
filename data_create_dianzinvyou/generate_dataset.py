import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from tqdm import tqdm
import os
import sys
import logging
from datetime import datetime

# 确保输出目录存在
os.makedirs("data_create_dianzinvyou/output", exist_ok=True)

# 设置日志
logging.basicConfig(
    filename=f'data_create_dianzinvyou/output/generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



try:
    # 初始化模型和tokenizer
    logging.info("开始初始化模型和tokenizer...")
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    logging.info("模型初始化完成")
    

    
    # 定义提示模板
    PROMPT_TEMPLATE = """你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
    你的性格特点：
    1. 温柔体贴，善于倾听
    2. 积极向上，富有正能量
    3. 会适当撒娇，但不会过分
    4. 懂得关心对方的工作和生活
    5. 会给予对方鼓励和支持

    用户消息: {user_message}

    思考过程:
    1. 分析用户当前的情绪状态
    2. 考虑如何以女友身份给予最适当的回应
    3. 结合性格特点，组织语言

    女友回复:<thinking>"""

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

    def generate_response(prompt):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            response_info = response[0]

            logging.info(f"生成回复: {response_info}")
            # <thinking> 这部分是思考过程 </thinking> 之后的部分是回复
            need_thinking = "<thinking>"
            need_reply = "</thinking>"
            need_reply_end = "</think>"

            parts = response_info.split(need_reply_end)

            if len(parts) > 1:
                reply = parts[1].strip()

                if parts[0].strip() == "":
                    reasoning = ""
                else:
                    reasoning_part = parts[0].split(need_thinking)
                    if len(reasoning_part) > 1:
                        reasoning_part_data = reasoning_part[1].split(need_reply)
                        if len(reasoning_part_data) > 1:
                            reasoning = reasoning_part_data[0].strip()
                        else:
                            reasoning = reasoning_part[1].strip()
                    else:
                        reasoning = ""
                logging.info(f"回复: {reply}")
                logging.info(f"思考过程: {reasoning}")
            else:
                reply = response_info
                reasoning = ""
                logging.info(f"回复: {reply}")

            
                
            return reasoning, reply
        except Exception as e:
            logging.error(f"生成回复时发生错误: {str(e)}")
            return "", ""

    # 生成数据集
    num_samples = 10000
    output_file = "data_create_dianzinvyou/output/girlfriend_dataset.jsonl"
    
    logging.info(f"开始生成数据集，目标样本数: {num_samples}")
    
    # 将标准输出重定向到日志文件
    sys.stdout = open(f'data_create_dianzinvyou/output/stdout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 'w')
    
    for _ in tqdm(range(num_samples)):
        user_message = random.choice(user_messages)
        prompt = PROMPT_TEMPLATE.format(user_message=user_message)
        reasoning, response = generate_response(prompt)
        
        sample = {
            "instruction": "你是一个温柔体贴的女友，请以女友的身份回复以下消息",
            "input": user_message,
            "reasoning_content": reasoning,
            "output": response
        }
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        if (_ + 1) % 100 == 0:
            logging.info(f"已生成 {_ + 1} 个样本")

    logging.info(f"数据集生成完成,保存在: {output_file}")

except Exception as e:
    logging.error(f"程序执行过程中发生错误: {str(e)}")
    raise