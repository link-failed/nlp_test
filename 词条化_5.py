# 下载和导入transfromers
from transformers import AutoTokenizer

text = '''对公司品牌进行负面舆情实时监测，事件演变趋势预测，预警实时触达，帮助公司市场及品牌部门第一时间发现负面舆情，及时应对危机公关，控制舆论走向，防止品牌受损。'''

# 初始化tokenizer
tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')

# 使用tokenizer对文本进行编码
inputs=tokenizer.encode(text)
print(inputs)
# 使用tokenizer对文本编码进行解码
outputs = tokenizer.decode(inputs)
print(outputs)