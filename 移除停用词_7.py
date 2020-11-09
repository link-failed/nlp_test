import jieba

text = "达观数据客户意见洞察平台对公司品牌进行负面舆情实时监测，事件演变趋势预测，预警实时触达，帮助公司市场及品牌部门第一时间发现负面舆情，及时应对危机公关，控制舆论走向，防止品牌受损。"

my_stopwords =  [i.strip() for i in open('stop_words_zh.txt',encoding='utf-8').readlines()]
new_tokens=[]

# Tokenization using word_tokenize()
all_tokens=jieba.lcut(text)

for token in all_tokens:
  if token not in my_stopwords:
    new_tokens.append(token)

" ".join(new_tokens)

print(new_tokens)