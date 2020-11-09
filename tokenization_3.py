import jieba

text ='''通过对全网主流媒体及社交媒体平台进行实时数据抓取和深度处理，可以帮助政府/企业及时、全面、精准地从海量的数据中了解公众态度、掌控舆论动向、聆听用户声音、洞察行业变化。'''

text_segment = jieba.lcut(text)

print(text_segment )