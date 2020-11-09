from pyltp import SentenceSplitter

docs = '''社会化聆听可以7*24小时全天侯对全网进行实时监测，采用文本挖掘技术对民众意见反馈、领导发言、政策文件进行监测分析。通过这些先进的nlp技术的处理，可以帮助政府及时的了解社会各阶层民众对社会现状和发展所持有的情绪、态度、看法、意见和行为倾向 。最终实现积极主动的响应处理措施和方案，对于互联网上一些错误的、失实的舆论做出正确的引导作用，控制舆论发展方向。'''

sentences = SentenceSplitter.split(docs)

print('\n'.join(list(sentences)))