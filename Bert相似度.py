import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine


def get_word_indeces(tokenizer, text, word):
    '''
    确定 "text "中与 "word "相对应的标记的index或indeces。`word`可以由多个字词复合而成，如 "数据分析（数据+分析）"。

    确定indeces是很棘手的，因为一个词汇可能会被分解成多个token。
    我用一种比较迂回的方法解决了这个问题--我用一定数量的`[MASK]`的token代替`word`，然后在词条化（tokenization）结果中找到这些token。
    '''
    # 将'word'词条化--它可以被分解成多个词条(token)或子词（subword）
    word_tokens = tokenizer.tokenize(word)

    # 创建一个"[MASK]"词条序列来代替 "word"
    masks_str = ' '.join(['[MASK]'] * len(word_tokens))

    # 将"word"替换为 mask词条
    text_masked = text.replace(word, masks_str)

    # `encode`环节同时执行如下功能:
    #   1. 将文本词条化
    #   2. 将词条映射到其对应的id
    #   3. 增加特殊的token，主要是 [CLS] 和 [SEP]
    input_ids = tokenizer.encode(text_masked)

    # 使用numpy的`where`函数来查找[MASK]词条的所有indeces
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces


def get_embedding(b_model, b_tokenizer, text, word=''):
    '''
    使用指定的model和tokenizer对喂进来的文本和词进行句嵌入或者词汇语境嵌入输出。
    '''

    # 如果提供了一个词，找出与之对应的token
    if not word == '':
        word_indeces = get_word_indeces(b_tokenizer, text, word)

    # 对文本进行编码，添加(必要的!)特殊token，并转换为PyTorch tensors
    encoded_dict = b_tokenizer.encode_plus(
        text,  # 待encode的文本
        add_special_tokens=True,  # 增加特殊token ，加在句首和句尾添加'[CLS]' 和 '[SEP]'
        return_tensors='pt',  # 返回的数据格式为pytorch tensors
    )

    input_ids = encoded_dict['input_ids']

    b_model.eval()

    # 通过模型运行经编码后的文本以获得hidden states
    bert_outputs = b_model(input_ids)

    # 通过BERT运行经编码后的文本，集合所有12层产生的所有隐藏状态
    with torch.no_grad():

        outputs = b_model(input_ids)

        # 根据之前`from_pretrained`调用中的配置方式，评估模型将返回不同数量的对象。
        # 在这种情况下，因为我们设置了`output_hidden_states = True`，
        # 第三项将是所有层的隐藏状态。更多细节请参见文档。
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

        hidden_states = outputs[2]

    # `hidden_states`的shape是 [13 x 1 x <文本长度> x 768]

    # 选择第二层到最后一层的嵌入，`token_vecs` 是一个形如[<文本长度> x 768]的tensor
    token_vecs = hidden_states[-2][0]

    # 计算所有token向量的平均值
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # 将上述token平均嵌入向量转化为numpy array
    sentence_embedding = sentence_embedding.detach().numpy()

    # 如果提供了`word`，计算其token的嵌入。
    if not word == '':
        # 假如是词长大于等于2的词汇，取`word`T中token嵌入的平均值
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)

        # 转化为numpy array
        word_embedding = word_embedding.detach().numpy()

        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding


bert_model = BertModel.from_pretrained(r"E:\2020.09.07 pytorch_pretrained_models\bert-sim-chinese",
                                       output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained(r"E:\2020.09.07 pytorch_pretrained_models\bert-sim-chinese")

text_query = "如何针对用户群体进行数据分析"

text_A = "基于25W+知乎数据，我挖掘出这些人群特征和内容偏好"
text_B = "揭开微博转发传播的规律：以“人民日报”发布的G20文艺晚会微博为例"
text_C = '''不懂数理和编程，如何运用免费的大数据工具获得行业洞察？'''

# 使用BERT获取各语句的向量表示
emb_query = get_embedding(bert_model, bert_tokenizer, text_query)
emb_A = get_embedding(bert_model, bert_tokenizer, text_A)
emb_B = get_embedding(bert_model, bert_tokenizer, text_B)
emb_C = get_embedding(bert_model, bert_tokenizer, text_C)

# 计算query和各语句的相似余弦值（cosine similarity）
sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)
sim_query_C = 1 - cosine(emb_query, emb_C)

print('')
print('BERT Similarity:')
print('  sim(query, A): {:.4}'.format(sim_query_A))
print('  sim(query, B): {:.4}'.format(sim_query_B))
print('  sim(query, C): {:.4}'.format(sim_query_C))