from sklearn.metrics.pairwise import cosine_similarity
from pymagnitude import *
vectors  = Magnitude(r'E:\2020.09.04 基于wiki的词嵌入wikipedia2vec-master\百度百科.magnitude')

def sklearn_cosine(x, y):
    return cosine_similarity(x, y)

word1 =  '倚马千言'
word2 =  '才思敏捷'
word3 =  '倚门倚闾'
word4 =  '倚马七纸'

vector1 = vectors.query(word1).reshape(1,-1)
vector2 = vectors.query(word2).reshape(1,-1)
vector3 = vectors.query(word3).reshape(1,-1)
vector4 = vectors.query(word4).reshape(1,-1)

print("'倚马千言'和'才思敏捷'之间的余弦相似度:",sklearn_cosine(vector1 ,vector2)[0][0])
print("'倚马千言'和'倚门倚闾'之间的余弦相似度:",sklearn_cosine(vector1 ,vector3)[0][0])
print("'倚门倚闾'和'才思敏捷'之间的余弦相似度:",sklearn_cosine(vector3 ,vector2)[0][0])
print("'倚门千言'和'倚马七纸'之间的余弦相似度:",sklearn_cosine(vector1 ,vector4)[0][0])
print("'才思敏捷'和'倚马七纸'之间的余弦相似度:",sklearn_cosine(vector4 ,vector2)[0][0])
print("'倚门倚闾'和'倚马七纸'之间的余弦相似度:",sklearn_cosine(vector3 ,vector4)[0][0])