from gensim.models import word2vec


path = '/root/text8.model';

model = word2vec.Word2Vec.load(path);


print model.most_similar(positive=['woman', 'king'], negative=['man']);
print model.doesnt_match("breakfast cereal dinner lunch".split());
print model.similarity('woman', 'man');
print model['computer'];