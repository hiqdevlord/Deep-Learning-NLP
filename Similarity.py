from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


dictionary = corpora.Dictionary.load('/root/deerwester.dict');
corpus = corpora.MmCorpus('/root/deerwester.mm'); # comes from the first tutorial, "From strings to vectors"
print(corpus);

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2);

doc = "Human computer interaction";
vec_bow = dictionary.doc2bow(doc.lower().split());
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)