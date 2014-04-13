from gensim import corpora, models, similarities;
import logging;
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO);

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
 	       [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
           [(0, 1.0), (4, 2.0), (7, 1.0)],
           [(3, 1.0), (5, 1.0), (6, 1.0)],
           [(9, 1.0)],
           [(9, 1.0), (10, 1.0)],
           [(9, 1.0), (10, 1.0), (11, 1.0)],
           [(8, 1.0), (10, 1.0), (11, 1.0)]];

tfidf = models.TfidfModel(corpus);

vec = [(0, 1), (4, 1)];
print(tfidf[vec]);


index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12);
sims = index[tfidf[vec]];
print(list(enumerate(sims)));



# From Strings to Vectors
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"];

stoplist = set('for a of the and to in'.split());


texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents];


all_tokens = sum(texts, []);
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1);
texts = [[word for word in text if word not in tokens_once] for text in texts];

print(texts);


dictionary = corpora.Dictionary(texts);
dictionary.save('/tmp/deerwester.dict'); # store the dictionary, for future reference
print(dictionary);


new_doc = "Human computer interaction";
new_vec = dictionary.doc2bow(new_doc.lower().split());
print(new_vec); # the word "interaction" does not appear in the dictionary and is ignored


corpora.MmCorpus.serialize('/root/deerwester.mm', corpus) # store to disk, for later use
print(corpus);