from pathlib import Path
import pandas as pd

import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Importing Gensim
import gensim
from gensim import corpora

# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

proc_data_dir = Path('Prepared data')
proc_data_dir = Path('Raw data')
data = pd.read_excel(proc_data_dir / 'training_data_AlixPartners_Val_Dist.xlsx')

corpus = list(data['Paragraph'])
corpus = list(set(corpus))
############## Pre-proc
# stop loss words 
stop = set(stopwords.words('english'))

# punctuation 
exclude = set(string.punctuation) 

# lemmatization
lemma = WordNetLemmatizer() 

# One function for all the steps:
def clean(doc):
    
    # convert text into lower case + split into words
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    
    # remove any stop words present
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    
    # remove punctuations + normalize the text
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())  
    return normalized

# clean data stored in a new list
clean_corpus = [clean(doc).split() for doc in corpus]

dict_ = corpora.Dictionary(clean_corpus)
doc_term_matrix = [dict_.doc2bow(i) for i in clean_corpus]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dict_, passes=1, random_state=0, eval_every=None)
ldamodel.print_topics()
print(ldamodel.print_topics(num_topics=10, num_words=5))

count = 0
for i in ldamodel[doc_term_matrix]:
    print("doc : ",count,i)
    count += 1
