import spacy

nlp = spacy.load('en_core_web_lg')
stop_words = nlp.Defaults.stop_words

# Remove Stopwords from train and test
f = 'Text_column_name'
train_texts = [' '.join([t for t in text.split() if(t.lower() not in stop_words)]) for text in X_train[f]]
test_texts = [' '.join([t for t in text.split() if(t.lower() not in stop_words)]) for text in X_test[f]]

# Get dataframes with text converted to spaCy vectors
tr_df = pd.DataFrame([list(nlp(text).vector) for text in train_texts])
te_df = pd.DataFrame([list(nlp(text).vector) for text in test_texts])

