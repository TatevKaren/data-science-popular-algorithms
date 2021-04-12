# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NLP: Cleaning the data and transforming the review data to a unique data where to each asin corresponds a list of words containing all reviews of that item
# ------------------------------------------------------------------------------------------------------------------------------------------------------
def clean(doc):
    listl = {"'s", "'re", "'d", "n't", "'ve", "ca", "it.i", '--', '...', 'mr.', "''", '``'}
    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc]
    doc_nosinglechar = ' '.join([word for word in doc if len(word) > 1])
    doc_w_nopunct = ' '.join(ch for ch in doc_nosinglechar.split() if not ch in exclude)
    doc_w_nostopwords = ' '.join([i for i in doc_w_nopunct.split() if not i in stop_words])
    doc_w_nonumbers = ' '.join([char for char in doc_w_nostopwords.split() if not char.isdigit()])
    doc_w_normalized = ' '.join(lemma.lemmatize(word) for word in doc_w_nonumbers.split())
    doc_w_noweirdpunc = ' '.join(ch for ch in doc_w_normalized.split() if not ch in listl)
    normalized = ' '.join(lemma.lemmatize(word) for word in doc_w_noweirdpunc.split())
    doc = normalized.split()
    return (doc)


# --------------------------------------------------------------------------------------------------------------------------------
# Function for presenting word frequencies in reviews
# --------------------------------------------------------------------------------------------------------------------------------
def freq_words(x, terms=30):
    all_w = ' '.join([text for text in x])
    all_w = all_w.split()
    fdist = FreqDist(all_w)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.show()
