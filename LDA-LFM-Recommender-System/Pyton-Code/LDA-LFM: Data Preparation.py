# -------------------------------------------------------------------------------------------------------
# Importing required tools for performing NLP
# -------------------------------------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#downloading necessary nltk datasets
#vacabulary from the "WordNet"
nltk.download('wordnet')
# all english stopwords
nltk.download('stopwords')
# setting the stopwords
stop_words = set(stopwords.words("english"))
# for excluding the punctuation in a given string
exclude = set(string.punctuation)
# for lemmatization
lemma = WordNetLemmatizer()


# -------------------------------------------------------------------------------------------------------
# Importing and cleaning data
# -------------------------------------------------------------------------------------------------------
data = pd.read_json('/data_path/reviews_productcategory.json',orient='columns',typ = 'table',lines = True,convert_axes=True,encoding='utf-8')
normaldata = json_normalize(data)  # transforming json type dat to usual dataframe without braces and quotations
newData = pd.DataFrame(normaldata.drop(['helpful', 'reviewerName', 'summary', 'reviewTime','unixReviewTime'], axis=1))


# -------------------------------------------------------------------------------------------------------
# Descriptive statistics
# -------------------------------------------------------------------------------------------------------
# Determining number of unique users and items in the current dataset
print(len(newData.groupby('reviewerID').nunique()))
print(len(newData.groupby('asin').nunique()))
# Average words in the reviews(descriptive stat.)
newData['numWords'] = newData['reviewText'].str.count(' ') + 1
avgwords = sum(newData['numWords']) / len(newData['numWords'])  # average words in reviews
print(avgwords)
