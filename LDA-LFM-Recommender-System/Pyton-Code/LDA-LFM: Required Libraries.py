import tensorflow as tf
from pandas.io.json import json_normalize
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.sparse as sparse
import nltk.data
# tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
from nltk.tokenize import word_tokenize  # for word tokenization
from nltk.corpus import stopwords  # for removing stop words
from nltk.stem import PorterStemmer  # for stemming
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from scipy.special import gammaln
import itertools
import json
import decimal
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import math
import re
import numpy as np
import time
import csv
import os
import snips as snp
import string
import spacy
from nltk import FreqDist
import gensim
from gensim import corpora
from gensim.models import LdaModel
import scipy.sparse
import numpy as np
import envoy
import progressbar
import sys
import os
