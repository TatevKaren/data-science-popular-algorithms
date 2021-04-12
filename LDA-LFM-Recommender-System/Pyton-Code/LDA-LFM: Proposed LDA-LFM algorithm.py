# -----------------------------------------------------------------------
# LDA-LFM model (Proposed Approach)
# -----------------------------------------------------------------------
def get_LDA_data(data):
    documents = get_documents(data)
    dictionary = corpora.Dictionary(documents['documents'])
    dictionary.filter_extremes(keep_n=5000)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in documents['documents']]
    return(doc_term_matrix,documents,dictionary)


def get_doc(review):
    revi = sum(review,[])
    return(revi)


def get_documents(data):
    data['reviews'] = data['reviewText'].apply(clean)
    #newData['reviews']= pd.Series(newData['reviews']).apply(lambda x: x.split())
    documents =  data[['asin','reviews']]
    documents = documents.groupby('asin', as_index=False).agg(lambda x: x.tolist())
    documents['documents'] = documents['reviews'].apply(get_doc)
    documents = documents[['asin','documents']]
    return(documents)


def get_vocab_clear(data):
    documents_text = get_documents(data)
    vocab = corpora.Dictionary(documents_text['documents'])
    vocab.filter_extremes(keep_n=5000)
    return(vocab,documents_text)


def get_Corpus(clean_data,vocab):
    def get_docs(doc):
        bow = vocab.doc2bow(doc)
        return (bow)
    clean_data['docs'] = clean_data['documents'].apply(get_docs)
    documents = clean_data['docs']
    itemWords = clean_data['documents'].apply(len)
    return(documents,itemWords)


def initializeLDA(vocab,documents,num_topics):
    n_words = len(vocab)
    n_docs = len(documents)
    topics = {}  # key-value pairs of form (m,i):z

    for m in range(n_docs):
        for i in documents[m]:
            w = i[0]
            # choose an arbitrary topic as first topic for word i
            z = np.random.randint(num_topics)
            topics[(m, i)] = z
    return(n_words, n_docs, topics)


# -------------------------------------------------------------------
# Getting the clean data and the corpus in a desired form
# -------------------------------------------------------------------
clear = get_vocab_clear(train_data_reviews)
vocab = clear[0]
clean_data = clear[1]
corpus = get_Corpus(clean_data, vocab)
documents = corpus[0]
itemWords = corpus[1]

#here use the num_topics = 5/10
Init_LDA = initializeLDA(vocab,documents,num_topics)
n_words = Init_LDA[0]
n_docs = Init_LDA[1]
topics = Init_LDA[2]
mu = calculate_global_mean_item_rating()
num_users = newData.reviewerID.unique().shape[0]
num_items = newData.asin.unique().shape[0]


# --------------------------------------------------------------------
# Function performing Gread Search as a result one can conclude
# from the printed results which are best_gamma and best_lambda values
# --------------------------------------------------------------------
def Grid_Search(n_iter):

    gamma_s = [1, 10, 100, 1000, 10000]
    lambda_regs = [0, 0.001, 0.01, 0.1, 1, 10]

    grid_search = []

    for gamma in gamma_s:

     for reg in lambda_regs:

        t = Tensorflow_LDA_LFM(mu, num_users, num_items, rank, reg, kappa, gamma, documents, vocab, itemWords, num_topics)
        # fitting on the training data
        users_items_ratings = sparse_to_UserItemRating(training_matrix_coo)
        # testing on the validation data
        valid_users_items_ratings = sparse_to_UserItemRating(valid_matrix_coo)
        obj = t.fit(users_items_ratings, valid_users_items_ratings, n_iter)
        valid_MSE = obj[n_iter-1]
        print('regul_corpus: ' + str(gamma))
        print('lambda: ' + str(reg))
        print('MSE: ' + str(valid_MSE))
        grid_search.append(["gamma", gamma, "reg", reg, "MSE", valid_MSE])


# --------------------------------------------------------------------
# Topic-Identification function which requires the Phi matrix
# corresponding to the last iteration of the  LDA-LFM model
# The function will then print top 10 words per topic
# --------------------------------------------------------------------
Phi = 0
def topterms(n_terms=10, phi = Phi):
    vec = np.at_2d(np.arange(0, n_words))
    Topics = []
    for k in range(num_topics):
        probs = np.atleast_2d(phi[k, :])
        mat = np.append(probs, vec, 0)
        sind = np.array([mat[:, i] for i in np.argsort(mat[0])]).T
        Topics.append([vocab[int(sind[1, n_words - 1 - i])] for i in range(n_terms)])
    return(Topics)

# -----------------------------------------------------------------------
# LDA initialization (with LDA Gibbs Sampling )
# -----------------------------------------------------------------------
def initializeLDA(vocab,documents):
    n_words = len(vocab)
    n_docs = len(documents)
    # The (i,j) entry of self.nmz is the number of words in document i ←􏰀assigned to topic j.
    nmz = np.zeros((n_docs, num_topics))
    # The (i,j) entry of self.nzw is the number of times term j is assigned←􏰀 to topic i.
    nzw = np.zeros((num_topics, n_words))
    # The (i)-th entry is the number of times topic i is assigned in the corpus.
    nz = np.zeros(num_topics)
    # Initialize the topic assignment dictionary.
    topics = {}  # key-value pairs of form (m,i):z

    for m in range(n_docs):
        for j in documents[m]:
            w = j[0]
            z = np.random.randint(num_topics)
            nmz[m,z] +=1
            nz[z] += 1
            nzw[z, w] += 1
            topics[(m, j)] = z
    return(n_words, n_docs, nzw, nz, nmz, topics)


# -----------------------------------------------------------------------
# The class corresponding to the LDA-LFM model
# -----------------------------------------------------------------------
kappa = 0
n_iter = 0
num_topics = 0
class Tensorflow_LDA_LFM_version:

        def __init__(self, mu, num_users, num_items, rank, lambda_reg, kappa, gamma, alpha, beta, documents, vocab,itemWords, num_topics):
            self.rank = rank
            self.num_users = num_users
            self.num_items = num_items
            self.lambda_reg = lambda_reg
            self.mu = mu

            # for the LDA model
            self.num_topics = num_topics
            self.gamma = gamma
            self.alpha = alpha
            self.beta = beta
            self.documents = documents
            self.vocab = vocab
            self.n_docs = num_items
            self.n_words = n_words
            self.nzw = nzw
            self.nz = nz
            self.nmz = nmz
            self.topics = topics
            self.lk = 0.0
            self.itemWords = itemWords

            # for variable initialization
            self.global_variables_initializer()

        # Getting Phi matrix from count matrix nzw
        def get_Phi(self, nzw):
            psi = self.nzw + self.beta
            Psi = psi / np.sum(psi, axis=1)[:, np.newaxis]
            exp_tensorPsi = tf.exp(Psi)
            totalt = tf.reduce_sum(exp_tensorPsi, 1)
            Phi = exp_tensorPsi / tf.reshape(totalt, (-1, 1))
            return(Phi)

        # Getting Theta matrix from count matrix nzw
        def get_Theta_nmz(self, Q):
            nmz = self.nmz
            tensorQ = tf.multiply(Q, self.kappa)
            exp_tensorQ = tf.exp(tensorQ)
            totalt = tf.reduce_sum(exp_tensorQ, 1)
            Theta_large = exp_tensorQ / tf.reshape(totalt, (-1, 1))
            # from rank --> number of topics matrix
            Theta = Theta_large.eval()
            if self.num_topics < self.rank
               Theta = Theta[:, 0:self.num_topics]
               # transforming to a count matrix
               nmz = tf.round(tf.multiply(Theta, self.itemWords[:, tf.newaxis]) * self.num_topics)
            print(Theta)
            return (Theta, nmz.eval())

        def global_variables_initializer(self):
            self.b_u = tf.Variable(tf.random_normal([self.num_users, 1], stddev=0.01, mean=0), name="user_bias",trainable=True)
            self.b_i = tf.Variable(tf.random_normal([self.num_items, 1], stddev=0.01, mean=0), name="item_bias",trainable=True)
            self.P = tf.Variable(tf.random_normal([self.num_users, self.rank], stddev=0.01, mean=0), name="users",trainable=True)
            self.Q = tf.Variable(tf.random_normal([self.num_items, self.rank], stddev=0.01, mean=0), name="items",trainable=True)
            self.kappa = tf.Variable(kappa, name="kappa", dtype='float32',trainable=True)
            self.Phi = tf.Variable(self.get_Phi(self.nzw), name="topicWords")

        # Predicting all ratings
        def predict(self, users, items):
            P_ = tf.squeeze(tf.nn.embedding_lookup(self.P, users))
            Q_ = tf.squeeze(tf.nn.embedding_lookup(self.Q, items))
            predictions = tf.nn.sigmoid((tf.reduce_sum(tf.multiply(P_, Q_), reduction_indices=[1])))
            ubias = tf.squeeze(tf.nn.embedding_lookup(self.b_u, users))
            ibias = tf.squeeze(tf.nn.embedding_lookup(self.b_i, items))
            prediction = self.mu + ubias + ibias + tf.squeeze(predictions)
            return (prediction)

        # calculating the conditional probability for random topic assignments
        def conditional_prob(self, m, w):
            p_z = (self.nmz[m, :] + self.alpha) * (self.nzw[:, w] + self.beta) / (self.nz + self.beta * self.n_words)
            return (p_z / np.sum(p_z))

        # sampling new topics and updating count matrices
        def sample(self):
            for m in range(self.n_docs):
                for i in self.documents[m]:
                    w = i[0]
                    z = self.topics[(m, i)]
                    self.nmz[m, z] -= 1
                    self.nz[z] -= 1
                    self.nzw[z, w] -= 1
                    p_z = self.conditional_prob(m, w)
                    print(p_z)
                    z = np.random.multinomial(1, p_z).argmax()
                    print(z)
                    self.nmz[m, z] += 1
                    self.nz[z] += 1
                    self.nzw[z, w] += 1
                    self.topics[(m, i)] = z
            return (self.nz, self.nzw, self.nmz, self.topics)

        # calculating the corpus likelihood
        def corpus_likelihood(self, nmz_new):
            lk = 0
            for z in range(self.num_topics):
                lk += np.sum(gammaln(self.nzw[z, :] + self.beta)) - gammaln(np.sum(self.nzw[z, :] + self.beta))
                lk -= self.n_words * gammaln(self.beta) - gammaln(self.n_words * self.beta)
            for m in range(self.n_docs):
                lk += np.sum(gammaln(nmz_new[m, :] + self.alpha)) - gammaln(np.sum(nmz_new[m, :] + self.alpha))
                lk -= self.num_topics * gammaln(self.alpha) - gammaln(self.num_topics * self.alpha)
            return(lk)

        def regLoss(self):
            reg_lo = 0
            reg_lo += tf.reduce_sum(tf.square(self.Q))
            reg_lo += tf.reduce_sum(tf.square(self.P))
            reg_lo += tf.reduce_sum(tf.square(self.b_u))
            reg_lo += tf.reduce_sum(tf.square(self.b_i))
            return (reg_lo * self.lambda_reg)

        def loss(self, usersitemsratings):
            users, items, ratings = usersitemsratings
            prediction = self.predict(users, items)
            err_loss = tf.nn.l2_loss(prediction - ratings)
            reg_loss = self.regLoss()
            self.total_loss = err_loss + reg_loss
            tf.summary.scalar("loss", self.total_loss)
            return (self.total_loss - self.gamma * self.lk)

        def fit(self, usersitemsratings, test_usersitemsratings=None, n_iter=n_iter):
            tferrors = []
            cost = self.loss(usersitemsratings)
            # all varaibles that will be learnt by Adam Optimizer
            adam_train = tf.train.AdamOptimizer(0.01)
            gradients_and_vars = adam_train.compute_gradients(cost)
            optimizer = adam_train.apply_gradients(gradients_and_vars)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                users, items, ratings = usersitemsratings

                for i in range(n_iter):
                    print(i)
                    sess.run(optimizer)
                    Theta_old = self.nmz
                    # new topic assignment
                    gibbs = self.sample()
                    self.nz = gibbs[0]
                    self.nzw = gibbs[1]
                    self.topics = gibbs[3]
                    self.Phi = self.get_Phi(self.nzw)
                    topic_prob = self.get_Theta_nmz(self.Q)
                    nmz_new = topic_prob[1]
                    Theta= topic_prob[0]
                    self.lk = self.corpus_likelihood(nmz_new)
                    tferrors.append(self.evalTestError(test_usersitemsratings).eval())
            return(tferrors)

        def evalTestError(self, test_user_items_ratings):
            testusers, testitems, testratings = test_user_items_ratings
            testprediction = self.predict(testusers, testitems)
            diff = tf.subtract(testprediction, testratings, name='test_diff')
            sum_sqr_error = tf.reduce_sum(tf.math.square(diff, name="abs_difference"), name="sum_squared_errors")
            error = tf.divide(sum_sqr_error, tf.cast(len(testusers), tf.float32), name="mean_squared_error")
            return (error)

def LDA_LFM(n_iter, rank, num_topics, best_lambda, best_gamma, kappa, alpha, beta):

    start = time.time()
    print('Model fitting and testing')
    t = Tensorflow_LDA_LFM_version(mu, num_users, num_items, rank, best_lambda, kappa, best_gamma, alpha, beta, documents, vocab,itemWords, num_topics)
    users_items_ratings = sparse_to_UserItemRating(training_matrix_coo)
    test_users_items_ratings = sparse_to_UserItemRating(test_matrix_coo)
    obj = t.fit(users_items_ratings, test_users_items_ratings, n_iter)
    test_MSE = obj[n_iter-1]

    print('MSE of test set in LDA_LMF is:' + str(test_MSE))
    end = time.time()
    print("LDA_LFM RS is Finished in {} seconds".format(end - start))


