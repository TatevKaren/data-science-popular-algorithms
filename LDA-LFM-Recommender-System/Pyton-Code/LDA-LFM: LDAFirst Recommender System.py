# -----------------------------------------------------------------------
# LDA-based-model
# -----------------------------------------------------------------------
#to avoid arror with rank
rank = 5
def get_doc(review):
    revi = sum(review, [])
    return (revi)

# Function for getting all documents in the corpus
def get_documents(data):
    data['reviews'] = data['reviewText'].apply(clean)
    # newData['reviews']= pd.Series(newData['reviews']).apply(lambda x: x.split())
    documents = data[['asin', 'reviews']]
    documents = documents.groupby('asin', as_index=False).agg(lambda x: x.tolist())
    documents['documents'] = documents['reviews'].apply(get_doc)
    documents = documents[['asin', 'documents']]
    return (documents)


def get_prob(topic_dist):
    item_row = np.zeros(rank)
    topic_dist2 = np.asarray(topic_dist)
    if len(topic_dist) == rank:
        for i in range(len(topic_dist2)):
            item_row[i] = topic_dist2[i][1]
    else:
        for l in range(rank):
            for x in topic_dist:
                if x[0] == l:
                    item_row[l] = x[1]
    return (item_row)


def get_dictionary(documents):
    dictionary = corpora.Dictionary(documents['documents'])
    dictionary.filter_extremes(keep_n=5000)
    return (dictionary)


# Function for obtaining the matrix Q from LDA using gensim library
def get_LDA(dictionary, documents):

    # cleanining the reviewdata using 'reviewText' varaible
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in documents['documents']]
    LDA = gensim.models.ldamodel.LdaModel
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=rank, random_state=42, chunksize=1000,passes=1)

    def get_topic_dist(doc):
        bow = dictionary.doc2bow(doc)
        return (lda_model.get_document_topics(bow))

    documents['topic_dist'] = documents['documents'].apply(get_topic_dist)
    print('LDA finished, creating items/latent factors matrix Q')

    documents['topic_dist_new'] = documents['topic_dist'].apply(get_prob)
    num_items = newData.asin.unique().shape[0]
    Q = np.zeros(shape=(num_items, rank))
    term = 0
    for row in documents['topic_dist_new']:
        Q[term, :] = row
        term += 1
    Q = np.float32(Q)
    return (documents, Q, doc_term_matrix)

class Tensorflow_LDA_MF:

    def __init__(self, mu, num_users, num_items, rank, lambda_reg, Q):
        self.rank = rank
        self.num_users = num_users
        self.num_items = num_items
        self.lambda_reg = lambda_reg
        self.Q = Q
        self.mu = mu
        self.global_variables_initializer()

    def global_variables_initializer(self):
        self.b_u = tf.Variable(tf.random_normal([self.num_users, 1], stddev=0.01, mean=0), name="user_bias",trainable=True)
        self.b_i = tf.Variable(tf.random_normal([self.num_items, 1], stddev=0.01, mean=0), name="item_bias",trainable=True)
        self.P = tf.Variable(tf.random_normal([self.num_users, rank], stddev=0.01, mean=0), name="users",trainable=True)
        self.Q = tf.constant(self.Q, shape=[self.num_items, rank]) # Q is not trainable we hold it contant thorugh all iteration process

    def predict(self, users, items):
        P_ = tf.squeeze(tf.nn.embedding_lookup(self.P, users))
        Q_ = tf.squeeze(tf.nn.embedding_lookup(self.Q, items))
        predictions = tf.nn.sigmoid((tf.reduce_sum(tf.multiply(P_, Q_), reduction_indices=[1])))
        ubias = tf.squeeze(tf.nn.embedding_lookup(self.b_u, users))
        ibias = tf.squeeze(tf.nn.embedding_lookup(self.b_i, items))
        prediction = self.mu + ubias + ibias + tf.squeeze(predictions)
        return(prediction)

    def regLoss(self):
        reg_loss = 0
        reg_loss += tf.reduce_sum(tf.square(self.P))
        # reg_loss += tf.reduce_sum(tf.square(self.Q))
        reg_loss += tf.reduce_sum(tf.square(self.b_u))
        reg_loss += tf.reduce_sum(tf.square(self.b_i))
        return (reg_loss * self.lambda_reg)

    def loss(self, usersitemsratings):
        users, items, ratings = usersitemsratings
        prediction = self.predict(users, items)
        err_loss = tf.nn.l2_loss(prediction - ratings)
        reg_loss = self.regLoss()
        self.total_loss = err_loss + reg_loss
        tf.summary.scalar("loss", self.total_loss)
        return (self.total_loss)

    def fit(self, usersitemsratings, test_usersitemsratings = None, n_iter=n_iter):
        cost = self.loss(usersitemsratings)
        adam_train = tf.train.AdamOptimizer(0.001)
        gradients_and_vars = adam_train.compute_gradients(cost)
        optimizer = adam_train.apply_gradients(gradients_and_vars)
        with tf.Session() as sess:
            tf_errors = []
            sess.run(tf.global_variables_initializer())
            users, items, ratings = usersitemsratings
            for i in range(n_iter):
                sess.run(optimizer)
                tf_errors.append(self.evalTestError(test_usersitemsratings).eval())
        return(tf_errors)

    def evalTestError(self, test_usersitemsratings):
        testusers, testitems, testratings = test_usersitemsratings
        testprediction = self.predict(testusers, testitems)
        diff = tf.subtract(testprediction, testratings, name='test_diff')
        sum_sqr_error = tf.reduce_sum(tf.math.square(diff, name="sqr_diff"), name="sum_squared_errors")
        error = tf.divide(sum_sqr_error, tf.cast(len(testusers), tf.float32), name="mean_squared_error")
        return (error)

def LDA_based_LFM(n_iter, rank, lambda_reg):

    print('Initialization of LDA_MF model')
    documents = get_documents(train_data_reviews)
    dictionary = get_dictionary(documents)
    mu = calculate_global_mean_item_rating()
    num_users = newData.reviewerID.unique().shape[0]
    num_items = newData.asin.unique().shape[0]
    LDA_MF = get_LDA(dictionary, documents)
    Q = LDA_MF[1]
    Q = Q.T

    print('Model fitting and testing')
    start = time.time()
    t = Tensorflow_LDA_MF(mu, num_users, num_items, rank, lambda_reg, Q)
    usersitemsratings = sparse_to_UserItemRating(training_matrix_coo)
    test_usersitemsratings = sparse_to_UserItemRating(test_matrix_coo)
    obj = t.fit(usersitemsratings, test_usersitemsratings, n_iter)
    test_MSE = obj[n_iter-1]
    print('MSE of test set in LDA_MF is:' + str(test_MSE))
    end = time.time()
    print("LDA_Matrix_Factorization Finished in {} seconds".format(end - start))

