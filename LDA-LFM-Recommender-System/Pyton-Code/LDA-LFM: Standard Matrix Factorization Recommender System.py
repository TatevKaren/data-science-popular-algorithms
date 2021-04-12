# -------------------------------------------------------------------------------------------------------
# Standard Matrix Factorization Recommender System
# -------------------------------------------------------------------------------------------------------
class TensorflowMF:

    def __init__(self, mu, num_users, num_items, rank, lambda_reg):
        self.rank = rank
        self.num_users = num_users
        self.num_items = num_items
        self.lambda_reg = lambda_reg
        self.mu = mu
        self.global_variables_initializer()

    def global_variables_initializer(self):
        self.b_u = tf.Variable(tf.random_normal([self.num_users, 1], stddev=0.01, mean=0), name="user_bias", trainable=True)
        self.b_i = tf.Variable(tf.random_normal([self.num_items, 1], stddev=0.01, mean=0), name="item_bias",trainable=True)
        self.P = tf.Variable(tf.random_normal([self.num_users, self.rank], stddev=0.01, mean=0), name="users",trainable=True)
        self.Q = tf.Variable(tf.random_normal([self.num_items, self.rank], stddev=0.01, mean=0), name="items", trainable=True)

    def predict(self, users, items):
        P_ = tf.squeeze(tf.nn.embedding_lookup(self.P, users))
        Q_ = tf.squeeze(tf.nn.embedding_lookup(self.Q, items))
        prediction = tf.nn.sigmoid((tf.reduce_sum(tf.multiply(P_, Q_), reduction_indices=[1])))
        ubias = tf.squeeze(tf.nn.embedding_lookup(self.b_u, users))
        ibias = tf.squeeze(tf.nn.embedding_lookup(self.b_i, items))
        prediction = self.mu + ubias + ibias + tf.squeeze(prediction)
        return (prediction)

    def regLoss(self):
        reg_lo = 0
        reg_lo += tf.reduce_sum(tf.square(self.P))
        reg_lo += tf.reduce_sum(tf.square(self.Q))
        reg_lo += tf.reduce_sum(tf.square(self.b_u))
        reg_lo += tf.reduce_sum(tf.square(self.b_i))
        return (reg_lo * self.lambda_reg)

    def loss(self, usersitemsratings):
        users, items, ratings = usersitemsratings
        predictions = self.predict(users, items)
        errloss = tf.nn.l2_loss(predictions - ratings)
        regloss = self.regLoss()
        self.totalloss = errloss + regloss
        tf.summary.scalar("total_loss", self.totalloss)
        return (self.totalloss)

    def fit(self, usersitemsratings, test_usersitemsratings=None, n_iter=n_iter):
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
        return (tf_errors)

    def evalTestError(self, test_usersitemsratings):
        testusers, testitems, testratings = test_usersitemsratings
        testpredictions = self.predict(testusers, testitems)
        diff = tf.subtract(testpredictions, testratings, name='test_diff')
        diff_abs = tf.abs(diff, name="abs_difference")
        sum_sqr_error = tf.reduce_sum(tf.math.square(diff_abs, name="abs_difference"), name="sum_squared_errors")
        error = tf.divide(sum_sqr_error, tf.cast(len(testusers), tf.float32), name="mean_squared_error")
        return (error)

def Standard_LFM(n_iter, rank, lambda_reg):
    start = time.time()
    mu = calculate_global_mean_item_rating()
    num_users = newData.reviewerID.unique().shape[0]
    num_items = newData.asin.unique().shape[0]
    t = TensorflowMF(mu, num_users, num_items, rank, lambda_reg)
    usersitemsratings = sparse_to_UserItemRating(training_matrix_coo)
    test_usersitemsratings = sparse_to_UserItemRating(test_matrix_coo)
    obj = t.fit(usersitemsratings, test_usersitemsratings, n_iter)
    test_MSE = obj[n_iter-1]
    print('MSE of test set in LFM is:' + str(test_MSE))
    end = time.time()
    print("Latent Factor Model Finished in {} seconds".format(end - start))


