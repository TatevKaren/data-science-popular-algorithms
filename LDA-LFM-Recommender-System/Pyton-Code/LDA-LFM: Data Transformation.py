# -------------------------------------------------------------------------------------------------------
# Function transforming sparse rating matrix to a tuples of user, item, rating
# -------------------------------------------------------------------------------------------------------
def sparse_to_UserItemRating(sparse_matrix_coo):
    temp = sparse_matrix_coo
    user = temp.row.reshape(-1, 1)
    item = temp.col.reshape(-1, 1)
    rating = temp.data
    return user, item, rating


# -------------------------------------------------------------------------------------------------------
# Function of splitting the data into train/test/validation sets
# -------------------------------------------------------------------------------------------------------
# to obtain 80/10/10 sets depending on the dataset test_size = 0.5-0.7 usually does this
def get_data(rating_data, test_size):
    df = rating_data[['reviewerID', 'asin', 'overall', 'reviewText']]
    unique_users= sorted(pd.unique(df['reviewerID']))
    unique_items = sorted(pd.unique(df['asin']))
    uid2idx = dict((uid, idx) for (idx, uid) in enumerate(unique_users))
    sid2idx = dict((sid, idx) for (idx, sid) in enumerate(unique_items))
    n_ratings = df.shape[0]
    test = np.random.choice(n_ratings, size=int(test_size * n_ratings), replace=False)
    test_items = np.zeros(n_ratings, dtype=bool)
    test_items[test] = True
    test_df = df[['reviewerID', 'asin', 'overall']][test_items]
    train_df = df[['reviewerID', 'asin', 'overall']][~test_items]
    test_df_rev = df[['reviewerID', 'asin', 'reviewText']][test_items]
    train_df_rev = df[['reviewerID', 'asin', 'reviewText']][~test_items]
    train_uid = set(pd.unique(train_df['reviewerID']))
    left_uid = list()
    for i, uid in enumerate(pd.unique(df['reviewerID'])):
        if uid not in train_uid:
            left_uid.append(uid)

    #id's of users that should be moved back to the training set
    move_ids = test_df['reviewerID'].isin(left_uid)
    train_df = train_df.append(test_df[move_ids])
    test_df = test_df[~move_ids]
    train_df_rev = train_df_rev.append(test_df_rev[move_ids])
    test_df_rev = test_df_rev[~move_ids]

    #id's of items that should be moved back to the training set
    train_items = set(pd.unique(train_df['asin']))
    left_sid = list()
    for i, sid in enumerate(pd.unique(df['asin'])):
        if sid not in train_items:
            left_sid.append(sid)
    move_ids2 = test_df['asin'].isin(left_sid)

    train_df = train_df.append(test_df[move_ids2])
    train_df_rev = train_df_rev.append(test_df_rev[move_ids2])
    test_df = test_df[~move_ids2]
    test_df_rev = test_df_rev[~move_ids2]

    #then the test set is equally devided to validation set and test set
    num = round(len(test_df)/2)
    valid_df = test_df[(num+1):len(test_df)-1]
    valid_df_rev = test_df_rev[(num+1):len(test_df)-1]
    test_df = test_df[0:num]
    test_df_rev = test_df_rev[0:num]

    # determining the sparse user-item rating matrices in three formats using above datasets
    train_sparse_csr = sparsematrix(train_df)[0]
    test_sparse_csr = sparsematrix(test_df)[0]
    valid_sparse_csr = sparsematrix(valid_df)[0]

    train_coo_matrix = sparsematrix(train_df)[1]
    test_coo_matrix = sparsematrix(test_df)[1]
    valid_coo_matrix = sparsematrix(valid_df)[1]

    train_csc_matrix = sparsematrix(train_df)[2]
    test_csc_matrix = sparsematrix(test_df)[2]
    valid_csc_matrix = sparsematrix(valid_df)[2]

    print("Size of the Training set")
    print(len(train_df)/len(rating_data))
    print("Size of the Test set")
    print(len(test_df)/len(rating_data))
    print("Size of the Validation set")
    print(len(valid_df)/len(rating_data))

    return(train_sparse_csr, train_coo_matrix, train_csc_matrix, train_df, test_sparse_csr, test_coo_matrix, test_csc_matrix,
           test_df, train_df_rev, test_df_rev,valid_sparse_csr, valid_coo_matrix, valid_csc_matrix, valid_df, valid_df_rev)


# -------------------------------------------------------------------------------------------------------
# Generating the required datasets using the data
# -------------------------------------------------------------------------------------------------------
object = get_data(newData, 0.60)

training_matrix_csr = object[0]
training_matrix_coo = object[1]
training_matrix_csc = object[2]
train_data = object[3]
train_data_reviews = object[8]

test_matrix_csr = object[4]
test_matrix_coo = object[5]
test_matrix_csc = object[6]
test_data = object[7]
test_data_reviews = object[9]

valid_matrix_csr = object[10]
valid_matrix_coo = object[11]
valid_matrix_csc = object[12]
valid_data = object[13]
valid_data_reviews = object[14]
