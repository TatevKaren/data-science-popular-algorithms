
# -------------------------------------------------------------------------------------------------------
# Function generating the sparse rating matrix in csr, cpp and csc formats
# -------------------------------------------------------------------------------------------------------
def sparsematrix(datar):
    # Getting the rating matrix
    n_users = datar.reviewerID.unique().shape[0]
    n_items = datar.asin.unique().shape[0]
    users_locations = datar.groupby(by=['reviewerID', 'asin', 'overall']).apply(lambda x: 1).to_dict()
    row, col, value = zip(*(users_locations.keys()))  # row-> users,  col-> locations
    map_u = dict(zip(datar['reviewerID'].unique(), range(n_users)))
    map_l = dict(zip(datar['asin'].unique(), range(n_items)))
    row_idx = [map_u[u] for u in row]
    col_idx = [map_l[l] for l in col]
    data = np.array(value)
    sparse_csr = csr_matrix((data, (row_idx, col_idx)), shape=(n_users, n_items))
    coo_format_sparse = sparse_csr.tocoo([False])
    csc_format_sparse = sparse_csr.tocsc([False])
    return(sparse_csr, coo_format_sparse, csc_format_sparse,data)


# -------------------------------------------------------------------------------------------------------
# Function for determining the sparsity level of the data
# -------------------------------------------------------------------------------------------------------
def get_sparsity(datar):
    num_users = datar.reviewerID.unique().shape[0]
    num_items = datar.asin.unique().shape[0]
    ratings = sparsematrix(datar)[3]
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (num_users * num_items)
    sparsity *= 100
    print('Sparsity:'+ str(sparsity))
    return(sparsity)

