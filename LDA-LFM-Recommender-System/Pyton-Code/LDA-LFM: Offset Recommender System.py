# -------------------------------------------------------------------------------------------------------
# Offset Recommender System
# -------------------------------------------------------------------------------------------------------
# Function for calculating global avarege of the training data
def calculate_global_mean_item_rating():
    summed_item_rating = 0
    for i, j, v in zip(training_matrix_coo.row, training_matrix_coo.col, training_matrix_coo.data):
        summed_item_rating = summed_item_rating + v
    num_ratings = newData.overall.shape[0]
    global_mean = float(summed_item_rating) / num_ratings
    return (global_mean)

def calculate_error_test(estimated_rating, true_rating):
    error = true_rating - estimated_rating
    return (error)

# Function for calculating the MSE of Offset model
def OffsetRS_MSE():
    summed_sqr_error = 0
    # calculating global mean on training dataset
    global_mean = calculate_global_mean_item_rating()
    estimated_rating = global_mean

    # Loop through each entry in the test dataset
    for user, item, true_rating in zip(test_matrix_coo.row, test_matrix_coo.col, test_matrix_coo.data):
        # Calculate the error between the predicted rating and the true rating
        summed_sqr_error = summed_sqr_error + math.pow(calculate_error_test(estimated_rating, true_rating), 2)
        # summed_abs = summed_abs + calculate_error_test(estimated_rating, true_rating)

    # Calculate the number of entries in the test set
    test_dataset_size = test_matrix_coo.nnz
    # Compute the MSE on the test set
    MSE = float(summed_sqr_error) / test_dataset_size
    return(MSE)


def OffsetRS():
    start = time.time()
    MSE = OffsetRS_MSE()
    print('MSE of Baseline RS is:' + str(MSE))
    end = time.time()
    print("Offset Recommender System of Ratings Finished in {} seconds".format(time.time() - start))

