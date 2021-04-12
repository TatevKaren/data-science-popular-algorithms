# -------------------------------------------------------------------------------------------------------
# Baseline Recommender System
# -------------------------------------------------------------------------------------------------------
def calculate_mean_item_rating():
    item_centered = {}
    item_average = {}
    # Calculate the mean of each item
    item_sums = training_matrix_csr.sum(axis=1)
    # Calculate the number of ratings for each item
    item_rating_counts = training_matrix_csr.getnnz(axis=1)
    global_mean = calculate_global_mean_item_rating()
    # Loop through each item
    number_of_items = training_matrix_csc.shape[1]
    for index in range(0, number_of_items):
        # Check if the item has not been rated
        if item_sums[index] != 0:
            item_average[index] = float(item_sums[index]) / item_rating_counts[index]
            item_centered[index] = item_average[index] - global_mean
        else:

            item_average[index] = 0
            item_centered[index] = 0
    return (item_centered, item_average)


def calculate_mean_user_rating():
    user_centered = {}
    user_average = {}
    # Calculate the mean of each user
    user_sums = training_matrix_csc.sum(axis=1)
    # Reshape the matrix to array form for proper indexing
    user_sums = user_sums.reshape((user_sums.size, 1))
    # Calculate the number of ratings for each user
    user_rating_counts = training_matrix_csc.getnnz(axis=1)
    global_mean = calculate_global_mean_item_rating()
    # Loop through each user
    number_of_users = training_matrix_csc.shape[0]
    for index in range(0, number_of_users):
        # Check to see if the user has not rated
        if user_sums[index] != 0:
            user_average[index] = float(user_sums[index]) / user_rating_counts[index]
            user_centered[index] = user_average[index] - global_mean
        else:
            user_average[index] = 0
            user_centered[index] = 0
    return (user_centered, user_average)


def baseline_ratings_MSE():
    baseline_rating = {}
    global_mean = calculate_global_mean_item_rating()
    item_centered = calculate_mean_item_rating()[0]
    user_centered = calculate_mean_user_rating()[0]
    summed_sqr_error = 0

    # Loop through each entry in the test dataset
    for user, item, true_rating in zip(test_matrix_coo.row, test_matrix_coo.col, test_matrix_coo.data):
        # Get the baseline rating for this item and user in the test set
        bi = item_centered[item]
        bu = user_centered[user]
        estimated_rating = bi + bu + global_mean
        baseline_rating[(item, user)] = estimated_rating
        # Calculate the error between the predicted rating and the true rating
        summed_sqr_error = summed_sqr_error + math.pow(calculate_error_test(estimated_rating, true_rating), 2)
    # Calculate the number of entries in the test set
    test_dataset_size = test_matrix_coo.nnz
    # Compute the RMSE on the test set
    MSE = float(summed_sqr_error) / test_dataset_size
    return (MSE)


def baseline_RS():
    start = time.time()
    MSE = baseline_ratings_MSE()
    print('MSE of Baseline RS is:' + str(MSE))
    end = time.time()
    print("Baseline Estimate of Ratings Finished in {} seconds".format(time.time() - start))
