# Data Science (Code and Papers)
# 1: TopN Movie Recommender Case Study
File name: TopN_MovieRecommender.py, TopN_MovieRecommender.pdf

The MovieLens dataset consisting of 20M ratings and 466K tag applications across 27K movies spanning a period of 20 years, from January 1995 to March 2015, is used to constract simple Top N Movie Recommender that generates N recommendations for a user using Item based Nearest-Neighbor Collaborative Filtering. In case of Item-based CF, the system finds similar items and assuming that similar items will be rated in a similar way by the same person, it predicts the rating corresponding to a user assigned to that item. Usually, the Item-based CF is preferred over the User-based CF because users can change their preferences and choices (aging, change of environment) whereas items (relatively) does not change over time.

Files consists of the following parts:
- Introduction to recommender systems
- Description and Descriptive Statistics for MovieLens data
- MovieLens data visualization
- Methodology: description of Collaborative Filtering (CF) and KNN algorithmms
- Top-N Movie Recommender System Algorithm step-by-step
- Evaluation of the results 

Publications: 
- Aggarwal, C. (2016). Recommender Systems. Thomas J. Watson Research Center.
- Billsus, D. and Pazzani, M. J. (1998). Learning collaborative information filters. In Proceedings of the Fifteenth International Conference on Machine Learning, pages 46–54. Morgan Kaufmann Publishers Inc.

# 2: Linear Discriminant Analysis (LDA) Algorithm
File name: LDA.R 
Programming Language: R 
Note: the code contains LDA and robust LDA mannually written functions (checked with the library function's output)

Linear discriminant analysis (LDA) (don't confuss this with Latent Dirichlit Allocation which is Topic Modelling technique) is a generalization of Fisher's linear discriminant, which is a statistical method to find a linear combination of features that characterizes/separates two or more classes of objects. The resulting combination may be used as a linear classifier. LDA is closely related to analysis of variance (ANOVA) and regression analysis, which also attempt to express one (dependent) variable as a linear combination of other (independent) variables. However, ANOVA uses a continuous dependent variable and categorical independent variables, whereas LDA uses a categorical dependent variable (classes of LDA) and continuous independent variables. Logistic regression and Probit regression are more similar to LDA than ANOVA is, as they also explain a categorical (dependent) variable by the values of continuous (independent) variables. The key difference between Logistic Regression/Probit regression and LDA is the assumption about the probability distribution about the explanatory (independent) variables. In case of LDA , fundamental assumtion is that the independent variables are normally distributed. This can be checked by looking at the probability distribution of the variables.

Publications:

- Nasar, S., Aldian, A., Nuredin, J., and Abusaeeda, I. (2016). Classification depend on linear discriminant analysis using desired outputs. 1109(10)
- Zhao, H., Wang, Z., and Nie, F. (2019) A New Formulation of Linear Discriminant Analysis for Robust Dimensionality Reduction. 31(4):629-640


# 3: K-Means Algorithm
File name: K_Means_Clustering.R, K_Means_Algorithm.pdf

K-means clustering is a method of vector quantization with a goal to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. k-means clustering minimizes within-cluster variances (squared Euclidean distances). This algorithm is also referred to as Lloyd's algorithm, particularly in the computer science community. It is sometimes also referred to as "naïve k-means", because there exist much faster alternatives. Target number k needs to be pre-determined, it refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster. Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares. In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.

Publications:
- Forgy,  E.  W.  (1965).   Cluster  analysis  of  multivariate  data:  efficiency  versus  interpretability  ofclassifications.biometrics, 21:768–769.
- Na, S., Xumin, L. and Yong, G. (2010), Research on k-means Clustering Algorithm: An Improved k-means Clustering Algorithm, 2010, pp. 63-67.


# 4: Descision Tree Algorithm
File name: Decision_Tree_Clustering.R, Decision_Trees_Clustering.pdf

Decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including event probabilities. In Machine Learning, this algorithm is often referred as "Decision Tree Learning". Decision Tree Learning is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a Decision Tree (as a predictive model) to cluster the entire sample of observations into clsuters (represented by the leaves of the table). There are two type of Decision Trees: Classification and Regression Trees. Tree models where the target variable can take a discrete set of values are called classification trees; in this type of tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision Trees where the target variable can take continuous values (usually real numbers) are called regression trees. Because of its intelligibility and simplicity, Decision Tree Algorithms are considered one of most popular ML algorithms.

Publications:
- Haughton, D. and Oulabi, S. (1993). Direct marketing modeling with cart and chaid. Journal of direct marketing, 7(3):16–26
- Duchessi, P. and Lauria, E. (2013). Decision tree models for profiling ski resorts’ promotional and advertising strategies and the impact on sales. 40(15):5822–5829.








