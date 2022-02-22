#------------------------------------------------------------------------------------------------------------------
# Decision-tree analysis
#------------------------------------------------------------------------------------------------------------------
#data preparation for the tree, now we also include demographics
tree_data = data.frame(customerid,Frequency, Recency, Monetary, Female, Age, Allow_analysis,Opt_in_com, Loyal_time ,class)
colnames(tree_data) = c("ID","Frequency","Recency","Monetary","Female", "Age","Opt_Analysis", "Opt_Commercial","Loyal_time","Class")
#data with pca_class from k-means pca
tree_data_pca = data.frame(customerid,Frequency,Recency,Monetary,Female,Age,Allow_analysis,Opt_in_com,Loyal_time,class_pca)
colnames(tree_data_pca) = c("ID","Frequency","Recency","Monetary","Female", "Age","Opt_Analysis", "Opt_Commercial","Loyal_time","Class_pca")

#Standard tree
tree.fit = rpart(Class ~ Recency+ Frequency + Monetary + Female + Age + Allow_analysis + Opt_in_com + Loyal_time, data = tree_data, method = "class") #the default uses the gini index
tree.fit
rpart.plot(tree.fit,box.palette=list("peachpuff1", "gray75","darkorange"))
#pca tree
tree.fit_pca = rpart(Class_pca ~ Recency+ Frequency + Monetary + Female + Age + Allow_analysis + Opt_in_com + Loyal_time, data = tree_data_pca, method = "class")
tree.fit_pca
rpart.plot(tree.fit_pca,box.palette=list("peachpuff1", "gray75","darkorange"))

#Tree pruning
printcp(tree.fit)
plotcp(tree.fit)
printcp(tree.fit_pca)
plotcp(tree.fit_pca)

#Pruning the tree and plotting the pruned tree by using the minimal gini index criteria
tree.pruned = prune(tree.fit, cp = 0.063609)
rpart.plot(tree.pruned, box.palette=list("peachpuff1", "gray75","darkorange"))
tree.pruned_pca = prune(tree.fit_pca, cp = 0.10374)
rpart.plot(tree.pruned_pca, box.palette=list("peachpuff1", "gray75","darkorange"))

#Prediction of the classes and transitional probabilities
pred = predict(tree.pruned,data = tree_data, method = "class")
pred_pca = predict(tree.pruned_pca,data = tree_data_pca, method = "class")
rate_class = data.frame(tree_data[,1], pred, class)
rate_class_pca = data.frame(tree_data_pca[,1], pred_pca, class_pca)
colnames(rate_class) = c("customerid","trans.class1","trans.class2","trans.class3","kmeans.class")
colnames(rate_class_pca) = c("customerid","trans.class1","trans.class2","trans.class3","kmeans.class_pca")
