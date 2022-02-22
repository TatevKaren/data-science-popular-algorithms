#-----------------------------------------------------------------------------------------------------------
# K-means Analysis
#-----------------------------------------------------------------------------------------------------------
#data's for standard K-means and K-means PCA
data_kmeans = data.frame(customerid,Frequency_s, Recency_s, Monetary_s)
data_kmeans_pca = data.frame(customerid,pca_Recency, pca_Frequency, pca_Monetary)
set.seed(20190218) # we set the seed to get the same clusters each time
km = kmeans(x = data_kmeans[,-1], centers = 3) #note customer ID won't be used in kmeans
set.seed(20190218) # we set the seed to get the same clusters each time
km_pca = kmeans(x = data_kmeans_pca[,-1], centers = 3)
summary(km)
classkm = km$cluster #class varaible where 1:Good 2:Best 3: Better
table(classkm)
#k-means pca-based results
summary(km_pca)
classkm_pca = km_pca$cluster
table(classkm_pca)
# in order to have the desired order of class variable
# Creating new categorical varaible which will take following order of segments 1:Good, 2:Better, 3:Best
n = nrow(data_kmeans)
class = rep(0,n)
for(i in 1:n){
  if(classkm[i] == 1){
    class[i] = 1
  }
  else if(classkm[i] == 2){
    class[i] = 3 #Best
  }
  else if(classkm[i] == 3){
    class[i] = 2
  }
}
table(class)
#the real pca k-means class
class_pca = rep(0,n)
for(i in 1:n){
  if(classkm_pca[i] == 1){
    class_pca[i] = 1 
  }
  else if(classkm_pca[i] == 2){
    class_pca[i] = 3 #Best
  }
  else if(classkm_pca[i] == 3){
    class_pca[i] = 2
  }
}
table(class_pca)

# Plotting the resuts of K-means
plot(data_kmeans[,2:4],col = class,main="k-means clusters") #the standard kmeans results
plot(data_kmeans_pca[,2:4],col = class_pca,main="PCA k-means clusters") #the standard kmeans results
colors = c("#FFCC66", "#CC6633","#993300")
colors = colors[as.numeric(class)]
scatterplot3d(data_kmeans[,2:4], color=colors,angle = 155)
scatterplot3d(data_kmeans_pca[,2:4], color=colors, angle = 110)  

########Kmeans Father's day 2018
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_Fathersday_18_rfm`"
#######Kmeans Father's day 2017  
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_Fathersday_17_rfm`"
########Kmeans Easter 2018
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_Easter_18_rfm`"
#######Kmeans Easter 2017
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_Easter_17_rfm`"
#######Kmeans Sinterklass 2018
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_Sinterklass_18_rfm`"
#######Kmeans Sinterklass 2017
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_Sinterklass_17_rfm`"
#######Kmeans December 2017
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_December_17_rfm`"
#######Kmeans December 2018
sql7 <- "SELECT cast(lkuid as string) as customer_id, total_revenue as totrevenue, last_purchase_date as most_recent_visit, recency as recency_days, frequency as number_of_orders 
FROM `edc-gall-en-gall.data.H_December_18_rfm`"

#########After we choose the desirable holiday, we run the things below##########
comb_data <- query_exec(sql7, project, use_legacy_sql = FALSE, max_pages = Inf)
#varaible definitions for data(1)
customer_id = comb_data[,1]
totrevenue = comb_data[,2]
most_recent_visit = comb_data[,3]
recency_days = comb_data[,4]
number_of_orders = comb_data[,5]
data_customer <- data.frame(customer_id,totrevenue,most_recent_visit,number_of_orders,recency_days)
##########
data=data.frame(data_customer$totrevenue,data_customer$number_of_orders,data_customer$recency_days)
datascale=scale(data)
Clusters=kmeans(datascale, 3)
Clusters

