#------------------------------------------------------------------------------------------------------------------
# Algorithm to identifiy potential customers, cussutomers who are likely to move up or down to other class
#------------------------------------------------------------------------------------------------------------------
rate = rate_class[,-c(1,5)]
rate_pca = rate_class_pca[,-c(1,5)]

realClass = as.character(rate_class[,5])
realClass_pca = as.character(rate_class_pca[,5])

customerid = rate_class[,1]
customerid_pca = rate_class_pca[,1]

newClass = rep(NA,nrow(rate))
newClass_pca = rep(NA,nrow(rate_pca))

max.prob = rep(0,nrow(rate))
max.prob_pca = rep(0,nrow(rate_pca))

max.position = rep(NA,nrow(rate))
max.position_pca = rep(NA,nrow(rate_pca))

for(i in 1:nrow(rate)){
  new = NULL
  max.prob[i] = max(rate[i,])
  new = which(rate[i,] == max.prob[i])
  max.position[i] = paste(new,collapse="--")
  if(length(new)==1) {newClass[i] = new} 
  if (length(new)!=1) {
    if(sum(new==realClass[i]) == 1) newClass[i] = realClass[i]
    if(sum(new==realClass[i]) == 0) newClass[i] = max.position[i]
  }
}
for(i in 1:nrow(rate_pca)){
  new_pca = NULL
  max.prob_pca[i] = max(rate_pca[i,])
  new_pca = which(rate_pca[i,] == max.prob_pca[i])
  max.position_pca[i] = paste(new_pca,collapse="--")
  if(length(new_pca)==1) {newClass_pca[i] = new_pca} 
  if (length(new_pca)!=1) {
    if(sum(new_pca==realClass_pca[i]) == 1) newClass_pca[i] = realClass_pca[i]
    if(sum(new_pca==realClass_pca[i]) == 0) newClass_pca[i] = max.position_pca[i]
  }
}

table = cbind(customerid,max.prob,max.position,realClass,newClass,tree_data)
table_pca = cbind(customerid_pca,max.prob_pca,max.position_pca,realClass_pca,newClass_pca,tree_data_pca)

#------------------------------------------------------------------------------------------------------------------
# Determining the data and movement in Good class PCA
#------------------------------------------------------------------------------------------------------------------
Class1ind_pca = as.numeric(rownames(table_pca[table_pca[,4] ==1,])) #indicies of good customers from k-means
Class1data_pca = table_pca[Class1ind_pca,]
nk = nrow(Class1data_pca)
Good_pca = rep(0,nk) #staying in good
UpGoodToBetter_pca = rep(0,nk) #From Good to Better
UpGoodToBest_pca = rep(0,nk) #From Good to Best

for(i in 1:nk){
  if(Class1data_pca[i,5] == 2){
    UpGoodToBetter_pca[i] = 1
  }
  else if( Class1data_pca[i,5] == 3){
    UpGoodToBest_pca[i] = 1
  }
  else if(Class1data_pca[i,5] == 1){
    Good_pca[i] = 1
  }
} 

#updating the data
Trans_class1_data_pca = data.frame(Class1data_pca,Good_pca, UpGoodToBetter_pca, UpGoodToBest_pca)
#corresponding indecies and data's of these customers
GoodInd_pca = as.numeric(rownames(Trans_class1_data_pca[Trans_class1_data_pca[,16]==1,]))
UpGoodToBetterInd_pca = as.numeric(rownames(Trans_class1_data_pca[Trans_class1_data_pca[,17]==1,]))
UpGoodToBestInd_pca = as.numeric(rownames(Trans_class1_data_pca[Trans_class1_data_pca[,18]==1,]))
dataGood_pca = Trans_class1_data_pca[GoodInd_pca,]
dataUpGoodToBetter_pca =  Trans_class1_data_pca[UpGoodToBetterInd_pca,]
dataUpGoodToBest_pca =  Trans_class1_data_pca[UpGoodToBestInd_pca,] 


#------------------------------------------------------------------------------------------------------------------
#Determining the data and movement in Better class PCA
#------------------------------------------------------------------------------------------------------------------
Class2ind_pca = as.numeric(rownames(table_pca[table_pca[,4] ==2,])) #indicies of good customers from k-means
Class2data_pca = table[Class2ind_pca,]
nk = length(Class2ind_pca)
Better_pca = rep(0,nk) #staying in good
UpBetterToBest_pca = rep(0,nk) #From Better to Best
DownBetterToGood_pca = rep(0,nk) #From Better to Good

for(i in 1:nk){
  if(Class2data_pca[i,5] == "2"){
    Better_pca[i] = 1
  }
  else if(Class2data_pca[i,5] == "3"){
    UpBetterToBest_pca[i] = 1
  }
  else if(Class2data_pca[i,5] == "1"){
    DownBetterToGood_pca[i] = 1
  }
} 

#updating the data
Trans_class2_data_pca = data.frame(Class2data_pca, Better_pca, UpBetterToBest_pca, DownBetterToGood_pca)
#corresponding indecies and data's of these customers
BetterInd_pca = as.numeric(rownames(Trans_class2_data_pca[Trans_class2_data_pca[,16]==1,]))
UpBetterToBestInd_pca = as.numeric(rownames(Trans_class2_data_pca[Trans_class2_data_pca[,17]==1,]))
DownBetterToGoodInd_pca = as.numeric(rownames(Trans_class2_data_pca[Trans_class2_data_pca[,18]==1,]))
#the data of each group
dataBetter_pca = Trans_class2_data_pca[BetterInd_pca,]
dataUpBetterToBest_pca =  Trans_class2_data_pca[UpGoodToBetterInd_pca,]
dataDownBetterToGood_pca =  Trans_class2_data_pca[DownBetterToGoodInd_pca,] 


#------------------------------------------------------------------------------------------------------------------
#Determining the data and movement in Best class PCA
#------------------------------------------------------------------------------------------------------------------
Class3ind_pca = as.numeric(rownames(table[table_pca[,4] ==3,])) #indicies of good customers from k-means
Class3data_pca = table[Class3ind_pca,]
nk = length(Class3ind_pca)
Best_pca = rep(0,nk) #staying in good
DownBestToBetter_pca = rep(0,nk) #from best to better
DownBestToGood_pca = rep(0,nk) #from best to better

for(i in 1:nk){
  if( Class3data_pca[i,5] == "3"){
    Best_pca[i] = 1
  }
  else if(Class3data_pca[i,5] == "2"){
    DownBestToBetter_pca[i] = 1
  }
  else if(Class3data_pca[i,5] == "1"){
    DownBestToGood_pca[i] = 1
  }
} 
#updating the data
Trans_class3_data_pca = data.frame(Class3data_pca, Best_pca, DownBestToBetter_pca, DownBestToGood_pca)
#corresponding indecies and data's of these customers
BestInd_pca = as.numeric(rownames(Trans_class3_data_pca[Trans_class3_data_pca[,16]==1,]))
DownBestToBetterInd_pca = as.numeric(rownames(Trans_class3_data_pca[Trans_class3_data_pca[,17]==1,]))
DownBestToGoodInd_pca = as.numeric(rownames(Trans_class3_data_pca[Trans_class3_data_pca[,18]==1,]))
#the data of each group
dataBest_pca = Trans_class3_data_pca[BestInd_pca,]
dataDownBestToBetter_pca =  Trans_class3_data_pca[DownBestToBetterInd_pca,]
dataDownBestToGood_pca =  Trans_class3_data_pca[DownBestToGoodInd_pca,] 