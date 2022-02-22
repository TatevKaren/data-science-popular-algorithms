#------------------------------------------------------------------------------------------------------------------
# Determining the data and movement in Good class
# Prediction estimation of K-means standard class
#------------------------------------------------------------------------------------------------------------------
Class1ind = as.numeric(rownames(table[class==1,])) #indicies of good customers from k-means
Class1data = table[Class1ind,]
n1 = length(Class1ind)
Good = rep(0,n1) #staying in good
UpGoodToBetter = rep(0,n1) #From Good to Better
UpGoodToBest = rep(0,n1) #From Good to Best

for(i in 1:n1){
  if(Class1data[i,4] == "1" & Class1data[i,5] == "2"){
    UpGoodToBetter[i] = 1
  }
  else if(Class1data[i,4] == "1" & Class1data[i,5] == "3"){
    UpGoodToBest[i] = 1
  }
  else if(Class1data[i,4] == "1" & Class1data[i,5] == "1"){
    Good[i] = 1
  }
} 
sum(Good)*100/n1
sum(UpGoodToBetter)*100/n1
sum(UpGoodToBest)*100/n1
#updating the data
Trans_class1_data = data.frame(Class1data[,1],Class1data[,4:5],Class1data[,7:14],Good,UpGoodToBetter, UpGoodToBest)
#corresponding indecies and data's of these customers
GoodInd = as.numeric(rownames(Trans_class1_data[Trans_class1_data[,12]==1,]))
UpGoodToBetterInd = as.numeric(rownames(Trans_class1_data[Trans_class1_data[,13]==1,]))
UpGoodToBestInd = as.numeric(rownames(Trans_class1_data[Trans_class1_data[,14]==1,]))
dataGood = Trans_class1_data[GoodInd,]
dataUpGoodToBetter =  Trans_class1_data[UpGoodToBetterInd,]
dataUpGoodToBest =  Trans_class1_data[UpGoodToBestInd,] 

#------------------------------------------------------------------------------------------------------------------
#Determining the data and movement in Better class
#------------------------------------------------------------------------------------------------------------------
Class2ind = as.numeric(rownames(table[class ==2,])) #indicies of good customers from k-means
Class2data = table[Class2ind,]
n2 = length(Class2ind)
Better = rep(0,n2) #staying in good
UpBetterToBest = rep(0,n2) #From Better to Best
DownBetterToGood = rep(0,n2) #From Better to Good

for(i in 1:n2){
  if(Class2data[i,4] == "2" & Class2data[i,5] == "2"){
    Better[i] = 1
  }
  else if(Class2data[i,4] == "2" & Class2data[i,5] == "3"){
    UpBetterToBest[i] = 1
  }
  else if(Class2data[i,4] == "2" & Class2data[i,5] == "1"){
    DownBetterToGood[i] = 1
  }
} 
sum(Better)*100/n2
sum(UpBetterToBest)*100/n2
sum(DownBetterToGood)*100/n2

#updating the data
Trans_class2_data = data.frame(Class2data[,1],Class2data[,4:5],Class2data[,7:14],Better,UpBetterToBest,DownBetterToGood)
#corresponding indecies and data's of these customers
BetterInd = as.numeric(rownames(Trans_class2_data[Trans_class2_data[,12]==1,]))
UpBetterToBestInd = as.numeric(rownames(Trans_class2_data[Trans_class2_data[,13]==1,]))
DownBetterToGoodInd = as.numeric(rownames(Trans_class2_data[Trans_class2_data[,14]==1,]))
#the data of each group
dataBetter = Trans_class2_data[BetterInd,]
dataUpBetterToBest =  Trans_class2_data[UpGoodToBetterInd,]
dataDownBetterToGood =  Trans_class2_data[DownBetterToGoodInd,] 

#------------------------------------------------------------------------------------------------------------------
#Determining the data and movement in Best class
#------------------------------------------------------------------------------------------------------------------
Class3ind = as.numeric(rownames(table[class ==3,])) #indicies of good customers from k-means
Class3data = table[Class3ind,]
n3 = length(Class3ind)
Best = rep(0,n3) #staying in good
DownBestToBetter = rep(0,n3) #from best to better
DownBestToGood = rep(0,n3) #from best to good

for(i in 1:n3){
  if(Class3data[i,4] == "3" & Class3data[i,5] == "3"){
    Best[i] = 1
  }
  else if(Class3data[i,4] == "3" & Class3data[i,5] == "2"){
    DownBestToBetter[i] = 1
  }
  else if(Class3data[i,4] == "3" & Class3data[i,5] == "1"){
    DownBestToGood[i] = 1
  }
} 
sum(Best)*100/n3
sum(DownBestToBetter)*100/n3
sum(DownBestToGood)*100/n3
#updating the data
Trans_class3_data = data.frame(Class3data[,1],Class3data[,4:5],Class3data[,7:14],Best,DownBestToBetter,DownBestToGood)
#corresponding indecies and data's of these customers
BestInd = as.numeric(rownames(Trans_class3_data[Trans_class3_data[,12]==1,]))
DownBestToBetterInd = as.numeric(rownames(Trans_class3_data[Trans_class3_data[,13]==1,]))
DownBestToGoodInd = as.numeric(rownames(Trans_class3_data[Trans_class3_data[,14]==1,]))
#the data of each group
dataBest = Trans_class3_data[BestInd,]
dataDownBestToBetter =  Trans_class3_data[DownBestToBetterInd,]
dataDownBestToGood =  Trans_class3_data[DownBestToGoodInd,] 
