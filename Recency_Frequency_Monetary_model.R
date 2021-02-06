# Installing and loading packages 
install.packages("bigrquery")
install.packages("readr")
install.packages("lubridate")
install.packages("dplyr")
install.packages("rfm")
install.packages("ggplot2")
install.packages("magrittr")
install.packages("VIM")
install.packages("mice")
install.packages("psych")
install.packages("cluster")
install.packages("scatterplot3d") 
install.packages("arsenal")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("flexmix")
install.packages("countreg", repos="http://R-Forge.R-project.org")
install.packages("amap")
install.packages("tidyr")
install.packages("clues")
library(bigrquery)
library(readr)
library(lubridate)
library(dplyr)
library(rfm)
library(ggplot2)
library(magrittr) 
library(VIM)
library(mice)
library(psych)
library(cluster)
library(scatterplot3d) 
library(arsenal)
library(rpart)
library(rpart.plot)
library(flexmix)
library(countreg)
library(tidyr)
library(clues)
library(amap)
#-----------------------------------------------------------------------------------------------------------
# Setting the project and filename
#-----------------------------------------------------------------------------------------------------------
filename <-"/content/datalab/notebooks/config.json"
token = readChar(filename, file.info(filename)$size)
set_service_token(token)
analysis_date = lubridate::as_date('2018-12-31', tz = 'UTC') #setting the analysis date benchmark
#-----------------------------------------------------------------------------------------------------------
# Graph: The orders histogram
#-----------------------------------------------------------------------------------------------------------
sql3 <- "SELECT *
FROM `edc-gall-en-gall.data.figure33`"
order_data2 <- query_exec(sql3, project, use_legacy_sql = FALSE, max_pages = Inf)
orders=order_data2[,1]
customers=order_data2[,2]
newdata=order_data2[order(orders),]
newdata=newdata[1:20,]
ggplot(newdata, aes(x=orders, y=customers)) + geom_bar(stat = "identity")+geom_text(aes(label=customers), vjust=1.6, color="orange", size=3.5)

#-----------------------------------------------------------------------------------------------------------
# Imortting the demographic variables from Big Query table tree_table_new2
#-----------------------------------------------------------------------------------------------------------
sql10 = "SELECT cast(lkuid as string) as customer_id , cast(allow_analysis as string) as allow_analysis, gender as gender,
cast(opt_in_commercial_email as string)as opt_in,age as age,age_category as age_category,loyal_time as loyal_time
FROM `edc-gall-en-gall.data.tree_table_new2`
Order by lkuid"
demdata = query_exec(sql10, project, use_legacy_sql = FALSE, max_pages = Inf)
colnames(demdata) = names(demdata)
allow_analysis = demdata[,2]
gender = demdata[,3]
opt_in = demdata[,4]
age = demdata[,5] 
age_category = demdata[,6] #categorical age variable
loyal_time = demdata[,7]
#-----------------------------------------------------------------------------------------------------------
# Creating dummy varaible for GENDER from variable gender 1:female 0:male
#-----------------------------------------------------------------------------------------------------------
n = nrow(demdata)
GENDER = rep(0,n) 
for(i in 1:n){
  if(demdata[i,3] =="m"){ #replace 0's by 1 if not male
    GENDER[i] = 0
    
  }
  else if(demdata[i,3]=="f"){
    GENDER[i] = 1
  }
}
#-----------------------------------------------------------------------------------------------------------
# Defining the segments for customers for the benchmark method RFM-based Segmnentation
#-----------------------------------------------------------------------------------------------------------

segment_names <- c("BestCustomers", "Loyal Customers", "Potential Loyalist",
                   "New Customers", "Promising Customers", "Need Attention", "About To Sleep",
                   "At Risk", "Can not be lost" ,"Lost Customer")
recency_lower <-   c(4, 3, 3, 4, 3, 2, 2, 1, 1, 1)
recency_upper <-   c(5, 5, 5, 5, 4, 3, 3, 2, 2, 1)
frequency_lower <- c(4, 3, 1, 1, 1, 2, 1, 2, 4, 1)
frequency_upper <- c(5, 5, 3, 2 ,1, 3, 2, 5, 5, 1)
monetary_lower <-  c(4, 3, 1, 1, 1, 2, 1, 2, 4, 1)
monetary_upper <-  c(5, 4, 3, 2, 1, 3, 2, 5, 5, 1)
#-----------------------------------------------------------------------------------------------------------
# Importing the transactional data from Big Query table trans_table_new2
#-----------------------------------------------------------------------------------------------------------
sql1 = "SELECT cast(lkuid as string) as customer_id, cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.trans_table_new2`
Order by lkuid "
order_data = query_exec(sql1, project, use_legacy_sql = FALSE, max_pages = Inf)
rfm_data_orders = data.frame(order_data[,1],as.Date(order_data[,4]), order_data[,3])
colnames(rfm_data_orders) = c("customer_id","order_date","revenue")
#-----------------------------------------------------------------------------------------------------------
# RFM-based Segmentation 
#-----------------------------------------------------------------------------------------------------------
rfm_order_result = rfm_table_order(rfm_data_orders,customer_id,order_date,revenue,analysis_date)
RFMsegments_main = rfm_segment(rfm_order_result, segment_names, recency_lower, recency_upper,frequency_lower, frequency_upper, monetary_lower, monetary_upper)
#Percentages per segment
table = RFMsegments_main %>% count(segment) %>%arrange(desc(n)) %>%rename(Segment = segment, Count = n)
total = sum(table[,2])
percentages = NULL
for(i in 1:nrow(table[,2])){
  percentages[i] = table[i,2]/total*100
}
table
percentages
# RFM graphs
rfm_heatmap(rfm_order_result,plot_title = "RFM Heat Map",plot_title_justify = 0.5, xaxis_title = "Frequency",yaxis_title = "Recency", legend_title = "Mean Monetary Value",brewer_n = 5, brewer_name = "Oranges")
rfm_bar_chart(rfm_order_result,bar_color = "gray23",xaxis_title = "Monetary Score", sec_xaxis_title = "Frequency Score",yaxis_title = " ", sec_yaxis_title = "Recency Score")
rfm_rm_plot(rfm_order_result,point_color = "gray23", xaxis_title = "Monetary",yaxis_title = "Recency", plot_title = "Recency vs Monetary")
rfm_fm_plot(rfm_order_result,point_color = "gray23", xaxis_title = "Monetary",yaxis_title = "Frequency", plot_title = "Frequency vs Monetary")
rfm_rf_plot(rfm_order_result,point_color = "gray23", xaxis_title = "Frequency",yaxis_title = "Recency", plot_title = "Recency vs Frequency")


#RFM-based segmentation on holidays only
#December 2017
sql1 = "SELECT cast(lkuid as string) as customer_id , cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.H_December_17`"
#December 2018
sql1 = "SELECT cast(lkuid as string) as customer_id , cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.H_December_18`"
#Sinterklaas 2017
sql1 = "SELECT cast(lkuid as string) as customer_id , cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.H_Sinterklass_17`"
#Sinterklaas 2018
sql1 = "SELECT cast(lkuid as string) as customer_id , cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.H_Sinterklass_18`"
#Easter 2017
sql1 = "SELECT cast(lkuid as string) as customer_id , cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.H_Easter_17`"
#Easter 2018
sql1 = "SELECT cast(lkuid as string) as customer_id , cast(trans_nr as string) as TransNo, trans_amount as revenue ,cast(date(betaal_datum)as string)as order_date
FROM `edc-gall-en-gall.data.H_Easter_18`"

#for each sql1 following code should be repeated
order_data1 = query_exec(sql1, project, use_legacy_sql = FALSE, max_pages = Inf)
rfm_data_orders1 = data.frame(order_data1[,1],as.Date(order_data1[,4]), order_data1[,3])
colnames(rfm_data_orders1) = c("customer_id","order_date","revenue")
rfm_Holidays = rfm_table_order(rfm_data_orders1,customer_id,order_date,revenue,analysis_date)
RFMsegments = rfm_segment(rfm_Holidays , segment_names, recency_lower, recency_upper,frequency_lower, frequency_upper, monetary_lower, monetary_upper)
#table with amount of customers per segment
table1 = RFMsegments %>%count(segment) %>%arrange(desc(n)) %>%rename(Segment = segment, Count = n)
total = sum(table1[,2])
percentages = NULL
for(i in 1:nrow(table1[,2])){
  percentages[i] = table1[i,2]/total*100
}
table1
percentages


#-----------------------------------------------------------------------------------------------------------
# Removing customers from Lost Customer segment
#-----------------------------------------------------------------------------------------------------------
alldata = data.frame(RFMsegments_main, allow_analysis, GENDER, age_category, opt_in,loyal_time)

#List of exceptionally best customers with RFM score 555
Excep_best = RFMsegments_main[RFMsegments_main[,3]=='555',1]
#List of customers from BEst Customer segment
Best_cust_data_index = as.numeric(rownames(alldata[alldata[,2] == 'BestCustomers',]))
Best_segment_id = RFMsegments_main[Best_cust_data_index,1]

lost_cust_data_index = as.numeric(rownames(alldata[alldata[,2] == 'Lost Customer',])) #indices of customers from Lost customer group
alldata = alldata[-lost_cust_data_index,] #new data without inactive customers
customerid = alldata[,1]
Frequency = alldata[,4] 
Recency = alldata[,5] 
Monetary = alldata[,6] 
Allow_analysis = alldata[,11]
Female = alldata[,12]
Age = alldata[,13]
Opt_in_com = alldata[,14]
Loyal_time = alldata[,15]
#scacling the R,F,M variables
Frequency_s = scale(Frequency) 
Recency_s = scale(Recency) 
Monetary_s = scale(Monetary) 
