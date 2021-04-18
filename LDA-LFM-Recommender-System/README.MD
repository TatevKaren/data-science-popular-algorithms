# LDA-LFM Recommender System 
**This work is based on a paper preseted and published at 36th ACM/SIGAPP Symposium on Applied Computing (SAC 2021)**
<br>
<p> <img href ="https://medium.com/mlearning-ai/using-customer-and-product-features-in-recommender-systems-2734258873cf" src="https://miro.medium.com/max/1120/1*OAHUmDf7SkFFeYF6Bsa5EQ.png?raw=true"
  alt="Multivariate Linear Regression plot"
  width="900" height="300"> </p>
  <br>
  <br>
  
  - Check Out <a href="https://medium.com/mlearning-ai/using-customer-and-product-features-in-recommender-systems-2734258873cf">Medium Article </a> About This Work
  - Check Out <a href="https://www.youtube.com/watch?v=i_j59gQTU6w&t=42s&ab_channel=TatevKaren">Conference Presentation Youtube Video</a>
  - Check Out <a href="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/ConferencePresentation_SAC2021.pdf"> Conference Presentation Slides</a>
  
  <br>
  <br>
  
  
# Amazon Review Big Data
In this expeiment, we used collection of datasets provided corresponding to the 23 product categories supplied by **Amazon.com Review Data**. This data without duplicates was prepared by <a href = "https://jmcauley.ucsd.edu/data/amazon/">Julian McAuley</a>. It consists of **142.8 million** product reviews and a metadata for 9.4 million products, spanning a period of 18 years, from May 1996 to July 2014 
<br>
  <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/10_30_rating_review_figure.png?raw=true?raw=true"
  width="600" height="400"> </p>
This figure visualizes the percentage of ratings and reviews per item. Brown (dark grey for black and white print) bars represent the percentage of items with less than 10 ratings per category. Green (light grey for black and white print) bars rep- resent the percentage of items having on average more than 30 words per product category.
  <br>
    <br>
  <br>
 
 
# Python Code
To reproduce the results obtained in this analysis you can execute following Python coded step-by-step. Note that many of the libraries such as **Tensorflow** need to be preinstalled on your computer. You can use the following syntax in your command line to install the desired Python package:

```Python
# Check the versions of Pip3 and Python
python3 --version
pip3 --version

# Install necessary python packages.
sudo apt-get install python3-pip python3-dev -y

# Upgrade pipo this is 
pip3 install --upgrade pip

# Install tensorflow (cpu only)
pip3 install tensorflow
```

<br>

- Step 1: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Required%20Libraries.py"> Importing Required Python Libraries </a>
- Step 2: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Data%20Preparation.py"> Data Preparation</a>
- Step 3: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Data%20Transformation.py"> Data Transformation</a>
- Step 4: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Data%20Transformation%20NLP.py"> Data Transformation Natural Language Preprocessing (NLP)</a>
- Step 5: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Sparse%20Matrix%20Generation.py"> Data Transformation (Sparse Matrix Generation)</a>
- Step 6: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Offset%20Recommender%20System.py"> Offset Recommender Algorithm</a>
- Step 7: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Baseline%20Recommender%20System.py"> Baseline Recommender Algorithm</a>
- Step 8: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Standard%20Matrix%20Factorization%20Recommender%20System.py"> Standard Matrix Factorization Recommender Algorithm </a>
- Step 9: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20LDAFirst%20Recommender%20System.py"> LDAFirst Recommender Algorithm</a>
- Step 10: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Proposed%20LDA-LFM%20algorithm.py"> LDA-LFM Recommender Algorithm</a>

 
  <br>
    <br>
  <br> 
 
 # Model Performance Visualizatiion 

 <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/Results%201.png?raw=true"
  width="650" height="400"> </p>
  <br>
    <br>
  <br>
  <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/Picture%202.png?raw=true"
  width="650" height="450"> </p>
  <br>

  <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/Picture%203.png?raw=true"
  width="650" height="350"> </p>
  <br>
  <br>
  <br>

