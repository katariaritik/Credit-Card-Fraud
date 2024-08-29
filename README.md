# Credit-Card-Fraud
The dataset consist of transactions made by cardholders through various credit cards. Data set contains only numerical(continuous) input variable which are result of Principle Component Analysis (PCA) feature selection transformation, resulting in 28 principal components and total 30 input features are utilized in this data.
Features V1, V2, ... V28 are the principal components obtained with PCA
The only features which have not been transformed with PCA are Time and Amount. Feature Time contains the seconds elapsed between each transaction and the first 5 transaction in the dataset. The feature Amount is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. 
Concept Used- Isolation Forest and Local Factor outline Algorithm.

Isolation Forest is an unsupervised machine learning technique for detecting anomalies or outliers in a dataset.
Use Cases: It's used in various domains like fraud detection, network security, and quality control in manufacturing. It utilizes decision trees to isolate anomalies by randomly partitioning the data, aiming        to find anomalies with fewer splits.

LOF (Local outlier factor Algorithm) considers both the density of the data point and the density of the data point’s immediate surroundings. One important characteristic of the LOF approach is that the outliers score takes into account the data point’s relative density.

Observation- 
Isolation Forest detected 73 errors versus Local Outlier Factor detecting 97 errors. Isolation Forest has a 99.74% more accurate than LOF of 99.65%.
When comparing error precision & recall for  2 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is around 27 % versus LOF detection rate of just 2 %.
So overall Isolation Forest Method performed much better in determining the fraud cases.








