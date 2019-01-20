# DatingRecommender

This is EECS 6893 Big Data Analytics Project: 
Romantic partner recommendation based on Speed Dating Experiment.
Team member: Yijia Xu, Shih-Ying Jeng, Min Fu

#### Find the project
- [YouTube Video](https://youtu.be/HtjJiySTdIE)


### Dataset
- [Speed Dating Experiment](https://www.kaggle.com/annavictoria/speed-dating-experiment)
- The dataset is publicly available on Kaggle

### Abstract
Most dating recommenders are based on online dating sites that make use of virtual user profiles portrayed by the users themselves. While speed dating collects real-life face-to-face dat- ing interactions among people, the data is rarely applied in dating recommender in any form. This report explores the research gap and presents a novel dating recommender which combines speed dating study with dating recommender system. Speed dating data suggests that user profile portrayed by themselves may not accurately reflect their likeability as a potential romantic partner. We introduce an approach of extracting objective evaluation based on the objective ratings given by the dating partners in the speed dating events to construct an objective profile library. Due to the low match rate in speed dating, there is a high class imbalance on the match label in the speed dating dataset. The project applies SMOTE oversampling to mitigate the class imbalance issue. A random forest regression-based reciprocal recommender is presented. Experiment results confirm the effectiveness of the proposed approach. The combination of the objective profile library, SMOTE oversampling, and a random forest regression-based reciprocal recommender achieves a match prediction accuracy of 92.19%, outperforming existing benchmark algorithms.

### Preprocessing
- type: resampling
- purpose: mitigate class imbalance
- method: SMOTE oversampling. Generate synthetic minority class examples to make the class balanced.

### Recommender Model
- type: reciprocal recommender
- underalying model: regressor/classifier
- input: person (feature vector)
- output: recommended people (list of feature vectors)

### Regressor Model
- type: logistic regression/neural network/xgboost/random forest
- input: pair of people (feature vector pair)
- output: reciprocal score (probability of match on the pair)

### Package Structure
```
DatingRecommender
├── Exploratary Data Analysis
│   ├── AttributesAnalysis.py
│   ├── DataPreprocessing.ipynb
│   ├── CorrelationAnalysis.ipynb
│   └── profileComparision.ipynb
├── Model
│   ├── data_prep.ipynb
│   ├── Model-O.ipynb
│   ├── Model-S.ipynb
│   ├── regressor-spark.ipynb
│   ├── Recommender.ipynb
│   └── nn.py
└── Visualization
    ├── wave1.ipynb
    ├── original.ipynb
    ├── new.ipynb
    ├── datadealing.py
    ├── spark_wave1.py 
    ├── spark_original.py
    ├── spark_new.py
    ├── graph_all_wave_1.html
    ├── graph_all_wave_ori.html
    └── graph_all_wave_new.html
```

#### Language
- Python
- javascript
- html5

#### Python packages
- sklearn
- tensorflow
- imblearn
- numpy
- pandas
- matplotlib
- seaborn
- imblearn

### Exploratary Data Analysis

#### AttributesAnalysis.py
- description: Draw the radar chart to analyze the gender difference in choosing romantic partners

#### DataPreprocessing.ipynb
- description: Preprocess the data to calculate statitical attribute scores.

#### CorrelationAnalysis.ipynb
- description: Calculate the correlation matrix of subjective ratings, objective ratings, decision result and match result.

#### profileComparision.ipynb
- description: Calculate and compare subjective and objective scores.


### Model

#### data_prep.ipynb
- description: prepare the data to be used by the regressor model and the recommender model
- input: original data csv file (renamed to data.csv)
- output: intermediate data csv files

#### nn.py
- note: Python 3.7 is not supported as Tensorflow supports up to Python 3.6
- description: the neural network regressor model
- Usage: python3.6 nn.py

#### Model-O.ipynb
- description: regressor models using the objective profile library

#### Mode-S.ipynb
- description: regressor models using the subjective profile library

#### Recommender.ipynb
- description: recommender using the objective profile library

#### regressor-spark.ipynb
- description: pyspark version of the regressor models using the objective profile library


### Visualization
- data processing 
- spark graph
- d3.js

#### datadealing.py
- description: prepare original data for visualization in Python

#### wave1.ipynb
- description: data prepocessing of people in wave 1

#### original.ipynb
- description: data prepocessing of people in all waves without recommender 

#### new.ipynb
- description: data prepocessing of all waves with recommendation for people in wave 1

#### spark_wave1.py
- description: make a graph using graphframes and calculate pagerank for people in wave 1 in Spark

#### spark_original.ipynb
- description: make a graph using graphframes for people in all waves without recommender in Spark

#### spark_new.ipynb
- description: make a graph using graphframes for people in all waves with recommendation for people in wave 1 in Spark

#### graph_all_wave_1.html
- description: visualization of people in wave 1 without recommender in html webpage

#### graph_all_wave_original.html
- description: visualization of people in all waves without recommender in html webpage

#### graph_all_wave_new.html
- description: visualization of people in all waves with recommendation for the two isolated person in wave 1 in html webpage


