# Predicting Heart Disease

For machine learning project, I chose to build a binary classifier to predict [heart disease](https://www.kaggle.com/danimal/heartdiseaseensembleclassifier). Because this model has healthcare applications, the emphasis is on recall rather than accuracy or precision when evaluating model performance. I explore a variety of relatively simple classifiers (read: no neural networks)––Support Vector Machines, Decision Trees and Random Forests, AdaBoost and XGBoost, KNN, naive bayes, gradientboosting––and fine tune each to upwards of 85% recall on test data. My final model, an ensemble model of AdaBoost hypertunning with GuassianNB, combines some of the best models to achieve 95% recall on test data.

### EDA and Preprocessing

Of the raw data of 303 patients, no null values ;

The original labels, ranging from 0, no heart disease, to 4, the most advanced stage of heart disease, were redesigned to range from `0`, no heart disease, and the original values of 1, 2, 3, and 4 were squished into a single category, `1`, "presence of heart disease."

Features were standardized and then analyzed for correlation among each other using a Pearson correlation heatmap. Simultaneously, a random forest was trained to evaluate feature importance, with `thalach`, `cp`, `thal`, and `ca` taking the top four spots.

![Pearson Correlation Heatmap](output/heatmap.png)

none of my features are so highly correlated with each other. I keep the next seven most significant features, none of which are egregiously correlated with each other, for my final set of features.

![Histograms for final, preprocessed predictors](output/heatmap.png)

### Model Training

Models were evaluated primarily for their recall, given the healthcare setting, while still trying to maintain modestly high accuracy and precision. Nine models were trained.
The model predicted heart disease––which featured ~0.96 recall while still retaining a relative maximum in accuracy ~0.95. Confusion matrices for before and after threshold installed below.

![Confusion matrix](output/adaboost_using_gNB_cmatrix.png)
### Results

| Model | Final Test Accuracy |
|-|-|-|
| AdaBoost (Base = GuassianNB) | **95%** |
| Linear SVC | 89.5% |
| GuassianNB | 88% |
| BernoulliNB | 88% |
| Decision Tree | 85.5% |
| Random Forest | 88% |
| AdaBoost (Base = Decision Tree) | 95% |
| GradientBoost | 91.8% |
| XGBoost | 93.4% |
| KNN | 86% |

### Conclusions

By (1) list-wise removing missing data; (2) converting the classification problem to a binary one and standardizing features; and (3) manually setting the probability threshold for disease detection to 0.35, I was able to achieve **0.95 recall** and **0.89 accuracy** from my best model, a soft voting ensemble classifier made up of a linear SVM, logistic regression classifier, and an AdaBoost classifier.

## Libraries

Standard packages for data analysis and visualization are required–[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/)––as well as [pydotplus](https://pypi.org/project/pydotplus/) and [Graphviz](https://graphviz.org/) to make a visualization of a decision tree, [scikit-learn](https://scikit-learn.org/stable/index.html) for a number of classifiers, and [XGBoost](https://xgboost.readthedocs.io/en/latest/).


## Contributing

Due to the nature of the project, this project is not open to contributions. If, however, after looking at the project you'd like to give advice to someone new to the field and eager to learn, please reach out to me at [rajarshi921@gmail.com]

## Author

**Rajarshi SinhaRoy** <br/>



## Acknowledgments
Thanks to [Kaggle](https://www.kaggle.com) for access to data found in [Heart Disease Ensemble Classifier](https://www.kaggle.com/danimal/heartdiseaseensembleclassifier), and particular thanks to [Nathan S. Robinson](https://www.kaggle.com/iamkon/ml-models-performance-on-risk-prediction) for his work on the same dataset: it was beautifully organized, instructive, and a constant source of clarity and inspiration.
