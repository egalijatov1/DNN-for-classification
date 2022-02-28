# Million Song Dataset classification
Simple classification of songs into decade categories using Deep Neural Networks in Tensorflow and Keras.

## Data
For this project a subset of [Milion Song Dataset](http://millionsongdataset.com/) is used. The data set contains features extracted from a variety of songs released in four different decades  (1970s, 1980s, 1990s and 2000s). The training set consists of 40,000 examples (i.e., 10,000 songs from each decade). Each data sample is represented by a 90-dimensional feature vector. The test set contains the same extracted features from a different set of 4,000 songs (1,000 examples from each class). Class labels are indicated by strings from the following
set: {70s', 80s', 90s', 00s'}.

#### Data exploration
To get familiar with the dataset feature distributions and their correlations were explored. Since there are 90 features, only some of them were analysed individualy. 

#### Data preprocessing
Experiments showed that best preprocessing technique for features is standardization using [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) from [sklearn](https://scikit-learn.org/stable/) library. For label [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) was used.

## NN Architectures
For this problem only linear layers were used. Many experiments were conducted with different architectures differing in number of hidden layers and number of hidden units. Additionaly, the batch sizes was changed. 

The list of conducted experiments: 
<br>
<img src="https://user-images.githubusercontent.com/88715320/156023199-08dfab17-8646-4678-bcf6-eaf4cdffd473.png" width="500">


In above table all computations were done with stochastic SGD and constant parameters. After this step, four model architectures with best performance were chosen for further parameter tunning (models 1, 6, 7 and 9).

Additional experiments were done, with changing the following:
- optimizer (SGD, ADAM)
- learning rate
- momentum (when applicable)
- learning rate scheduling
- adding dropout layers
- adding regularization

Moreover, early stopping was used to determine the number of epochs and avoid overfitting.

In the project file, the best performing model architecure and parameters are presented.
