# Semi-Supervised-Learning-on-Fashion-MNIST-dataset

## Contributors
- **Avirup Das**
- **Ayush Thada**

## Introduction

Clustering is usually used for problems related to unsupervised learning but we will use it as a pre-processing tool for semi-supervised learning. If we only have a few labels, we could perform clustering and propagate the labels to all the instances (or to the closest instances decided by percentile) in the same cluster. This technique can greatly increase the number of labels available for a subsequent supervised learning algorithm, and thus improve its performance.

## Data Set
`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. `Fashion-MNIST` was intended to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## Methodology

We will use logistic regression and a 7 layer deep neural network to classify the Fashion MNIST dataset. For each of these models we would first train the model with the whole data set (60000 instances) and test it for 10000 instances. The accuracy will be our baseline and we would try to improve upon that.<br>

|        Model        | Baseline Accuracy |
|:-------------------:|:-----------------:|
| Logistic Regression |       84.1%       |
|    Neural Network   |       89.68%      |


The Fashion MNIST dataset contains images of dimension $28\times28$, so as a pre-processing step for Logistic regression, we flatten the image matrix into a vector. We have also pickled our models so they can be re-used without retraining as we observed that the training time was very high since we were also experimenting with different values of the number of clusters. These files will be provided along with the code.

First we have created a pipeline that will cluster the training set into 100, 200 and 300 clusters and replace the images with their distances to these clusters, then apply a Logistic Regression model. 

| Cluster | Accuracy |
|:-------:|:--------:|
|   100   |  82.62%  |
|   200   |  83.89%  |
|   300   |  84.56%  |

Then we took random n-labelled instances (for n=500, 1000, 2000) and check how our models perform in terms of accuracy.

| Cluster | Logistic Regression | Neural Network |
|:-------:|:-------------------:|:--------------:|
|   500   |        78.52%       |     73.51%     |
|   1000  |        79.24%       |       76%      |
|   2000  |        80.89%       |     78.53%     |

We can see that the neural network does not perform very well compared to Logistic Regression in this case since we are using a very small amount of data as training set. On the other hand, the neural network takes much lesser time to get trained. We also see that the models perform better for large cluster sizes suggesting that for further experimentation we should work with large cluster sizes.<br>

Next, we cluster the instances into 2000 clusters and use the centroids to train our model.

| Cluster | Logistic Regression | Neural Network |
|:-------:|:-------------------:|:--------------:|
|   500   |          -          |      75.5%     |
|   1000  |          -          |     76.36%     |
|   2000  |        81.46%       |     80.24%     |

Again we see that the accuracy increases when the number of clusters is increased. Now we propagate the labels of these representative points (centroids) to all the instances under the same cluster and run our models.

| Cluster | Logistic Regression | Neural Network |
|:-------:|:-------------------:|:--------------:|
|   500   |          -          |     76.49%     |
|   1000  |          -          |     77.80%     |
|   2000  |        81.21%       |     79.90%     |


We do not see a significant difference between the results that is because when we have propagated the labels of the centroids to all instances, we have also included outliers or instances which are ambiguous in terms of which cluster they fall in. So let us propagate the labels to the instances which are close (25 percentile) to the cluster centroids an train our models again.

| Cluster | Logistic Regression | Neural Network |
|:-------:|:-------------------:|:--------------:|
|   500   |          -          |     75.98%     |
|   1000  |          -          |     77.34%     |
|   2000  |        80.44%       |     79.42%     |


Finally for 2000 clusters we try to find the optimum distance from the centroid (in terms of percentile) so as to achieve maximum accuracy.

| Percentile Distance | NN Accuracy |
|:-------------------:|:-----------:|
|          20         |     ~10%    |
|          25         |    79.42%   |
|          30         |    80.12%   |
|          50         |    80.12%   |
|          75         |    79.60%   |


We can see that the optimum distance is around 30-50th percentile after which the accuracy drops (from both ends of the range). It is to be noted that although we could not improve upon our baseline accuracy using semi-supervised learning techniques, but in a situation where we get completely unlabelled data, these techniques come handy for boosting the accuracy of our models after we have labelled a small but somewhat significant portion of the data manually.<br><br>
**Note that the accuracy scores for neural network may vary due to random initialisations or GPU configuration. Also, we haven't included the output of the neural networks in the notebook to avoid making it unnecessarily long.** <br><br>

## Details Of Pickle Files:<br>
To load the joblib file 
```
clf= load('cluster_nn_200.joblib')
clf.transform(X_train)
```

To load trained neural network
```
model_1 = tf.keras.models.load_model('nn_full_propagated_2000')
model_1.summary()
```
- log_reg_orig.joblib : Logistic Regression on original dataset
- log_reg_kmeans_0.joblib : Logistic Regression on distances after doing kmeans for 100 clusters
- log_reg_kmeans_1.joblib : Logistic Regression on distances after doing kmeans for 200 clusters
- log_reg_kmeans_2.joblib : Logistic Regression on distances after doing kmeans for 300 clusters
- log_reg_few_label_0.joblib : Logistic Regression trained on 500 random instances
- log_reg_few_label_1.joblib : Logistic Regression trained on 1000 random instances
- log_reg_few_label_2.joblib : Logistic Regression trained on 2000 random instances
- kmeans_2000.joblib : Kmeans on original training data with 2000 clusters
- log_reg_centroids.joblib : Logistic Regression trained using the centroids of kmeans_2000.joblib
- log_reg_propagated.joblib : Logistic Regression trained using the original data after propagating the labels of the centroids to entire dataset
- log_reg_partially_propagated.joblib : Logistic Regression trained using the original data after propagating the labels of the centroids to those instances which fall under 25-percentile distance of the respective centroids
- nn_original : Neural Network trained using original dataset
- nn_labelled_500 : Neural Network trained on 500 random instances
- nn_labelled_1000 : Neural Network trained on 1000 random instances
- nn_labelled_2000 : Neural Network trained on 2000 random instances
- cluster_nn_500.joblib : Cluster with 500 centroids for Neural Network
- cluster_nn_1000.joblib : Cluster with 1000 centroids for Neural Network
- cluster_nn_2000.joblib : Cluster with 2000 centroids for Neural Network
- nn_centroid_cluster_500 : Neural Network trained using the centroids of cluster_nn_500.joblib
- nn_centroid_cluster_1000 : Neural Network trained using the centroids of cluster_nn_1000.joblib
- nn_centroid_cluster_2000 : Neural Network trained using the centroids of cluster_nn_2000.joblib
- nn_full_propagated_500 : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_500.joblib to entire dataset
- nn_full_propagated_1000 : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_1000.joblib to entire dataset
- nn_full_propagated_2000 : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib to entire dataset
- nn_partially_propagated_500 : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_500.joblib, to those instances which fall under 25-percentile distance of the respective centroids
- nn_partially_propagated_1000 : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_1000.joblib, to those instances which fall under 25-percentile distance of the respective centroids
- nn_partially_propagated_2000 : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib, to those instances which fall under 25-percentile distance of the respective centroids
- nn_partially_propagated_2000-clusters_20-percentile : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib, to those instances which fall under 20-percentile distance of the respective centroids
- nn_partially_propagated_2000-clusters_25-percentile : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib, to those instances which fall under 25-percentile distance of the respective centroids
- nn_partially_propagated_2000-clusters_30-percentile : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib, to those instances which fall under 30-percentile distance of the respective centroids
- nn_partially_propagated_2000-clusters_50-percentile : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib, to those instances which fall under 50-percentile distance of the respective centroids
- nn_partially_propagated_2000-clusters_75-percentile : Neural Network trained using the original data after propagating the labels of the centroids of cluster_nn_2000.joblib, to those instances which fall under 75-percentile distance of the respective centroids

Link to output folder: https://mega.nz/folder/QHhyADyK#1rk56JFrTMZ-RXJpjzXKAg <br>
