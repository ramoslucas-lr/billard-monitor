import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


data = pd.read_csv('../data/employee_file.csv')

features = data[data.columns[1:6]]
print(features.head())

scaled_features = MinMaxScaler().fit_transform(features[data.columns[1:6]])

pca = PCA(n_components=2).fit(scaled_features)

features_2d = pca.transform(scaled_features)
print(features_2d[0:10])

# plt.scatter(features_2d[:,0],features_2d[:,1])
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.title('Data BGR+HSV')
# plt.show()

# wcss = []
#
# for i in range(1, 15):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(features.values)
#
#     wcss.append(kmeans.inertia_)

# plt.plot(range(1, 15), wcss)
# plt.title('WCSS by Clusters HSV')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# print('aqui')

agg_model = AgglomerativeClustering(n_clusters=11)
agg_clusters = agg_model.fit_predict(features.values)

# def plot_clusters(samples, clusters):
#     col_dic = {0: '#c241c4', 1: '#e0dd72', 2: '#000dff', 3: '#d4db00', 4: '#804f00', 5: '#000dff', 6: '#d10000', 7: '#000000', 8: '#478500', 9: '#b0b0b0', 10: '#0035e3'}
#     mrk_dic = {0: '.', 1: '.', 2: '.', 3: '.', 4: '.', 5: '.', 6: '.', 7: '.', 8: '.', 9: '.', 10: '.'}
#     colors = [col_dic[x] for x in clusters]
#     markers = [mrk_dic[x] for x in clusters]
#     for sample in range(len(clusters)):
#         print(sample)
#         plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.title('Assignments')
#     plt.show()
#
# plot_clusters(features_2d, agg_clusters)

data['cluster'] = agg_clusters

balls_features = ['b_int', 'g_int', 'r_int', 'h_int', 's_int', 'v_int']
balls_label = 'cluster'

ball_classes = ['pink', 'purple', 'trash', 'yellow', 'brown', 'trash', 'red', 'black', 'green', 'gray', 'blue']
data_X, data_y = data[balls_features].values, data[balls_label].values

x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.30, random_state=0, stratify=data_y)

print('Training Set: %d, Test Set: %d \n' % (x_train.shape[0], x_test.shape[0]))

feature_columns = [0,1,2,3,4,5]
feature_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('preprocess', feature_transformer, feature_columns)])

# Create training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestClassifier(n_estimators=100))])

# fit the pipeline to train a linear regression model on the training set
multi_model = pipeline.fit(x_train, y_train)
print (multi_model)

# Get predictions from test data
data_predictions = multi_model.predict(x_test)
data_prob = multi_model.predict_proba(x_test)

# Overall metrics
print("Overall Accuracy:",accuracy_score(y_test, data_predictions))
print("Overall Precision:",precision_score(y_test, data_predictions, average='macro'))
print("Overall Recall:",recall_score(y_test, data_predictions, average='macro'))
print('Average AUC:', roc_auc_score(y_test,data_prob, multi_class='ovr'))

# Print the confusion matrix
mcm = confusion_matrix(y_test, data_predictions)
print(mcm)

# Save the model as a pickle file
filename = 'balls_model.pkl'
joblib.dump(multi_model, filename)

# Load the model from the file
multi_model = joblib.load(filename)

# The model accepts an array of feature arrays (so you can predict the classes of multiple penguin observations in a single call)
# We'll create an array with a single array of features, representing one penguin
x_new = np.array([[123.705,97.5625,82.045,87.9225,129.975,123.705]])
print ('New sample: {}'.format(x_new[0]))

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one penguin, so our prediction is the first one in the resulting array.
penguin_pred = multi_model.predict(x_new)[0]
print('Predicted class is', ball_classes[penguin_pred])