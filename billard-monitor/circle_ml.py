import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

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
model = KMeans(n_clusters=7, init='k-means++', n_init=100, max_iter=1000)
# print('ou aqui')
km_clusters = model.fit_predict(features.values)
# print('ou aqui')
#
# def plot_clusters(samples, clusters):
#     col_dic = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'yellow', 5: 'magenta', 6: 'cyan'}
#     mrk_dic = {0: '*', 1: 'x', 2: '+', 3: '*', 4: 'x', 5: '+', 6: '.'}
#     colors = [col_dic[x] for x in clusters]
#     markers = [mrk_dic[x] for x in clusters]
#     for sample in range(len(clusters)):
#         print(sample)
#         plt.scatter(samples[sample][0], samples[sample][1], color=colors[sample], marker=markers[sample], s=100)
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.title('Assignments')
#     plt.show()
#
#
# plot_clusters(features_2d, km_clusters)


agg_model = AgglomerativeClustering(n_clusters=7)
agg_clusters = agg_model.fit_predict(features.values)

# def plot_clusters(samples, clusters):
#     col_dic = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'yellow', 5: 'magenta', 6: 'cyan'}
#     mrk_dic = {0: '*', 1: 'x', 2: '+', 3: '*', 4: 'x', 5: '+', 6: '.'}
#     colors = [col_dic[x] for x in clusters]
#     markers = [mrk_dic[x] for x in clusters]
#     for sample in range(len(clusters)):
#         print(sample)
#         plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.title('Assignments')
#     plt.show()

plot_clusters(features_2d, agg_clusters)