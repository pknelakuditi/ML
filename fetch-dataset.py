from sklearn.datasets.mldata import fetch_mldata
import tempfile
test_data_home = tempfile.mkdtemp()
breast=fetch_mldata('datasets-UCI breast-w', transpose_data=True, data_home=test_data_home)
#breast=fetch_mldata('housing_scale', data_home=test_data_home)
print breast.data.shape
n_samples, n_features = breast.data.shape

print n_samples,n_features
print breast.data.shape
#print breast.data[0]
print breast.target.shape



