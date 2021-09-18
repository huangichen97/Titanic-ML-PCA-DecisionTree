"""
File: titanic_pandas_pca.py
Name: Ethan H
---------------------------
This file shows how to use pandas and sklearn
packages to build a machine learning project
from scratch by their high order abstraction.
The steps of this project are:
1) Data pre-processing by pandas
2) PCA to extract components
3) Learning by sklearn
4) Test on D_test
"""

import pandas as pd
from sklearn import linear_model, preprocessing, decomposition


# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'                    # Training set filename
TEST_FILE = 'titanic_data/test.csv'                      # Test set filename

# Global variable
nan_cache = {}                                           # Cache for test set missing data


def main():

	# Data cleaning
	data = data_preprocess(TRAIN_FILE, mode='Train')

	# Extract true labels
	y = data.Survived

	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
	# Extract feature_names first!
	feature_names = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
	x_train = data[feature_names]

	# Standardization
	standardizer = preprocessing.StandardScaler()
	x_train = standardizer.fit_transform(x_train)
	# 0 mean

	# PCA
	pca = decomposition.PCA(n_components=5)
	x_train_reduce = pca.fit_transform(x_train)

	poly_feature_extractor = preprocessing.PolynomialFeatures(degree=2)
	x_train_poly = poly_feature_extractor.fit_transform(x_train_reduce)

	# Degree 2 Polynomial
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(x_train_poly, y)
	acc = classifier.score(x_train_poly, y)
	print('Degree 2 Training Acc:', acc)

	# Test
	test_data = data_preprocess(TEST_FILE, mode='Test')
	test_data = test_data[feature_names]
	# standardize
	test_data = standardizer.transform(test_data)
	# pca
	test_data = pca.transform(test_data)
	test_poly = poly_feature_extractor.transform(test_data)

	predictions = classifier.predict(test_poly)
	test(predictions, "Ethan_pandas_PCA__components_submission_degree2.csv")


def data_preprocess(filename, mode='Train'):
	"""
	: param filename: str, the csv file to be read into by pd
	: param mode: str, the indicator of training mode or testing mode
	-----------------------------------------------
	This function reads in data by pd, changing string data to float, 
	and finally tackling missing data showing as NaN on pandas
	"""

	# Read in data as a column based DataFrame
	data = pd.read_csv(filename)
	if mode == 'Train':
		# Cleaning the missing data in Fare column by replacing NaN with its median
		fare_median = data['Fare'].dropna().median()
		data['Fare'] = data['Fare'].fillna(fare_median)

		# Cleaning the missing data in Age column by replacing NaN with its median
		age_median = data['Age'].dropna().median()
		data['Age'] = data['Age'].fillna(age_median)

		# Cache some data for test set
		nan_cache['Age'] = age_median
		nan_cache['Fare'] = fare_median

	else:
		# Fill in the NaN cells by the values in nan_cache to make it consistent
		data['Fare'] = data['Fare'].fillna(nan_cache['Fare'])
		data['Age'] = data['Age'].fillna(nan_cache['Age'])

	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data['Embarked'] = data['Embarked'].fillna('S')
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	return data
	

def test(predictions, filename):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n==========================')
	print('Writing predictions to ...')
	print(filename)
	with open(filename, 'w') as out:
		out.write('PassengerId,Survived\n')
		start_id = 892
		for ans in predictions:
			out.write(str(start_id)+','+str(ans)+'\n')
			start_id += 1
	print('\n==========================')


if __name__ == '__main__':
	main()
