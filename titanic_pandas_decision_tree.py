"""
File: titanic_pandas_decision_tree.py
Name: Ethan H
---------------------------
This file shows how to use pandas and sklearn
packages to build a decision tree, which enables
students to see the most important features 
on Webgraphviz.com
"""

import pandas as pd
from sklearn import tree

# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'                    # Training set filename


def main():

	# Data cleaning
	data = data_preprocess(TRAIN_FILE, mode='Train')

	# Extract true labels
	y = data.Survived

	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
	feature_names = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
	x_train = data[feature_names]

	# Construct Tree
	d_tree = tree.DecisionTreeClassifier()		# 可設定 max_depth = x
	d_tree.fit(x_train, y)
	print(d_tree.score(x_train, y))

	tree.export_graphviz(d_tree, feature_names=feature_names, out_file='tree_fig')
	

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

	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data['Embarked'] = data['Embarked'].fillna('S')
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	return data
	


if __name__ == '__main__':
	main()
