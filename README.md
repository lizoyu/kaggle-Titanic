# Kaggle competition: Titanic
This is for the kaggle competition **Titanic: Machine Learning from Disaster**.

## Goals
- analyze survival based on features
- predict passenger survival

## To-do lists
- feature engineering
	- family size
	- name(title, surfix)
	- discretize numeric variables if necessary
		- age, fare
	- cabin(maybe too sparse)
	- is Pclass treated as ordinal?
	- categorical correlation
- data imputation
	- MICE
- classification
	- dead-biased (higher survival rate in training set)
	- unsupervised pretraining