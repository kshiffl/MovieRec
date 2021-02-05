from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds # For SVP
from scipy import stats # For pearson correlation
import pandas as pd
import numpy as np # For number stuff
import sys # For reading files

train_data = []
train_data_mean = []
test_data = []
movie_genres = []
movie_directors = []
movie_actors = []
tags = []
movie_tags = []
user_taggedmovies = []

# Reads training data, test data, and all files in additional files
def readFiles():
	global train_data, test_data, movie_genres, movie_directors, movie_actors, tags, movie_tags, user_taggedmovies
	train_data = readTrainingFile("train.dat", ['userID','movieID','rating'])

	test_data = readTestFile("test.dat", ['userID','movieID'])

	'''
	movie_genres = readDatFile("movie_genres.dat", ['movieID', 'genre'])
	movie_directors = readDatFile("movie_directors.dat", ['movieID','directorID', 'directorName'])
	movie_actors = readDatFile("movie_actors.dat", ['movieID','actorID','actorName','ranking'])
	tags = readDatFile("tags.dat", ['id','value'])
	movie_tags = readDatFile("movie_tags.dat", ['movieID','tagID','tagWeight'])
	user_taggedmovies = readDatFile("user_taggedmovies.dat", ['userID','movieID','tagID'])
	'''

# Reads training file
def readTrainingFile(fileName, colNames):
	path = "additional_files/"
	with open(path+fileName,"r", errors = 'replace') as readFile:
		read_file = readFile.read().splitlines()
	file_data = [line.split(" ") for line in read_file]
	file_data.pop(0)
	converted_file_data = []
	for row in file_data:
		converted_file_data.append([int(row[0]), int(row[1]), float(row[2])])
	file_data = pd.DataFrame(converted_file_data, columns=colNames)
	file_data = file_data.pivot(index='userID', columns='movieID', values='rating').fillna(0)
	
	# Getting mean rating of user
	global train_data_mean
	for index, row in file_data.iterrows():	
		total = -1
		numReviews = 0
		for column in row:
			if (total == -1 and column > 0):
				total = column
				numReviews+=1 
			elif (column > 0):
				total += column
				numReviews+=1
		if (numReviews == 0):
			mean = None
		else:
			mean = (total / numReviews)
		train_data_mean.append([index, mean])
	train_data_mean = pd.DataFrame(train_data_mean, columns=['userID', 'average_rating'])
	train_data_mean = train_data_mean.set_index('userID')

	# Normalizing
	column_maxes = file_data.max()
	file_data_max = column_maxes.max()
	normalized_data = file_data / file_data_max

	return normalized_data

# Reads test file
def readTestFile(fileName, colNames):
	path = "additional_files/"
	with open(path+fileName,"r", errors = 'replace') as readFile:
		read_file = readFile.read().splitlines()
	file_data = [line.split(" ") for line in read_file]
	file_data.pop(0)
	converted_file_data = []
	for row in file_data:
		converted_file_data.append([int(row[0]), int(row[1])])
	return converted_file_data
		
# Reads the .dat files and splits them into a matrix
def readDatFile(fileName, colNames):
	path = "additional_files/"
	with open(path+fileName,"r", errors = 'replace') as readFile:
		read_file = readFile.read().splitlines()
	if (fileName == "movie_genres.dat" 
	or fileName == "movie_directors.dat"
	or fileName == "movie_actors.dat"
	or fileName == "tags.dat"
	or fileName == "movie_tags.dat"):
		file_data = [line.split("\t") for line in read_file]
	else:
		file_data = [line.split(" ") for line in read_file]
	file_data.pop(0)
	file_data = pd.DataFrame(file_data, columns=colNames)
	return file_data

# Pearson Correlation function
def pearsonCorr(a,b):
	(r,pValue) = stats.pearsonr(a,b)
	return r

# Takes a list of names of the matrices to merge and does an inner merge
# Ex. 	merge_list = ['movie_genres','movie_directors', 'movie_actors']
def mergeInner(merge_list):
	data = []
	if (len(merge_list) <= 1):
		return merge_list[0]
	elif (len(merge_list) == 2):
		data = pd.merge(merge_list[0], merge_list[1], how='inner')
	else:
		data = pd.merge(merge_list[0], merge_list[1], how='inner')
		for x in range(2, len(merge_list)):
			data = pd.merge(data, merge_list[x], how='inner')
	#print(data.head())
	return data

def mergeOn(merge_list, mergeOn):
	data = []
	if (len(merge_list) <= 1):
		return merge_list[0]
	elif (len(merge_list) == 2):
		data = pd.merge(merge_list[0], merge_list[1], on=mergeOn)
	else:
		data = pd.merge(merge_list[0], merge_list[1], on=mergeOn)
		for x in range(2, len(merge_list)):
			data = pd.merge(data, merge_list[x], on=mergeOn)
	return data

def predict(prediction_df):
	predictions = []
	for user in test_data:
		try:
			prediction = prediction_df.loc[user[0],user[1]]
			predictions.append(prediction)
		except KeyError as e: # Couldn't find a prediction for the user (probably a new user?)
			predictions.append(2.5)

	return predictions

def makePredictions():
	# SVD
	u,s,v = svds(train_data, k=100)
	s = np.diag(s)

	print("U:\n")
	print(u[-10:])
	print("\nS:\n")
	print(s[-10:])
	print("\nV:\n")
	print(v[-10:])
	
	# Adding mean rating of users to SVD matrix
	predicted_ratings=np.dot(np.dot(u,s),v)
	tdm = train_data_mean.to_numpy()
	predicted_ratings = predicted_ratings + tdm

	# Turning predictions into DataFrame
	predictions = pd.DataFrame(predicted_ratings, index=train_data.index, columns=train_data.columns)
	print(predictions.head());

	return predictions

# Main
# Command line parameters: train_data.csv, test_data.csv
def main():
	if (len(sys.argv) != 1):
		print("Not enough parameters - Please Enter: python3 recSystem.py")
		exit()

	# Displaying full matrix
	#pd.set_option("display.max_rows", None, "display.max_columns", None)

	# Reading files
	readFiles()

	# Getting predictions for the records
	predictions_df = makePredictions()
	
	# Predicting from test file
	predictions = predict(predictions_df)

	# Writing predictions into output file
	output = open("output.txt", "w")
	for prediction in predictions:
		output.write(str(prediction) + "\n")
	output.close()

main()
