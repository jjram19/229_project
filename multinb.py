from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys
import collections
import csv
import random
import copy
import random
import numpy as np
import decimal
import parse_artists
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from matplotlib.pyplot import *




def readMatrix(tokens, Y, file):
    fd = open(file, 'r')
    cols, rows = [int(s) for s in fd.readline().strip().split()]
    matrix = np.zeros((rows, cols))
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        for k, num in enumerate(nums):
	        matrix[i, k] = num
    return matrix, tokens, np.array(Y)



def evaluate(output, label, testSongs):
    num_raps = 0
    num_non = 0
    total_raps = 0
    total_non = 0
    wrong = []

    for i, out in enumerate(output):
        if out != label[i]:
            if label[i] == 1:
                num_raps+=1
                wrong.append(testSongs[i])
            else:
                num_non+=1
            
        if label[i] == 1:
            total_raps+=1
        else:
            total_non+=1
    print("Num raps wrongs: " + str(num_raps) + " out of " + str(total_raps))
    print("Num non wrongs: " + str(num_non) + " out of " + str(total_non))

    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error

def tokenParse(key):
	key = key.replace('\n','')
	key = key.replace(',', '')
	key = key.replace('(', '')
	key = key.replace(')', '')
	key = key.replace('.', '')
	key = key.replace(',', '')
	key = key.lower()
	return key.rstrip()
	
mapArtists = parse_artists.artistMap()

lyric_map = collections.defaultdict(int)


# parse_artists.tokenUpdate()

with open('tokens') as tokensFile:
	tokens = tokensFile.readline().strip().split()

num_tokens = len(tokens)
index_tokens = collections.defaultdict(int)
for i, token in enumerate(tokens):
	index_tokens[token] = i

testY = []
trainY = []
testMatrix = []
trainMatrix = []
X = []
Y = []
testSongs = []
with open('lyrics_updated.csv') as lyrics:
	reader = csv.reader(lyrics, delimiter=',')
	zeros = 0
	ones = 0
	for row in reader:
		new_row = [0]*num_tokens
		lyrics = row[3].split(' ')
		if row[0] in mapArtists:
			for lyric in lyrics:
				lyric.rstrip()
				if lyric in tokens:
					new_row[index_tokens[tokenParse(lyric)]] +=1

			Y.append(mapArtists[row[0]])
			X.append(new_row)
			testSongs.append((row[0], row[1]))

c = list(zip(X, Y))
random.shuffle(c)
X,Y = zip(*c)
print(len(X))
testY = Y[9000:]
devY = Y[8000:9000]
trainY = Y[:8000]
testMatrix = X[9000:]
devMatrix = X[8000:9000]
trainMatrix = X[:8000]
testSongs = testSongs[9000:]

# with open('tokens', 'wb') as tokens:
# 	tokens.write(" ".join(sorted(key_words)))


with open('testMatrix', 'wb') as test:
	test.write(str(num_tokens) + " " + str(len(testMatrix)) + "\n")
	for row in testMatrix:
		test.write(" ".join([str(x) for x in row]) + "\n")

with open('trainMatrix', 'wb') as train:
	train.write(str(num_tokens) + " " + str(len(trainMatrix)) + "\n")
	for row in trainMatrix:
		train.write(" ".join([str(x) for x in row])+"\n")

with open('devMatrix', 'wb') as dev:
	dev.write(str(num_tokens) + " " + str(len(devMatrix)) + "\n")
	for row in devMatrix:
		dev.write(" ".join([str(x) for x in row]) + "\n")





print("Loading Time")
trainMatrix, tokenlist, trainCategory = readMatrix(tokens, trainY, 'trainMatrix')
testMatrix, tokenlist, testCategory = readMatrix(tokens, testY, 'testMatrix')
devMatrix, tokenlist, devCategory  = readMatrix(tokens, devY, 'devMatrix')



print("Testing Time")


#LAPLACE GRAPH

# evaluate(output, testCategory, testSongs)
# alphas = [0.0, 1.0, 2.0, 3.0, 4.0]
# errors = []
# for a in alphas:

#OTHER MODELS

# clf = MultinomialNB(alpha = 2.0)
# clf.fit(trainMatrix, trainY)
# predicted = clf.predict(testMatrix)
# clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, tol=None)
# clf.fit(trainMatrix, trainY)
# predicted = clf.predict(testMatrix)

clf = SVC()
clf.fit(trainMatrix, trainY)
predicted_train = clf.predict(trainMatrix)
predicted = clf.predict(testMatrix)
print(predicted)

#generate confusion matrix 
genres = [0, 2, 3]
matrix = [[0]*3]*3
matrix = [0]*3
for i, predict in enumerate(predicted):
	# matrix[genres.index(predict)][genres.index(testY[i])]+=1
	if predict == testY[i]:
		matrix[genres.index(predict)]+=1
print(matrix)




#LAPLACE GRAPH


# predicted = clf.predict(devMatrix)
# dev_error = (predicted != devY).sum() *1./len(predicted)
# errors.append(dev_error)
# figure()
# plot(alphas, errors, 'bo')
# title('Error vs Laplace Smoothing Constant')
# xlabel('lambda value')
# ylabel('Error')
# show()
# print(predicted)
error = (predicted_train != trainY).sum() * 1. / len(predicted_train)
print 'Error: %1.4f' % error

error = (predicted != testY).sum() * 1. / len(predicted)
print 'Error: %1.4f' % error
# for i, predict in enumerate(predicted):
# 	if predict == testY[i]:
# 		pass
# 	else:
# 		print(testSongs[i])
# 		print(predict)
# 		print(testY[i])

