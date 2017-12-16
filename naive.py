import sys
import collections
import csv
import random
import copy
import random
import numpy as np
import decimal

random.seed(3)


def readMatrix(tokens, Y, file):
    fd = open(file, 'r')
    cols, rows = [int(s) for s in fd.readline().strip().split()]
    matrix = np.zeros((rows, cols))
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        for k, num in enumerate(nums):
	        matrix[i, k] = num
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}


    # N is number of documents
    # M is number of words/features
    N, M = matrix.shape
    pspam = 0.0
    for i in range(N):
        pspam += category[i]

    pspam = pspam/N
    lengths = [0]*N
    for i in range(N):
        for j in range(M):
            lengths[i] += matrix[i][j]


    phi_one = [0.0]*M
    phi_zero = [0.0]*M

    for i in range(M):
        phi_num = 0.0
        phi_den = 0.0
        for j in range(N):

            phi_num += (category[j])*matrix[j][i]
            phi_den += (category[j]*lengths[j])

        phi_one[i] = (phi_num+1.0)/(phi_den+M)

    for i in range(M):
        phi_num = 0.0
        phi_den = 0.0
        for j in range(N):
            if category[j] == 0:
                phi_num += (1)*matrix[j][i]
                phi_den += (1*lengths[j])


        phi_zero[i] = (phi_num+1.0)/(phi_den+M)

    state = (pspam, phi_one, phi_zero)
    return state


def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    p_spam = state[0]
    p_spam = decimal.Decimal(p_spam)
    p_spam0 = decimal.Decimal(1-p_spam)
    phi_one = state[1]
    phi_zero = state[2]
    n,m = matrix.shape
    a_b = []

    for i in range(n):
        prob_num = 0.0
        prob_den = 0.0
        for j in range(m):
            prob_num += matrix[i][j]*np.log(phi_one[j])
            prob_den += matrix[i][j]*np.log(phi_zero[j]) 
        a = decimal.Decimal(prob_num)
        b = decimal.Decimal(prob_den)
        if a > b: output[i] = 1
    for j in range(m):
        a_b.append(np.log(phi_one[j]) - np.log(phi_zero[j]))





    return output, a_b

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
    print(wrong)

def tokenParse(key):
	key = key.replace('\n','')
	key = key.replace(',', '')
	key = key.replace('(', '')
	key = key.replace(')', '')
	key = key.replace('.', '')
	key = key.replace(',', '')
	key = key.lower()
	return key.rstrip()
	
with open('artists.txt') as rappers:
    rappers = rappers.readlines()
rappers = [x.strip() for x in rappers] 

lyric_map = collections.defaultdict(int)

testY = []
trainY = []

# with open('lyrics_updated.csv', 'wb') as newcsv:
# 	ncsv = csv.writer(newcsv, delimiter = ',')
# 	with open('songdata.csv') as songs:
# 		reader = csv.reader(songs, delimiter=',')
# 		for row in reader:
# 			if row[0] in rappers:
# 				ncsv.writerow(row)
# 	selected = 1
# 	with open('songdata.csv') as songs:
# 		reader = csv.reader(songs, delimiter=',')
# 		for row in reader:
# 			if random.random() < selected and row[0] not in rappers:
# 				ncsv.writerow(row)

with open('tokens') as tokensFile:
	tokens = tokensFile.readline().strip().split()

num_tokens = len(tokens)
index_tokens = collections.defaultdict(int)
for i, token in enumerate(tokens):
	index_tokens[token] = i


testMatrix = []
trainMatrix = []
testSongs = []
with open('lyrics_updated.csv') as lyrics:
	reader = csv.reader(lyrics, delimiter=',')
	zeros = 0
	ones = 0
	for row in reader:
		new_row = [0]*num_tokens
		lyrics = row[3].split(' ')
		for lyric in lyrics:
			lyric.rstrip()
			if lyric in tokens:
				new_row[index_tokens[tokenParse(lyric)]] +=1
			# lyric_map[lyric] +=1
		if row[0] in rappers:
			if ones < 500:
				trainY.append(1)
				trainMatrix.append(new_row)

			else:
				testY.append(1)
				testMatrix.append(new_row)
				testSongs.append((row[0], row[1]))
			ones+=1
		else:
			if zeros < 1000:
				trainY.append(0)
				trainMatrix.append(new_row)
			else:
				testY.append(0)
				testMatrix.append(new_row)
				testSongs.append((row[0], row[1]))

			zeros+=1





# # 
# key_words = []
# for key in lyric_map:
# 	if lyric_map[key] < 3000 and lyric_map[key] > 1500:
# 		key = key.replace('\n','')
# 		key = key.replace(',', '')
# 		key = key.replace('(', '')
# 		key = key.replace(')', '')
# 		key = key.replace('.', '')
# 		key = key.replace(',', '')
# 		key = key.lower()
# 		key_words.append(key.rstrip())


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





print("Loading Time")
trainMatrix, tokenlist, trainCategory = readMatrix(tokens, trainY, 'trainMatrix')
testMatrix, tokenlist, testCategory = readMatrix(tokens, testY, 'testMatrix')

print(len(trainCategory))
print(trainMatrix.shape)

print("Testing Time")
state = nb_train(trainMatrix, trainCategory)
output, a_b = nb_test(testMatrix, state)
top_five = sorted(a_b, reverse=True)[:5]
for top in top_five:
    print(tokenlist[a_b.index(top)])


evaluate(output, testCategory, testSongs)

