import csv
import parse_artists
import collections
import numpy as np

mapp = parse_artists.artistMap()
train_set_counts = [3760, 1157, 3020]
print(mapp)
token_counts = collections.defaultdict(lambda : collections.defaultdict(int))

with open('lyrics_updated.csv', 'rU') as newcsv:
	reader = csv.reader(newcsv)
	count = 0
	for row in reader:
		lyrics = row[3].rstrip().split(' ')
		if row[0] in mapp:
			for lyric in lyrics:
				token_counts[mapp[row[0]]][parse_artists.tokenParse(lyric)]+=1
		count+=1
		if count > 

genres = [2, 3, 0]

genres = range(0, 8)
for i in enumerate(genres):
	inner_map = token_counts[i]
	count = 0
	for word in inner_map:
		ratio = [0.0]*8
		for j in range(8):
			ratio[j] += token_counts[j][word]
		summ = sum(ratio)
		arr = sorted(ratio)
		new_ratio = np.asarray(ratio)/(sum(ratio))
		if arr[len(arr)-1] == ratio[i] and ratio[i] > 1.5*arr[len(arr)-2] and summ > 150 and i != 2:
			words.append(word)
		elif i == 2 and new_ratio[i] > .9 and summ > 100:
			words.append(word)

with open('tokens', 'wb') as tokens:
	tokens.write(" ".join(sorted(words)))


