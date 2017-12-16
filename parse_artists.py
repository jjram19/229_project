import csv 
import collections
def artistMap():
	mapp = {}
	genres = [0,2,3]
	for i in genres:
		filename = 'artists_' + str(i) + '.csv'
		with open(filename, 'rU') as curcsv:
			reader = csv.reader(curcsv)
			for row in reader:
				if row[0] not in mapp:
					mapp[row[0]] = i
	return mapp

def tokenParse(key):
	key = key.replace('\n','')
	key = key.replace(',', '')
	key = key.replace('(', '')
	key = key.replace(')', '')
	key = key.replace('.', '')
	key = key.replace(',', '')
	key = key.lower()
	return key.rstrip()

def tokenUpdate():
	lyric_map = collections.defaultdict(int)
	with open('songdata.csv') as songs:
		reader = csv.reader(songs)
		for row in reader:
			lyrics = row[3].split(' ')
			for lyric in lyrics:
				lyric = tokenParse(lyric)
				lyric_map[lyric] +=1
	arr = []
	for word in lyric_map:
		if lyric_map[word] >1000 and lyric_map[word] < 1500:
			arr.append(word)

	with open('tokens', 'wb') as tokens:
		tokens.write(" ".join(sorted(arr)))

rappers = []
country = []
rock = []
genres = [0,2,3]
songcounts = [0, 0, 0]
for i in genres:
	filename = 'artists_' + str(i) + '.csv'
	with open(filename, 'rU') as curcsv:
		reader = csv.reader(curcsv)
		for row in reader:
			if i == 0:
				rappers.append(row[0])
			elif i == 2:
				country.append(row[0])
			else:
				rock.append(row[0])

with open('songdata.csv') as songs:
	reader = csv.reader(songs)
	for row in reader:
		if row[0] in rappers:
			songcounts[0]+=1
		elif row[0] in country:
			songcounts[1]+=1
		elif row[0] in rock:
			songcounts[2]+=1
print(songcounts)

