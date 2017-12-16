import csv
import collections
names = collections.defaultdict(list)
with open('fresh_artists.csv', 'rU') as newcsv:
	reader = csv.reader(newcsv)
	for row in reader:
		names[int(row[1])].append(row[0])

	print(names)
	for i in range(8):
		with open('artists_' + str(i) + '.csv', 'w') as artists:
			writer = csv.writer(artists)
			for name in names[i]:
				writer.writerow([name])



	

	