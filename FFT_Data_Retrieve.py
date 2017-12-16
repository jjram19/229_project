import urllib, urllib2, csv, matlab.engine, subprocess, soundfile
from bs4 import BeautifulSoup
from os import listdir, system, chdir, remove, getcwd
from os.path import isfile
import numpy as np

def download_songs(songs, directory, y_val):
    chdir(directory)
    eng = matlab.engine.start_matlab()
    eng.cd(directory)
    song_names = []
    writerx = csv.writer(open('fft_data.csv', 'a'))
    writery = csv.writer(open('y_data.csv', 'a'))
    for textToSearch in songs:
        query = urllib.quote(textToSearch)
        url = "https://www.youtube.com/results?search_query=" + query
        response = urllib2.urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html, 'lxml')
        try:
            for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
                if not vid['href'].startswith("https://googleads.g.doubleclick.net/"):
                    link = 'youtube-dl -o "%(title)s.%(ext)s" --extract-audio --audio-format "wav" ' + ('https://www.youtube.com' + vid['href'])
                    title = 'youtube-dl --get-filename -o "%(title)s.%(ext)s" --extract-audio --audio-format "wav" ' + ('https://www.youtube.com' + vid['href'])
                    Breaker = False
                    if(system(link)): break
                    song_file = subprocess.check_output(title, shell=True)
                    song_file = song_file[:song_file.rfind('.')] + '.wav'
                    if (not all(ord(c) < 128 for c in song_file)): break # if unicode cant find file so skip it
                    iteration = 0 # check if cant find the file
                    while(not isfile(directory + "/" + song_file)):
                        iteration += 1
                        if(iteration == 1000): Breaker = True; break
                    if(Breaker): break
                    new_song = eng.amp_freq_song(song_file, nargout=1)
                    remove(song_file)
                    if new_song == 0: break
                    flattened = [val for sublist in new_song for val in sublist]
                    writery.writerow(y_val)
                    writerx.writerow(flattened)
                    open('song_names.csv', 'a').write(song_file + '\r\n')
                    break
        except:
            continue        




def main(dir):
    for i in range(8):
        chdir(dir)
        temp_songs = []
        with open("artists_" + str(i) + ".csv") as artists:
            temp_artists = artists.readlines()
        temp_artists = [x.strip() for x in temp_artists]
        temp_artists = list(set(temp_artists))
        temp_check = np.zeros(len(temp_artists))

        y_val = [0,0,0,0,0,0,0,0]; y_val[i] = 1
        with open("lyrics_updated.csv") as lyrics:
            lyrics_reader = csv.reader(lyrics, delimiter=',')
            for row in lyrics_reader:
                if row[0] in temp_artists:
                    if temp_check[temp_artists.index(row[0])] > 30:
                        continue
                    temp_songs.append(row[0] + " " + row[1])
                    temp_check[temp_artists.index(row[0])] += 1
        download_songs(temp_songs, '/Users/DanielSalz/ML Songs/Song Data', y_val)




if __name__ == '__main__':
    main(getcwd())