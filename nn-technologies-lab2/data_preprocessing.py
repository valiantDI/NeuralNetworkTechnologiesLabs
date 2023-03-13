from __future__ import unicode_literals
import pandas as pd
import moviepy.editor as mp
from tqdm import tqdm
from pytube import YouTube

annotation = pd.read_csv("youtube.csv")


def load_data(i, id, category):
    selected_video = YouTube('http://www.youtube.com/watch?v=' + id)
    audio = selected_video.streams.filter(progressive=True, file_extension='mp4')[0]
    name = audio.download('data/sources/')
    clip = mp.VideoFileClip(name)
    clip.audio.write_audiofile(f"data/fold{category}/{str(i)}.mp3")


if __name__ == '__main__':
    for i in tqdm(range(len(annotation))):
        try:
            load_data(i, annotation["link"][i], annotation["category"][i])
        except:
            print(f"Can't load {annotation['link'][i]} file.")
