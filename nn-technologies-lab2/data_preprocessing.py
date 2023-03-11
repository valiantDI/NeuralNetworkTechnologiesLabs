from __future__ import unicode_literals
import pandas as pd
import moviepy.editor as mp
from tqdm import tqdm
from pytube import YouTube

df = pd.read_csv("youtube.csv")


def load_data(id, i, category):
    selected_video = YouTube('http://www.youtube.com/watch?v=' + id)
    audio = selected_video.streams.filter(progressive=True, file_extension='mp4')[0]
    name = audio.download('data/sources/')
    clip = mp.VideoFileClip(name)
    clip.audio.write_audiofile(f"data/fold{str(category + 1)}/{str(i)}.mp3")


n_c = {"history": 0, "art_music": 1, "travel": 2, "food": 3}

for i in tqdm(range(len(df))):
    try:
        load_data(df["link"][i], i, n_c[df["category"][i]])
    except:
        pass
