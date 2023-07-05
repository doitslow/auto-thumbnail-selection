import os
import shutil

din = '/raid/P15/4-data/mediacorp'

folders = []
for root, dirs, files in os.walk(din):
    for file in files:
        if file.endswith('.mp4'):
            folders.append(root)

for folder in folders:
    for subdir in os.listdir(folder):
        if subdir.endswith('-removals.txt'):
            shutil.copy(os.path.join(folder, subdir), os.path.join(folder, subdir).replace('-removals.txt', '-old.txt'))
            os.remove(os.path.join(folder, subdir))

