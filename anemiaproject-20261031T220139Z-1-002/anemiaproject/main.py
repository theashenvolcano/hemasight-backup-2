import os, collections
import kagglehub

path = kagglehub.dataset_download("harshwardhanfartale/eyes-defy-anemia")
print("Downloaded to:", path)

ext = collections.Counter()
shown = 0
for root, dirs, files in os.walk(path):
    for f in files:
        ext[os.path.splitext(f)[1].lower()] += 1
    if files and shown < 4:
        print(root, "->", files)
        shown += 1
print("File types:", dict(ext))