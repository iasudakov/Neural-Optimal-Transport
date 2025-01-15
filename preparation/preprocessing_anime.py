from PIL import Image
import os

def center_crop(im, size):
    left = int(im.size[0]/2-size/2)
    upper = int(im.size[1]/2-size/2)
    right = left + size
    lower = upper + size
    
    return im.crop((left, upper,right,lower))

def noncenter_crop(im, size, shift=(0,0)):
    left = int(im.size[0]/2-size/2) + shift[0]
    upper = int(im.size[1]/2-size/2) + shift[1]
    right = left + size
    lower = upper + size
    
    return im.crop((left, upper,right,lower))

path = './data/safebooru_jpeg'
files = os.listdir(path)

def preprocess_anime_face(path_in_out):
    in_path, out_path = path_in_out
    im = Image.open(in_path).resize((512,512))
    im = noncenter_crop(im, 256, (0, -14)).resize((128, 128))
    im.save(out_path)
    
in_paths = [os.path.join(path, file) for file in files]

out_path = './data/anime_faces'
out_names = [os.path.join(out_path, f'{i}.png') for i in range(len(files))]

if not os.path.exists(out_path):
    os.makedirs(out_path)
    
from multiprocessing import Pool
import time

start = time.time()
with Pool(64) as p:
    p.map(preprocess_anime_face, list(zip(in_paths, out_names)))
end = time.time()
print(end-start)