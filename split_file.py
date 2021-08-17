import os 
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
img_dir = r'C:\Users\Asus\Desktop\AI\SegmentData\FullData\train'
test_dir = r'C:\Users\Asus\Desktop\AI\SegmentData\FullData\test'
val_dir = r'C:\Users\Asus\Desktop\AI\SegmentData\FullData\valid'

files = []

for file in os.listdir(os.path.join(img_dir, 'image') ):
    files.append(file[:-4])


valid, test = train_test_split(files,train_size= 0.2, test_size=0.1, random_state=50)

print(len(valid))
print(len(test))

for file in valid:
    img_current_dir = os.path.join(img_dir, 'image')
    img_current_dir = os.path.join(img_current_dir, file + '.png')
    label_current_dir = os.path.join(img_dir, 'label')
    label_current_dir = os.path.join(label_current_dir, file + '.png')


    img_dest_dir = os.path.join(val_dir, 'image')
    img_dest_dir = os.path.join(img_dest_dir, file + '.png')
    label_dest_dir = os.path.join(val_dir, 'label')
    label_dest_dir = os.path.join(label_dest_dir, file + '.png')


    shutil.move(img_current_dir, img_dest_dir)
    shutil.move(label_current_dir, label_dest_dir)

for file in test:
    img_current_dir = os.path.join(img_dir, 'image')
    img_current_dir = os.path.join(img_current_dir, file + '.png')

    label_current_dir = os.path.join(img_dir, 'label')
    label_current_dir = os.path.join(label_current_dir, file + '.png')


    img_dest_dir = os.path.join(test_dir, 'image')
    img_dest_dir = os.path.join(img_dest_dir, file + '.png')
    label_dest_dir = os.path.join(test_dir, 'label')
    label_dest_dir = os.path.join(label_dest_dir, file + '.png')


    shutil.move(img_current_dir, img_dest_dir)
    shutil.move(label_current_dir, label_dest_dir)
