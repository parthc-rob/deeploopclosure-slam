# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 01:05:22 2019

@author: Kun Sun
"""


import shutil
import os


def move(data_dir, out_dir):
    '''
    Merge all images in the subfolders of 'data_256' into one uniform folder.
    Param:
       data_dir: path of folder 'data_256'
       out_dir: path of the target folder we want to move all images into
    '''

    # get initial characters a - z of all scene classes
    alphabet = os.listdir(data_dir)
    # BFS search
    for N in range(0, len(alphabet)):
        print('Parsing all scene classes starting with letter: ', alphabet[N])
        classes = os.listdir(data_dir + '/' + alphabet[N])
        # move the images in each class into the target folder
        for n in range(0, len(classes)):
            print('Parsing all images under scene class: ', classes[n])
            names = os.listdir(data_dir + '/' + alphabet[N] + '/' + classes[n])
            if names[0].endswith('.jpg'):
                # path in format like: a/airport/123.jpg
                copy_images(data_dir+'/'+alphabet[N]+'/'+classes[n], out_dir)
            else:
                # path in format like: l/lake/natural/123.jpg
                for i in range(0, len(names)):
                    folder_path = data_dir+'/'+alphabet[N]+'/'+classes[n]+'/'+names[i]
                    copy_images(folder_path, out_dir)

        print('Finishing copying all classes staring with letter: ', alphabet[N])


def copy_images(folder_path, out_dir):
    global tail
    names = os.listdir(folder_path)
    for i in range(0, len(names)):
        source_dir = folder_path + '/' + names[i]
        target_dir = out_dir + '/' + str(tail) + '.jpg'
        print('Copying image from: ', source_dir)
        shutil.copyfile(source_dir, target_dir)
        tail = tail + 1

    print('Finishing copying from folder: ', folder_path)

    return


if __name__ == "__main__":
    data_dir = '/home/ubuntu/data_256'
    out_dir = '/home/ubuntu/raw256_108G'
    tail = 1
    move(data_dir, out_dir)
