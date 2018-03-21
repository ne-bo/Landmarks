#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv, urllib.request
from PIL import Image
from io import StringIO, BytesIO
import numpy as np


def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def ParseDataWithLabels(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_label_list = [line[:3] for line in csvreader]
    return key_label_list[1:]  # Chop off header


def DownloadImage(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        # print('Image %s already exists. Skipping download.' % filename)
        return

    try:
        pil_image = Image.open('/media/natasha/Data/Landmark Kaggle/test-full-size/%s.jpg' % key)
        #print(pil_image)
    except Exception:
        print('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_resized = pil_image.resize((256, 256))
        pil_image_resized.save(filename, format='JPEG', quality=90)
    except Exception:
        print('Warning: Failed to save image %s' % filename)
        return


def Run():
    if len(sys.argv) != 3:
        print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=8)
    pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
    Run()
