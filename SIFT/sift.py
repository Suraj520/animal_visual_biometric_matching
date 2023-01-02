

import cv2
import sift
import os
from glob import glob
import pickle
import numpy as np

"""
Steps.

1. Finding the SIFT keypoints and descriptors of the Target Image.
2. Generate a SIFT feature database of all query image.
3. Compare the features of the target image to each of the query features in the database.
4. Use statastical methods to return the best match for a target image.

"""
def get_sift_features(_in_path,_debug_view = False):
    '''
    Generating the SIFT features
    :param _in_path: path to image
    :param _debug_view: -
    :return: keypoints , descriptors
    '''
    img = cv2.imread(_in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp,desc = sift.detectAndCompute(gray, None)

    if _debug_view:
        img = cv2.drawKeypoints(gray, kp, img)
        cv2.imshow('sift_keypoints', img)
        cv2.waitKey(0)

    return kp,desc


def compare_features_flann(_kp1,_dsc1,_kp2,_dsc2,_thres=0):

    # FLANN parameters
    FLANN_INDEX_KDTREE = 3
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(_dsc1, _dsc2, k=2)
    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0,] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            #matches_mask[i] = [1, 0]
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(_kp1) <= len(_kp2):
        number_keypoints = len(_kp1)
    else:
        number_keypoints = len(_kp2)

    return good_points , len(good_points) / number_keypoints * 100


def compare_features_bf(_kp1,_dsc1,_kp2,_dsc2,_thres = 0):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(_dsc1, _dsc2, k=2)
    # Apply ratio test
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            #matches_mask[i] = [1, 0]
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(_kp1) <= len(_kp2):
        number_keypoints = len(_kp1)
    else:
        number_keypoints = len(_kp2)


    return good_points , len(good_points) / number_keypoints * 100

def create_query_database(_path):

    img_db = {}

    for file in glob(_path):
        kp, desc = sift.get_sift_features(file)
        img_db[os.path.basename(file)] = {"keypoint": kp,
                                              "descriptors": desc}



    return img_db

def get_best_matches(_result_dict):

    mean = np.mean([val for key,val in _result_dict.items()])

    positive = {}
    negative = {}

    for key,val in _result_dict.items() :
        res = (val - mean)
        if  res > mean:
            positive[key] = val
        else:
            negative[key] = val

    return positive

if __name__ == "__main__":

    # Give paths to the Query and Targets folder
    target_path = "/home/suraj/Dvara E-Dairy/Search Cattle Problem Statement/SourceCode/SIFT/Test/*.jpg"
    query_path = "/home/suraj/Dvara E-Dairy/Search Cattle Problem Statement/SourceCode/SIFT/Database/*.jpg"

    query_db = create_query_database(query_path)

    for files in glob(target_path, recursive=True):
        results = {}
        kb1, des1 = sift.get_sift_features(files)
        print(os.path.basename(files), "\n")
        for keys, values in query_db.items():
            kb2 = values["keypoint"]
            des2 = values["descriptors"]
            good, percentage = sift.compare_features_flann(kb1, des1, kb2, des2)

            results[keys] = percentage

        print(get_best_matches(results))
        print("-----------------")
        best_matches = get_best_matches(results)
        if len(best_matches)!=0:
            Keymax = max(zip(best_matches.values(), best_matches.keys()))[1]
            print("Best match: ", Keymax)
        