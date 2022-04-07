from load_and_preprocess_utils import read_matched_images, load_img
# from face_alignment_utils import *
from face_recognition_utils import find_matches
import dlib
import numpy as np
import sys
import argparse
import os

# sys.path.append(".")
# sys.path.append("..")
import shutil

if __name__ == "__main__":
    # Load the models
    predictor = dlib.shape_predictor("HFGI/checkpoints/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    face_rec_model = dlib.face_recognition_model_v1('HFGI/checkpoints/dlib_face_recognition_resnet_model_v1.dat')

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", help="Path to new faces to add")
    parser.add_argument("--path_encodings", help="Path to face encodings numpy array")
    parser.add_argument("--path_encodings_paths", help="Path to encodings paths numpy array")
    parser.add_argument("--threshold", default=0.5, help="threshold value for distances between images for matching", type=float)
    parser.add_argument("--max_imgs", help="maximum number of images to return", default=5, type=int)
    parser.add_argument("--path_results", help="Path to new person to add", default = 'results_and_data/')
    parser.add_argument("--find_nearest_imgs", help="Find nearest images in case the person is not found", action='store_true')
    parser.add_argument("--return_distance", help="Return the distances with the colsest persons found", action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.path_results):
        os.mkdir(args.path_results)


    all_encodings = np.load(args.path_encodings)
    encoding_paths = np.load(args.path_encodings_paths)

    img = load_img(args.img_path)

    if args.return_distance==True:
        matched_img_paths, distances = find_matches(img, all_encodings, encoding_paths, predictor, detector, face_rec_model, args.threshold, args.max_imgs, args.find_nearest_imgs, args.return_distance)
        print('Distances:', distances)
    else:
        matched_img_paths = find_matches(img, all_encodings, encoding_paths, predictor, detector, face_rec_model, args.threshold, args.max_imgs, args.find_nearest_imgs, args.return_distance)
        
    matched_images = read_matched_images(matched_img_paths)
    # matched_images.insert(0,img)
    # matched_img_paths.insert(0,args.img_path)

    for i in range(0,len(matched_images)):
        img_name = matched_img_paths[i].split('/')[-1]
        matched_images[i].save(os.path.join(args.path_results, img_name))
    
    # display_images_2(matched_images[:5])
    
