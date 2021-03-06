# from load_and_preprocess_utils import *
# from face_alignment_utils import *
from face_recognition_utils import register_multiple_persons
import dlib
import numpy as np
import sys
import argparse
import os
# sys.path.append(".")
# sys.path.append("..")

if __name__ == "__main__":
    # Load the models
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_new_faces", help="Path to new faces to add")
    parser.add_argument("--path_encodings_old", help="Path to existing encodings database")
    parser.add_argument("--path_encodings_paths_old", help="Path to existing encodings paths database")
    parser.add_argument("--savename_encodings_updated", help="name to save encoding vectors as npy array", default='face_encodings_updated.npy')
    parser.add_argument("--savename_encodings_paths_updated", help="name to save encoding paths as npy array", default='face_imgs_paths_updated.npy')
    parser.add_argument("--save_dir", help="Directory to save encoding data", default='encodings_data/')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    all_encodings = np.load(args.path_encodings_old)
    encoding_paths = np.load(args.path_encodings_paths_old)

    all_encodings_updated, encoding_paths_updated = register_multiple_persons(args.path_new_faces, all_encodings, encoding_paths, predictor, detector, face_rec_model)
    
    np.save(os.path.join(args.save_dir,args.savename_encodings_updated), all_encodings_updated)
    np.save(os.path.join(args.save_dir,args.savename_encodings_paths_updated), encoding_paths_updated)  
