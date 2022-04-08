# face_recognition
This repository aims to build a basic face registration and matching system. The input faces are passed through a resnet feature extractor, which converts these images into their embeddings. These embeddings are then stored in a database. Then query image's embeddings are compared against the database and images of indentified person are returned.

## Creating a face recognition database
A face recognition system comprises of the following steps:

* Face Detection: This is the part where a face is detected in a given input image.

* Face Alignment: Once a face is succesfully found in an input image, this face is cropped and aligned for the model.

* Feature Extraction: An aligned face is given to a feature extraction model. This model takes an input image and extract distinguishable features out of the given face.

* Face Recognition: The features of the input image are compared against the features of all the images in the databse and the closest features are identified as the person.

![](<repository_images/Overview-of-the-Steps-in-a-Face-Recognition-Process.png>)

Use the ```create_face_database.py``` file to create the database of face embeddings along with their paths. Once run, there would be two output numpy arrays. One containing the embeddings of all the faces in the dataset, and the other containing the paths of all the images. 
The directory structure to create a database should look like this:
```
Dataset Folder/
    ├───Person 1/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext
    ├───Person 2/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext
    ├───Person 3/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext
                    │
                    │
                    │
```

```
python3  create_face_database.py --path_dataset 'face_recognition_dataset/train/' --savename_encodings 'encodings.npy' --savename_encodings_paths 'encoding_paths.npy'
```
To register a new face in the existing database, use the ```register_new_face.py``` file. Give path of the new person's images along with the paths of existing encodings database and their paths. This will create an updated database.
To register the new face the director structure should look like this:
```
├───Person 1/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext

```

```
python3 register_new_face.py --path_new_person 'face_recognition_dataset/test/7/' --path_encodings_old 'encodings_data/encodings.npy' --path_encodings_paths_old 'encodings_data/encoding_paths.npy' 
```

You can also register multiple people at a time using ```register_multiple_faces.py```:
```
python3 register_multiple_faces.py --path_new_faces 'face_recognition_dataset/test/' --path_encodings_old 'encodings_data/encodings.npy' --path_encodings_paths_old 'encodings_data/encoding_paths.npy'
```
## Finding a matching face in the database
To find a matching face in the database use ```find_matching_images.py```. If there is a face within a certain threshold of the input image, the code will return these images and save them in a folder. **--max_imgs** will limit the number of results you want to display. You can also use the **--find_nearest** flag to return nearest images if the person doesn't exist in the database. Or you can set the **--threshold** to a higher value. You can also display the distances with nearest images using the **--return_distance** flag.
```
python3 find_matching_images.py --img_path 'path/to/img.ext' --path_encodings 'path/to/encodings.npy' --path_encodings_paths 'path/to/encoding_paths.npy' --max_imgs 5 --threshold 0.6
```
