# VAE_Faces vae-faces
#### The vae_face, layers_face, and utils_face, license, and requirement files were created by Fast Foward Labs.
##### https://github.com/fastforwardlabs/vae-tf

## Directions Part 1 (choose A or B depending on your circumstance) (Pre-Processing)

### A) Using video (mp4)

#### A.1) Place the mp4 files in the videos directory.
#### A.2) Run the following line from terminal with the correct values inputted.
`python3 run.py name_of_person size_of_image fps True`

#### A.3) After the the video is made into jpgs, a new folder will appear in "faces" with the name you inputted. Using Preview (on a Mac) resize the images to a trainable size, I generally use anywhere from 64x64 to 512x512.
#### A.3.1) This size should be the same size as you inputted into size_of_image.
#### A.4) Type in "yes".

### B) Using images (jpgs)

#### B.1) Create a new folder within the "faces folder" with the name of the individual or object. And then another subfolder called name_faces.
#### B.2) Place all of your images (jpgs) in said folder.
#### B.3) Using Preview (on a Mac) resize the images to a trainable size, I generally use anywhere from 64x64 to 512x512.
#### B.4) Run your version of the following code from terminal:
`python3 run.py name_of_face_folder None size_of_image 0 False`

---

## Directions Part 2 (Training)
#### 2.1) You can edit the main_face.py file if you want to adjust any hyperperameters such as the number of iterations, or the batch size. They default to 5000 and 64 respectively.
#### 2.2) Train your code with:
`python3 main_face.py None name_of_face_folder size_of_images latent_space`

---

## Directions Part 3 (Video)
### A) Random Video

#### A.1) Run the following code to generate a random video!
`python3 z_make_random_video.py name numbers_vae_numbers no_of_key_frames size_of_images latent_space`

### B) Specific Video

#### B.1) Now you can choose your key frames from the faces/name/name_faces_grey folder. Place the images in video_data/name/video_#/images/
#### B.1.1) You need to name them like so, 0.png, 1.png, etc. Where 0.png would be the first image in your video.
#### B.2) After training is complete, generate the video run the following command:
`python3 main_face.py numbers_vae_numbers name_of_face_folder size_of_images latent_space`

---

## Possible errors.
* The numbers_vae_numbers can be found by going into the ./out/name/ directory and grabbing the most recent created file name. It should look something like 190120_1294_vae_5420_500_500_2_-10000 Do not add the .index, or the .meta or anything else to the end of it when you use it in the command line.
* Errors from this point could relate to having 'overtrained' and gotten a nan for the avg cost.
* Additionally, the latent space may have too many or too dimensions.

