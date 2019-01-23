import os
import sys

import numpy as np
import tensorflow as tf

from faces import Faces
import vae_face

from PIL import Image
import cv2
from scipy.interpolate import interp1d


HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 5000 #2**16
MAX_EPOCHS = np.inf

def load_images(image_path, image_size):
    number_of_files = len(os.listdir(image_path))
    number_of_pngs = sum('png' in s for s in os.listdir(image_path))
    images = []
    # assuming correctly named files
    for fileName in range(number_of_pngs):
        img = Image.open(image_path+str(fileName)+'.png')
        img = np.reshape(np.array(img, dtype='uint8'), (1, image_size*image_size))
        #img = np.reshape(img[0][:784], (1, 784))
        images.append(img)

    return images

def makeVideo(model, student_name, input_image_path, image_size, latent_space):
    images = load_images(input_image_path, image_size)

    video_name = 'video_data/'+student_name+'/video_1/video.mp4' # edit me
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 24.0, (image_size, image_size))

    image_interpolations = [24]*(len(images)-1)

    if (len(image_interpolations) != len(images) - 1):
        print('need images minus one interpolations')
        return

    totalImages = 0

    for index, image in enumerate(images):

        # get two images to interpolate between
        encoded_image = model.encode(image)

        if (index == len(images) - 1):
            break
        else:
            next_image = model.encode(images[index+1])

        # get a linear interpolation between the encodings of said images
        linfit = interp1d([1, image_interpolations[index]], np.vstack([encoded_image[0], next_image[0]]),
                           axis=0, bounds_error=False)

        # get N inerpolations between images
       # print('EN', [encoded_image[0], next_image[0]])
        for linfit_index in range(1, image_interpolations[index]):
            #print('l', linfit(linfit_index))
            interpolated_encoding = np.reshape(np.asarray(linfit(linfit_index)),
                                               (1, latent_space)) # edit depending on what was used

            # using the decoder to get the decoded image
            decoded_image = model.decode(interpolated_encoding)

            # process image back for video
            img = np.interp(decoded_image, (decoded_image.min(), decoded_image.max()), (0, 1))
            img = np.resize(img, (image_size, image_size)) # edit me

            img = Image.fromarray(np.uint8(img*255))

            # save image for video
            name = './video_data/'+student_name+'/video_1/video_images/'+str(totalImages)+'.png'
            img.save(name)

            # write image to video
            video.write(cv2.imread(name))

            # update name
            totalImages += 1
        print(index)


    cv2.destroyAllWindows()
    video.release()

def main(to_reload, student_name, image_size, latent_space):

    IMG_DIM = image_size

    ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                    500, 500, # intermediate encoding
                    latent_space] # latent space dims
    # (and symmetrically back out again)

    numpy_dir = './vae_ready_numpy_arrays/'+student_name+'/'+student_name+'.npy'
    images_dir = './video_data/'+student_name+'/video_1/images/'

    if to_reload != 'None': # restore
        meta_graph_dir = './out/'+student_name+'/'+to_reload
        v = vae_face.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=meta_graph_dir)
        print("Loaded!")
        if image_size != 0:
            makeVideo(v, student_name, images_dir, image_size, latent_space)
        else:
            print('Need image path to make video!')

    else: # train
        faces = Faces(numpy_dir)
        LOG_DIR = './log/'+student_name+'/'
        METAGRAPH_DIR = './out/'+student_name+'/'
        v = vae_face.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(faces, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
                verbose=True, save=True, outdir=METAGRAPH_DIR)
        print("Trained!")


if __name__ == "__main__":
    tf.reset_default_graph()

    try:
        to_reload = sys.argv[1]
        student_name = sys.argv[2]
        image_size = int(sys.argv[3])
        latent_space = int(sys.argv[4])
        main(to_reload, student_name, image_size, latent_space)
    except(IndexError):
        main()

# chris
# 1, 8

# runner
# 1, 4

# heart_2
# 2
# 4
