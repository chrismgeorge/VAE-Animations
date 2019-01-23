import os
import sys

import numpy as np
import tensorflow as tf

from faces import Faces
import vae_face

from PIL import Image
import cv2
from scipy.interpolate import interp1d
import random


HYPERPARAMS = {
    "batch_size": 32,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 5000 #2**16
MAX_EPOCHS = np.inf

def load_images(grey_images_dir, image_size, no_of_key_frames):
    grey_images = os.listdir(grey_images_dir)
    images = []

    while no_of_key_frames > 0:
        while True:
            random_index = random.randint(0, len(grey_images)-1)
            if '.png' in grey_images[random_index]:
                break

        random_picture = grey_images[random_index]
        img = Image.open(grey_images_dir+random_picture)
        img = np.reshape(np.array(img, dtype='uint8'), (1, image_size*image_size))
        images.append(img)
        no_of_key_frames -= 1

    return images

def makeVideo(model, student_name, grey_images_dir, video_images_dir, video_dir, image_size,
              latent_space, no_of_key_frames):

    images = load_images(grey_images_dir, image_size, no_of_key_frames)

    video_name = video_dir+'video.mp4' # edit me
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 24.0, (image_size, image_size))

    # image_interpolations = []
    # for i in range(len(images)-1):
    #     image_interpolations.append(random.randint(4, 8))
    image_interpolations = [20, 16, 4, 6, 7, 7, 10, 16, 5, 3,
                            20, 16, 6, 24, 20, 16, 7, 4, 43, 8, 4, 34,
                            5, 20, 16, 8, 10, 10, 9, 17, 10, 8, 12, 14, 20, 16, 9,
                            4, 20, 16, 10, 12, 20, 16, 11, 22, 20, 17, 3, 8, 6, 10, 16, 5,
                            20, 17, 6, 3, 20, 17, 7, 9, 10, 5, 7, 20, 17, 8, 32, 20, 17, 9, 15,
                            20, 17, 10, 11, 20, 17, 11, 8, 6, 20, 17, 12, 5, 20, 18, 2, 12, 20, 18, 3, 8,
                            20, 18, 4, 4, 4, 20, 18, 5, 8, 20, 19, 1, 13, 50, 16, 10, 11, 23]

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
        for linfit_index in range(1, image_interpolations[index]):

            interpolated_encoding = np.reshape(np.asarray(linfit(linfit_index)),
                                               (1, latent_space)) # edit depending on what was used

            # using the decoder to get the decoded image
            decoded_image = model.decode(interpolated_encoding)

            # process image back for video
            img = np.interp(decoded_image, (decoded_image.min(), decoded_image.max()), (0, 1))
            img = np.resize(img, (image_size, image_size)) # edit me

            img = Image.fromarray(np.uint8(img*255))

            # save image for video
            name = video_images_dir+str(totalImages)+'.png'
            img.save(name)

            # write image to video
            video.write(cv2.imread(name))

            # update name
            totalImages += 1
        print("Key-Frame: ", index)


    cv2.destroyAllWindows()
    video.release()

def main(name, meta_graph_name, no_of_key_frames, image_size, latent_space):

    IMG_DIM = image_size
    ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                    500, 500, # intermediate encoding
                    latent_space] # latent space dims
    # (and symmetrically back out again)

    numpy_dir = './vae_ready_numpy_arrays/'+name+'/'+name+'.npy'

    i = 1
    while os.path.exists('./video_data/'+name+'/video_%s/' % i):
        i += 1
    video_images_dir = './video_data/'+name+'/video_'+str(i)+'/video_images/'
    video_dir = './video_data/'+name+'/video_'+str(i)+'/'
    os.makedirs(video_images_dir)

    grey_images_dir = './faces/'+name+'/'+name+'_faces_grey/'
    meta_graph_dir = './out/'+name+'/'+meta_graph_name

    v = vae_face.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=meta_graph_dir)
    makeVideo(v, name, grey_images_dir, video_images_dir, video_dir, image_size, latent_space, no_of_key_frames)


if __name__ == "__main__":
    tf.reset_default_graph()

    name = sys.argv[1]
    meta_graph_name = sys.argv[2]
    no_of_key_frames = int(sys.argv[3])
    image_size = int(sys.argv[4])
    latent_space = int(sys.argv[5])
    main(name, meta_graph_name, no_of_key_frames, image_size, latent_space)


# kat_1, lat=8,
# heart_1, lat=4
# python3 z_make_random_video.py kat 190119_1727_vae_16384_500_500_8-5000 100 128 8
