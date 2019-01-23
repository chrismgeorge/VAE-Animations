import cv2
import numpy as np
import os
from PIL import Image
import random

################################################################################
################################################################################
################################################################################

# modified https://gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57
def video_2_jpgs(folderName, fps):
    FPS = fps
    currentFrame = 0
    for video in os.listdir('./videos/'):
        if ('mp4' in video): # ignore extra files
            # Playing video from file:
            cap = cv2.VideoCapture('./videos/'+video)
            cap.set(cv2.CAP_PROP_FPS, FPS)

            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Saves image of the current frame in jpg file
                name = '{num:0{width}}'.format(num=currentFrame, width=6)
                name = folderName + name + '.jpg'
                cv2.imwrite(name, frame)

                # To stop duplicate images
                currentFrame += 1

                if (currentFrame % 10 == 0):
                    print ('Created up to... ' + name)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


################################################################################
################################################################################
################################################################################


# by Christopher George
def jpgs_2_png_edges(folderName, newFolderName):
    for fileName in os.listdir(folderName):
        if ('.jpg' in fileName):
            image = cv2.imread(folderName+fileName)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)//250*255
            fileName = fileName.replace('.jpg', '.png')
            cv2.imwrite(newFolderName+fileName, edges)


################################################################################
################################################################################
################################################################################


# by Christopher George
def pngs_2_numpy(name, image_size, reload_image=True, make_test_image=True):
    outfile = './vae_ready_numpy_arrays/'+name+'/'+name+'.npy'

    if make_test_image:
        # reload image to make sure transfer correctly
        test_image_name = './vae_ready_numpy_arrays/'+name+'/'+name+'.png'
        np_array_of_images = np.load(outfile, 'r')
        for array in np_array_of_images:
            # note that the first image we save here is probably not the first
            # image in the correct order
            image = Image.fromarray(np.uint8(np.reshape((array*255), (image_size, image_size))))
            image.save(test_image_name)
            return

    # directory of images
    directory = './faces/'+name+'/'+name+'_faces_grey/'
    number_of_images = len(os.listdir(directory))

    # pre-processing data for vae
    list_of_images = []

    index = 0 # for placement of image in array

    for fileName in os.listdir(directory):
        if random.randint(0, 10) > 8:
            continue

        if '.png' in fileName:
            im_frame = Image.open(directory + fileName)
            # processes image for vae
            # the // 255 is because of image
            new_array = (np.array(im_frame.getdata())//255).tolist()
            # [0, 0, 1, 1, 0, 1, 0...]

            list_of_images.append(new_array)
            index += 1

        if (index % 10 == 0):
            print('Image', index)

    # save new array
    np_array_of_processed_images = np.asarray(list_of_images, dtype=int)
    np.save(outfile, np_array_of_processed_images)
