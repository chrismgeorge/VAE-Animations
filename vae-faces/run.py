from z_utils_preprocessing import video_2_jpgs, jpgs_2_png_edges, pngs_2_numpy
import os
import sys

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(name, size, fps, use_video=False):
    face_directory = './faces/'+name+'/'+name+'_faces/'
    grey_face_directory = './faces/'+name+'/'+name+'_faces_grey/'

    numpy_array_directory = './vae_ready_numpy_arrays/'+name+'/'
    video_data_directory_images = './video_data/'+name+'/video_1/images/'
    video_data_directory_video_images = './video_data/'+name+'/video_1/video_images/'

    LOG_DIR = './log/'+name+'/'
    METAGRAPH_DIR = './out/'+name+'/'

    # Make directories for images, if necessary
    checkDirectory(face_directory)
    checkDirectory(grey_face_directory)
    checkDirectory(numpy_array_directory)
    checkDirectory(video_data_directory_images)
    checkDirectory(video_data_directory_video_images)
    checkDirectory(LOG_DIR)
    checkDirectory(METAGRAPH_DIR)

    if (use_video):
        video_2_jpgs(face_directory, fps)

    # If using a video, the function above created the jpgs from the video,
    # now you must go through, use preview and resize all the images, and
    # put them in the correct orientation.
    has_squared_images = input('Have you processed the images into the correct size and orientation (Yes/No)')
    if (has_squared_images.lower() != 'yes'):
        print('process images')
        return

    jpgs_2_png_edges(face_directory, grey_face_directory)
    pngs_2_numpy(name, size, make_test_image=False)


if __name__ == "__main__":
    try:
        name = sys.argv[1]
        size = int(sys.argv[2])
        fps = int(sys.argv[3])
        use_video = (sys.argv[4] == 'True')
        main(name, size, fps, use_video=use_video)
    except(IndexError):
        main()

# Example run without video input
# python3 run.py kat None 128 0 False
