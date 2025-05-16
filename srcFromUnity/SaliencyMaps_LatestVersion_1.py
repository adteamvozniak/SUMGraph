import sys
import os
import os.path as path
import numpy as np
import cv2
import json
fixation_image = []
import glob
import io
import matplotlib.pyplot as plt
#import settings
from .settings import base_path, visible_fov, debug_mode, base_path_debug, list_of_processed_subjects
from PIL import Image
from scipy import optimize


# Variables


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def grayToRgb(img):
    gray = cv2.imread(img, 1)
    red = np.zeros(gray.shape,np.float32)
    for i in range(gray.shape[0]):
        j = 0
        for j in range(gray.shape[1]):
            val = gray[i, j, 0]
            gb = 255 - val
            red[i, j] = [255, gb, gb]

    red = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
    return red

#img_resolution is tuple of the form [height, width]

def increaseSal_3(img_resolution, name, save_loc, fixation_point, dva1, dva2, scale_factors):
    Xin, Yin = np.mgrid[0:img_resolution[1], 0:img_resolution[0]]
    color_map = plt.cm.get_cmap('binary')
    reversed_color_map = color_map.reversed()
    gaze_fix = np.zeros((int(img_resolution[1]), int(img_resolution[0])), np.uint8)
    xx, yy = fixation_point["x"], fixation_point["y"]
    xx = xx * scale_factors[0]
    yy = yy * scale_factors[1]
    yy = img_resolution[1] - yy  # Flip Y-coordinates
    gazeGauss = (gaussian(3, xx, yy, dva1, dva2))(Yin, Xin)
    gaze_fix[yy, xx] = 1.0
    buf = io.BytesIO()
    #print("Save location=", os.path.join(save_loc, "sal_tmp.png"))
    plt.imsave(buf, gazeGauss, cmap=reversed_color_map)
    buf.seek(0)

    img = Image.open(buf)#gazeGauss

    return np.array(img), gaze_fix

def increaseSal_3_memory(img_resolution, name, save_loc, fixation_point, dva1, dva2, scale_factors):
    Xin, Yin = np.mgrid[0:img_resolution[1], 0:img_resolution[0]]
    color_map = plt.cm.get_cmap('binary')
    reversed_color_map = color_map.reversed()
    iterator = 1
    gazeGauss = []

    gaze_fix = np.zeros((int(img_resolution[1]), int(img_resolution[0])), np.uint8)

    for key in reversed(fixation_point):
        xx, yy = key["x"], key["y"]
        xx = xx * scale_factors[0]
        yy = yy * scale_factors[1]
        yy = img_resolution[1] - yy  # Flip Y-coordinates
        #print("After scalling x=", xx, "y=", yy)
        if iterator == 1:
            gazeGauss = (len(fixation_point)/iterator)*(gaussian(30, xx, yy, dva1, dva2))(Yin, Xin)
        else:
            gazeGauss = gazeGauss+(len(fixation_point) / iterator) * (gaussian(30, xx, yy, dva1, dva2))(Yin, Xin)

        if 0 <= yy < img_resolution[1] and 0 <= xx < img_resolution[0]:
            gaze_fix[int(yy), int(xx)] = 1.0

        iterator += 1

    buf = io.BytesIO()
    #print("Save location=", os.path.join(save_loc, "sal_tmp.png"))
    plt.imsave(buf, gazeGauss, cmap=reversed_color_map)
    buf.seek(0)

    img = Image.open(buf)#gazeGauss

    return np.array(img), gaze_fix


def sal_creation(base_path, vpn="", trial_id="", trial_index="", debug_mode=False):
    if debug_mode == True:
        folder = base_path
    else:
        folder = f"{base_path}\\{vpn}\\{trial_id}\\{trial_index}\\"

    dir_xml = [x for x in os.listdir(folder) if x.endswith(".json") and os.path.isfile(os.path.join(folder, x))]
    # load json
    print("List of json files = ", dir_xml)
    for i in range(len(dir_xml)):
        f = open(folder + "\\" + dir_xml[i])
        data = json.load(f)
        # values to be extracted by the parcer
        try:
            memoryActive = data["salMemoryActive"]
        except:
            print("No sal memory is activated")
            memoryActive = False
        if memoryActive:
            fixations = data["fixationsMem"]
        else:
            fixations = data["fixations"]
        trialId = data["trialID"]
        participant = data["participantID"]
        res = data["resolution"]

        destination_folder = "Saliency"+str(dir_xml[i])
        save_loc = f"{base_path}\\{vpn}\\{trial_id}\\{trial_index}\\{destination_folder}\\"
        if not(os.path.isdir(save_loc)):
            os.mkdir(save_loc)

        dva1 = res["x"] / visible_fov
        dva2 = res["y"] / visible_fov

        color_map = plt.cm.get_cmap('binary')
        reversed_color_map = color_map.reversed()

        for fi in fixations:
            frame_nr = fi["frameNumber"]
            file_name = f"{frame_nr}.png"
            if memoryActive:
                #for i in fi["pos2dWithMemory"]:
                #    print("Sal content =",i["x"], i["y"])
                increaseSal_3_memory((int(res["x"]), int(res["y"])), file_name, save_loc, fi["pos2dWithMemory"], dva1, dva2)
            else:
                if((fi["pos2d"]["x"] >= 0.0) and (fi["pos2d"]["y"] >=0.0) and (fi["pos2d"]["x"] < res["x"]) and (fi["pos2d"]["y"] < res["y"])):
                    increaseSal_3((int(res["x"]), int(res["y"])), file_name, save_loc, fi["pos2d"], dva1, dva2)
                else:
                    empty_img = np.zeros((int(res["y"]), int(res["x"])), np.uint8)
                    plt.imsave(save_loc + file_name, empty_img, cmap=reversed_color_map)
                    print(f"frame nr {frame_nr} not valid")
        f.close()

def main():
        if debug_mode:
            sal_creation(base_path_debug, "", "", "", debug_mode)
            print("Debug mode is activated")
        else:
            for vpn in os.listdir(base_path + "\\"):
                if vpn not in list_of_processed_subjects:
                        for trial_id in os.listdir(f"{base_path}\\{vpn}\\"):
                            for trial_index in os.listdir(f"{base_path}\\{vpn}\\{trial_id}\\"):
                                sal_creation(base_path,vpn,trial_id,trial_index, debug_mode)
            else:
                print("Saliency maps already generated for user=", str(vpn))



def createVideo(base_path,trial_id, vpn):
    path_seg = f"{base_path}\\Segmentation\\"
    path_sal = f"{base_path}\\Saliency\\"
    video_folder = f"{base_path}\\Video\\"
    if not(os.path.isdir(video_folder)):
        os.mkdir(video_folder)
    
    images_sal = [img for img in os.listdir(path_sal) if img.endswith(".png")]
    images_seg = [img for img in os.listdir(path_seg) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(path_seg, images_seg[0]))
    height, width, layers = frame.shape 
    video_name = os.path.join(video_folder,'video.avi')
    print(video_name)
    #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))
    print(len(images_sal))
    for i in range(len(images_sal)):
        if i % 1000 == 0:
            print(i)
        seg_name = images_sal[i][-9:]
        if(seg_name in images_seg):
            img1 = cv2.imread(path_seg + seg_name)
            img2 = cv2.imread(path_sal + images_sal[i])
            dst = cv2.addWeighted(img1,1,img2,0.5,0)
            video.write(dst)
        #else:
            #print(images_sal[i])
    
    video.release()
    cv2.waitKey(5)
    cv2.destroyAllWindows()
    
#for vpn in os.listdir(base_path + "\\"):
#    for trial_id in os.listdir(f"{base_path}\\{vpn}\\"):
#        for trial_index in os.listdir(f"{base_path}\\{vpn}\\{trial_id}\\"):
#            print(f"creating video for vpn {vpn}")
#            createVideo(f"{base_path}\\{vpn}\\{trial_id}\\{trial_index}\\",trial_id,vpn)

#if __name__ == '__main__':
#    main()