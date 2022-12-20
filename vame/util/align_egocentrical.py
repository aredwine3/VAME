"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & J. Kürsch & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import cv2 as cv
import numpy as np
import pandas as pd
import tqdm    

from pathlib import Path
from vame.util.auxiliary import read_config  

#Returns cropped image using rect tuple
def crop_and_flip(rect, src, points, ref_index):
    #Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    #Get rotation matrix 
    M = cv.getRotationMatrix2D(center, theta, 1)
       
    #shift DLC points
    x_diff = center[0] - size[0]//2
    y_diff = center[1] - size[1]//2
    
    dlc_points_shifted = []
    
    for i in points:
        point=cv.transform(np.array([[[i[0], i[1]]]]),M)[0][0]

        point[0] -= x_diff
        point[1] -= y_diff
        
        dlc_points_shifted.append(point)
        
    # Perform rotation on src image
    dst = cv.warpAffine(src.astype('float32'), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)
    
    #check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = ((size[0]//2,size[0]//2),size,180)
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        #Get rotation matrix 
        M = cv.getRotationMatrix2D(center, theta, 1)
        
        
        #shift DLC points
        x_diff = center[0] - size[0]//2
        y_diff = center[1] - size[1]//2
        
        points = dlc_points_shifted
        dlc_points_shifted = []
        
        for i in points:
            point=cv.transform(np.array([[[i[0], i[1]]]]),M)[0][0]
    
            point[0] -= x_diff
            point[1] -= y_diff
            
            dlc_points_shifted.append(point)
    
        # Perform rotation on src image
        dst = cv.warpAffine(out.astype('float32'), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)
        
    return out, dlc_points_shifted


#Helper function to return indexes of nans        
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] 


#Interpolates all nan values of given array
def interpol(arr):
        
    y = np.transpose(arr)
     
    nans, x = nan_helper(y[0])
    y[0][nans]= np.interp(x(nans), x(~nans), y[0][~nans])   
    nans, x = nan_helper(y[1])
    y[1][nans]= np.interp(x(nans), x(~nans), y[1][~nans])
    
    arr = np.transpose(y)
    
    return arr

def background(path_to_file,filename,video_format='.mp4',num_frames=1000):
    """
    Compute background image from fixed camera.
    
    Parameters
    ----------
    path_to_file : string
        Project path from config.yaml.
    filename : string
        Name of video file to process (without extension).
    video_format : string, optional (default '.mp4')
        Format of video.
    num_frames : int, optional (default 1000)
        Number of frames to use for computing background.
        
    Returns
    -------
    Array of background values for video.
    """
    import scipy.ndimage
    capture = cv.VideoCapture(os.path.join(path_to_file,'videos',filename+video_format))
    
    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,'videos',filename+video_format)))
        
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    
    height, width, _ = frame.shape    
    frames = np.zeros((height,width,num_frames))

    for i in tqdm.tqdm(range(num_frames), disable=not True, desc='Compute background image for video %s' %filename):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1,rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[...,i] = gray
    
    print('Finishing up!')
    medFrame = np.median(frames,2)
    background = scipy.ndimage.median_filter(medFrame, (5,5))
    
    # np.save(path_to_file+'videos/'+'background/'+filename+'-background.npy',background)
    
    capture.release()
    return background


def align_mouse(path_to_file,filename,video_format,crop_size, pose_list, 
                pose_ref_index, confidence, pose_flip_ref,bg,frame_count,use_video=True):  
    """
    Parameters
    ----------
    path_to_file : string
        Project path from config.yaml.
    filename : string
        Name of video to be processed (without extension)
    video_format : string
        Extension of filename.
    crop_size : tuple of ints
        Size of aligned data to return. Should be larger than the maximum single-frame motion.
    pose_list : list
        List of x,y coordinates & likelihoods for each body part in DLC format.
    pose_ref_index : list
        Reference bodyparts to perform egocentric alignment to.
    confidence : float
        Likelihood cutoff below which x,y values will be interpolated.
    pose_flip_ref : list
        indices of 2 lists in dlc_list to flip mouse if flip was false #TODO Clarify with creators.
    frame_count : int
        Number of frames in video
    use_video : bool, optional (default True)
        Whether to use the video for alignment or only pose data.


    Returns
    -------
    images : list
        if use_video=True, list of cropped and aligned images
    points : list
        List of egocentrically aligned poses.
    time_series : list
        List of frame indices for points and images.

    """
    
    images = []
    points = []
    
    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0],j[1] = np.nan, np.nan      
                

    for i in pose_list:
        i = interpol(i)
    
    if use_video:
        capture = cv.VideoCapture(os.path.join(path_to_file,'videos',filename+video_format))

        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,'videos',filename+video_format)))
          
    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc='Align frames'):
        
        if use_video:
            #Read frame
            try:
                ret, frame = capture.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame - bg
                frame[frame <= 0] = 0
            except:
                print("Couldn't find a frame in capture.read(). #Frame: %d" %idx)
                continue
        else:
            frame=np.zeros((1,1))
            
        #Read coordinates and add border
        pose_list_bordered = []
                
        for i in pose_list:
            pose_list_bordered.append((int(i[idx][0]+crop_size[0]),int(i[idx][1]+crop_size[1])))
        
        img = cv.copyMakeBorder(frame, crop_size[1], crop_size[1], crop_size[0], crop_size[0], cv.BORDER_CONSTANT, 0)
        
        punkte = []
        for i in pose_ref_index:
            coord = []
            coord.append(pose_list_bordered[i][0])
            coord.append(pose_list_bordered[i][1])
            punkte.append(coord)
        punkte = [punkte]
        punkte = np.asarray(punkte)
        
        #calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)
    
        #change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)
        
        center, size, theta = rect
        
        #crop image
        out, shifted_points = crop_and_flip(rect, img,pose_list_bordered,pose_flip_ref)
        
        if use_video: #for memory optimization, just save images when video is used.
            images.append(out)
        points.append(shifted_points)
        
    if use_video:
        capture.release()
    
    time_series = np.zeros((len(pose_list)*2,frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx:idx+2,i] = points[i][j]
            idx += 2
        
    return images, points, time_series


#play aligned video
def play_aligned_video(a, n, frame_count, path_to_file, save=False, filename=None, crop_size=None):
    """
    Parameters
    ----------
    a : array
        Array or list of frames
    n : list-like
        Body parts for egocentric alignment
    frame_count : int
        Number of frames.
    path_to_file : string
        Project path from config.yaml.
    save : bool, optional (default False)
        Whether to save egocentrically aligned video.
    filename : string, optional (default None)
        Name of video to save if save=True.
    crop_size : tuple, optional
        Size of aligned video. Should be larger than the maximum single-frame motion.

    Returns
    -------
    None.

    """
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255)]
    if not os.path.exists(os.path.join(path_to_file, 'egocentricVideos/')):
        os.mkdir(os.path.join(path_to_file, 'egocentricVideos/'))
    if save:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(
            os.path.join(os.path.join(path_to_file, 'egocentricVideos/' + filename + '.mp4')), 
            fourcc, 
            30.0, 
            crop_size,
            isColor=True
        )
    for i in range(frame_count):
        # Capture frame-by-frame
        ret, frame = True,a[i]
        if ret == True:
            
          # Display the resulting frame
          frame = cv.cvtColor(frame.astype('uint8')*255, cv.COLOR_GRAY2BGR)
          im_color = cv.applyColorMap(frame, cv.COLORMAP_JET)
          
          for c,j in enumerate(n[i]):
              cv.circle(im_color,(j[0], j[1]), 5, colors[c], -1)
              
          if not save:
              cv.imshow('Frame',im_color)
              if cv.waitKey(25) & 0xFF == ord('q'): # Press Q on keyboard to  exit
                  break
          elif save:
              writer.write(im_color)    
        
        # Break the loop
        else: 
            break
    if save:
        writer.release()
    cv.destroyAllWindows()


def alignment(config, filename, pose_ref_index, video_format, crop_size, use_video=False, check_video=False, save=False, blank_background=True):
    """
    Parameters
    ----------
    config : string
        Path to config file
    filename : string
        Name of video to be processed (without extension)
    pose_ref_index : list
        List of 2 body parts to egocentrically align to.
    video_format : string
        Extension of filename.
    crop_size : tuple
        Size to crop aligned data to. Should be larger than maximum single-frmae movement.
    use_video : Tbool, optional (default False)
        Whether to use the video or only DLC poses.
    check_video : bool, optional (Default False)
        Whether to display the video. Press Q to exit while displaying.
    save : bool, optional, (default False)
        Whether to save the video instead of displaying.

    Returns
    -------
    time_series : list
        Frame indices
    frames : list of arrays
        Aligned poses or video frames.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    confidence = cfg['pose_confidence']
    #read out data
    data = pd.read_csv(os.path.join(path_to_file,'videos','pose_estimation',filename+'.csv'), skiprows = 0, header=[1,2], index_col=0)
    data_mat = pd.DataFrame.to_numpy(data)
    
    # get the coordinates for alignment from data table
    pose_list = []
    
    for i in range(int(data_mat.shape[1]/3)):
        pose_list.append(data_mat[:,i*3:(i+1)*3])  
        
    #list of reference coordinate indices for alignment
    #0: snout, 1: forehand_left, 2: forehand_right, 
    #3: hindleft, 4: hindright, 5: tail    
    
    pose_ref_index = pose_ref_index
    
    #list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index
        
    if use_video:
        if not blank_background:
        #compute background
            bg = background(path_to_file,filename,video_format)
        elif blank_background:
            bg=0

        capture = cv.VideoCapture(os.path.join(path_to_file,'videos',filename+video_format))
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,'videos',filename+video_format)))          
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        capture.release()
    else:
        bg = 0
        frame_count = len(data) # Change this to an abitrary number if you first want to test the code
    
    
    frames, n, time_series = align_mouse(path_to_file, filename, video_format, crop_size, pose_list, pose_ref_index,
                                         confidence, pose_flip_ref, bg, frame_count, use_video=use_video)
    
    if check_video and not save:
        play_aligned_video(frames, n, frame_count, path_to_file)
    elif check_video and save:
        play_aligned_video(frames, n, frame_count, path_to_file, save=True, filename=filename, crop_size=crop_size)

    return time_series, frames


def egocentric_alignment(config, pose_ref_index=[0,5], crop_size=(300,300), use_video=False, video_format='.mp4', check_video=False, save=False, blank_background=True):
    """ Happy aligning """
    #config parameters
    if save and not use_video:
        raise ValueError("If save=True use_video must also be True.")
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    path_to_file = cfg['project_path']
    filename = cfg['video_sets']
    confidence=cfg['pose_confidence']
    video_format=video_format
    crop_size=crop_size
    
    if cfg['egocentric_data'] == True:
        raise ValueError("The config.yaml indicates that the data is not egocentric. Please check the parameter egocentric_data")
    
    # call function and save into your VAME data folder
    for file in filename:
        if not os.path.exists(os.path.join(path_to_file,'data',file,file+'-PE-seq.npy')):
            print("Aligning data %s, Pose confidence value: %.2f" %(file, confidence))
            egocentric_time_series, frames = alignment(config, file, pose_ref_index, video_format, crop_size, 
                                                       use_video=use_video, check_video=check_video, save=save, blank_background=blank_background)
            np.save(os.path.join(path_to_file,'data',file,file+'-PE-seq.npy'), egocentric_time_series)
        else:
            print("Aligned video data already exists. To realign this video delete the PE-seq.npy file from the data directory and run this function again")
    print("Your data is now egocentrically aligned and you can call vame.create_trainset()")

