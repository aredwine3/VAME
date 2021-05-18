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

def background(path_to_file,filename,file_format='.mp4',num_frames=1000):
    """
    Compute background image from fixed camera 
    """
    import scipy.ndimage
    capture = cv.VideoCapture(path_to_file+'videos/'+filename+file_format)
    
    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(path_to_file+filename))
        
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


def align_mouse(path_to_file,filename,file_format,crop_size, pose_list, pose_ref_index,
                      pose_flip_ref,bg,frame_count,use_video=True):  
    """Docstring:
        Perform egocentric alignment of coordinates from CSV file.
        
        Parameters
        ----------
        path_to_file: string, directory
        filename: string, name of video file without format
        file_format: string, format of video file
        crop_size: tuple, x and y crop size
        dlc_list: list, arrays containg corresponding x and y DLC values
        dlc_ref_index: list, indices of 2 lists in dlc_list to align mouse along
        dlc_flip_ref: tuple, indices of 2 lists in dlc_list to flip mouse if flip was false
        bg: background image to subtract
        frame_count: number of frames to align
        use_video: boolean if video should be cropped or DLC points only
        
        Returns: list of cropped images (if video is used) and list of cropped DLC points
    """ 
    images = []
    points = []
    
    for i in pose_list:
        for j in i:
            if j[2] <= 0.8:
                j[0],j[1] = np.nan, np.nan      
                

    for i in pose_list:
        i = interpol(i)
    
    if use_video:
        capture = cv.VideoCapture(path_to_file+'videos/'+filename+file_format)

        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(path_to_file+'videos/'+filename))
            
    
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
        out,shifted_points = crop_and_flip(rect, img,pose_list_bordered,pose_flip_ref)
        
        images.append(out)
        points.append(shifted_points)
    
    time_series = np.zeros((len(pose_list)*2,frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx:idx+2,i] = points[i][j]
            idx += 2
        
    return images, points, time_series


#play aligned video
def play_aligned_video(a, n, frame_count, path_to_file, filename, crop_size, save=False):
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),(125,0,125),(125,125,125)]
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
    writer.release()
    cv.destroyAllWindows()
    

def alignVideo(path_to_file, filename, file_format, crop_size, use_video=False, check_video=False, save=False):
    """Docstring:
    Performs egocentric alignment of video data.
    
    Parameters
    ----------
    path_to_file : string
        path to CSV file
    filename : string
        name of subject in file, without format
    file_format : string
        format of video file
    crop_size : tuple
        tuple of ints for size of cropped egocentric frames
    use_video : bool (optional, default False)
        Whether to use openCV to read and analyze videos.
    check_video : bool (optional, default False)
        Whether to play result video upon completion.
    save : bool (optional, default False)
        Whether to save the result video. Check video must also be true.
    """
    #read out data
    data = pd.read_csv(path_to_file+'/videos/pose_estimation/'+filename+'-DC.csv', skiprows = 2)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:,1:] 
    
    # get the coordinates for alignment from data table
    pose_list = []
    
    for i in range(int(data_mat.shape[1]/3)):
        pose_list.append(data_mat[:,i*3:(i+1)*3])  
        
    #list of reference coordinate indices for alignment
    #0: snout, 1: forehand_left, 2: forehand_right, 
    #3: hindleft, 4: hindright, 5: tail    
    
    pose_ref_index = [0,1]
    
    #list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = [0,1]
        
    if use_video:
        #compute background
        bg = background(path_to_file,filename)
        capture = cv.VideoCapture(path_to_file+'videos/'+filename+file_format)
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(path_to_file+'videos/'+filename))
            
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    else:
        bg = 0
        frame_count = len(data) # Change this to an abitrary number if you first want to test the code
    
    
    a,n, ego_time_series = align_mouse(path_to_file, filename, file_format, crop_size, pose_list, pose_ref_index,
                      pose_flip_ref, bg, frame_count, use_video)

    if check_video and not save:
        play_aligned_video(a, n, frame_count, path_to_file, filename, crop_size, save=False)
    elif check_video and save:
        play_aligned_video(a, n, frame_count, path_to_file, filename, crop_size, save=True)
      
    return ego_time_series
