1# -*- coding: utf-8 -*-    
import cv2    
import numpy as np    
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

class Slam(object):
    """docstring for Slam"""
    def __init__(self):
        super(Slam).__init__()
    
    def GetGoodMatch(self,img1,img2 ):

        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img2,None)
        kp2, des2 = sift.detectAndCompute(img1,None)
     
        FLANN_INDEX_KDTREE = 0    
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    
        search_params = dict(checks = 50)   # or pass empty dictionary 

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)
       
        good = []    
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)
        return good

    def Slam2D(self, img1, img2):  
    

        good = GetGoodMatch(img1,img2)

        if len(good)>MIN_MATCH_COUNT:

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            wrap = cv2.warpPerspective(img2, H, (img1.shape[1]+img2.shape[1] , img1.shape[0]+img2.shape[0]))
           
            for x in range(1,img1.shape[0]):
                for y in range(1,img1.shape[1]):
                    if (sum (img1[x][y]) != 255*3):
                        wrap[x][y] = img1[x][y]
                    # else:
                    #     print(img1[x][y])
            rows, cols = np.where(wrap[:,:,0] !=0)
            min_row, max_row = min(rows), max(rows) +1
            min_col, max_col = min(cols), max(cols) +1
            result = wrap[min_row:max_row,min_col:max_col,:]

            # cv2.imshow('result.jpg',result)
            # cv2.waitKey(0)
            return result;   
        else:
            print("error:bad match!")
            return img2

    def Slam3D():
        pass



# demo 
'''
a,b = get_imgs("5.jpg","2.jpg")
mySlam = Slam()
result = mySlam.Slam2D(a,b)
cv2.imshow('result.jpg',result)
cv2.waitKey(0)
'''





