import cv2    
import numpy as np    
from matplotlib import pyplot as plt
# from pcl import save
# from pcl import PointCloud
MIN_MATCH_COUNT = 10
# def get_imgs(img1,img2):
#     im1 = cv2.imread(img1)          # queryImage
#     im2 = cv2.imread(img2)
#     return im1 , im2

def computeMatches(img1, img2):  

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
	    if m.distance < 0.7*n.distance:
	        good.append(m)
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
	    result = wrap[min_row:max_row,min_col:max_col,:]#去除黑色无用部分
	    cv2.imshow('result.jpg',result)
	    cv2.waitKey(0)

a = cv2.imread('15.jpg')
b = cv2.imread('12.jpg')
computeMatches(a,b)











