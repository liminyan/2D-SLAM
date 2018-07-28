# -*- coding: utf-8 -*-    
import cv2    
import numpy as np    
from matplotlib import pyplot as plt
import imageio
import random
from PIL import Image

MIN_MATCH_COUNT = 4
MAX_IMG_COUNT = 5
MAP_RANGE = 500
SEARCH_RANGE = 8
MAX_MISS = 1.2
PLANE_SIZE = 1
MERGE_RAT = 0.6



def point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("fuck",x,y)

cv2.namedWindow('image')        
cv2.setMouseCallback('image', point)

class Slam(object):
    """  docstring for Slam """
    def __init__(self):
        super(Slam).__init__()

        self.V = np.array([0,0])
        self.V_based_first = np.array([0,0])
        self.result = None
        self.flag = 0
        self.best = []
        self.pre = None
        self.resent_result = None
        self.pos = np.array([0,0])
        self.first_pos = np.array([0,0])
        self.target = np.array([0,0])
        self.temptarget = np.array([0,0])
        self.trace = None
        self.trace_flag = 0
        self.temp = None
        self.temp_pos = np.array([0,0])


    def show_plane_pos(self,pos_x,pos_y,r_x,r_y):
      
        self.temp = np.zeros([self.resent_result.shape[0],self.resent_result.shape[1],3])
        self.temp_pos = np.array([r_x,r_y])
        for x in range(-PLANE_SIZE,1+PLANE_SIZE):
            for y in range(-PLANE_SIZE,1+PLANE_SIZE):
                
                self.temp[x+r_x][y+r_y][2] = 255

        for x in range(-PLANE_SIZE,1+PLANE_SIZE):
            for y in range(-PLANE_SIZE,1+PLANE_SIZE):

                self.temp[x+self.temptarget[0]][y+self.temptarget[1]][1] = 255

       
        for x in range(0,self.resent_result.shape[0]):
            for y in range(0,self.resent_result.shape[1]):

                self.temp[x][y][0] = max(self.temp[x][y][0],self.resent_result[x][y])
                self.temp[x][y][1] = max(self.temp[x][y][1],self.resent_result[x][y])
                self.temp[x][y][2] = max(self.temp[x][y][2],self.resent_result[x][y])


        cv2.imshow( "image", self.temp )  



        cv2.waitKey(0)
        # del self.temp

    def rgb2gray(self,rgb):

        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def set_target(self,tx,ty):

        self.target = np.array([tx,ty])

    def clear_img(self,img):
        
        if np.size(np.shape(img)) == 3:
            img = self.rgb2gray(img)

        ret,thresh1 = cv2.threshold(img,25,100,cv2.THRESH_BINARY)
        return thresh1

    def no_T_Slam(self,img):
        img = mySlam.clear_img(img)
        if self.flag == 0:

            # rows, cols = np.where(img[:,:] !=0)
            # min_row, max_row = min(rows), max(rows) +1
            # min_col, max_col = min(cols), max(cols) +1
            # result = img[min_row:max_row,min_col:max_col] 
            
            result = img
            self.result = result
            self.resent_result = result
            self.flag = 1
            self.pre = result.sum()
            return result

        # rows, cols = np.where(img[:,:] !=0)
        # min_row, max_row = min(rows), max(rows) +1
        # min_col, max_col = min(cols), max(cols) +1
        # img2 = img[min_row:max_row,min_col:max_col]
        
        img2 = img

        temp_sum = img2.sum()
        flag_bad = 0

        merge = random.randint(0,10)/10.0

        self.pre = temp_sum
        best_vector = np.array([0,0])
        best_px = 0
        best_py = 0
        match_value = 0
        len_vector = SEARCH_RANGE*SEARCH_RANGE*2
        
        for rx in range(-SEARCH_RANGE,SEARCH_RANGE+1):
            for ry in range(-SEARCH_RANGE,SEARCH_RANGE+1):

                v1 = np.array([rx,ry]) + self.V
                max_x =  min(img2.shape[0],self.result.shape[0]-v1[0],self.result.shape[0]) 
                max_y =  min(img2.shape[1],self.result.shape[1]-v1[1],self.result.shape[1]) 
                min_x = max(0,v1[0])
                min_y = max(0,v1[1])

                p_min_x = min(0,v1[0])
                p_min_y = min(0,v1[1])

                part_result =  np.zeros(img2.shape - np.array([p_min_x,p_min_y]))
                part_result[0-p_min_x:max_x-min_x-p_min_x,0-p_min_y:max_y-min_y-p_min_y] = self.result[min_x:max_x,min_y:max_y]
                C = (np.fabs(img2 - part_result[0:img2.shape[0],0:img2.shape[1]]))
                count = part_result[0:img2.shape[0],0:img2.shape[1]].sum() + img2.sum()

                temp_match_value = (count - C.sum())
                
                if temp_match_value > match_value:
        
                    match_value = temp_match_value
                    best_vector = v1
                    len_vector = rx*rx+ry*ry
                    best_px = p_min_x
                    best_py = p_min_y
                else:
                    if temp_match_value == match_value:
                        temp_len_vector = rx*rx+ry*ry
                        if temp_len_vector < len_vector:
                            len_vector = temp_len_vector
                            best_vector = v1
                            best_px = p_min_x
                            best_py = p_min_y

        print("best_vector",best_vector,"best value",match_value/50)
        max_x = max(self.result.shape[0],best_vector[0]+img2.shape[0])
        max_y = max(self.result.shape[1],best_vector[1]+img2.shape[1]) 
        temp_new_img = np.zeros([max_x,max_y]-np.array([best_px,best_py]))
                 
        temp_new_img[0-best_px:self.result.shape[0]-best_px,0-best_py:self.result.shape[1]-best_py] = self.result
        for x in range(1,img2.shape[0]):
                for y in range(1,img2.shape[1]):
                    if ( (img2[x][y]) != 0):
                        temp_new_img[x+best_vector[0]-best_px][y+best_vector[1]-best_py] = 255

        #概率合并   
        lenx_pos = int(img2.shape[0]/2)
        leny_pos = int(img2.shape[1]/2)   
        pos_x = lenx_pos + best_vector[0]-best_px
        pos_y = leny_pos + best_vector[1]-best_py    



        print(match_value/2/img2.sum())
        
        match_rate = match_value/2/img2.sum()



        if  (match_rate > 0.5 ) and (merge < MERGE_RAT):
            
            self.best.append(best_vector)
            self.resent_result = temp_new_img
            self.V = best_vector 
            self.pos = np.array([best_px,best_py])
            self.temptarget =  self.target - self.pos
            
            self.show_plane_pos(lenx_pos,leny_pos,pos_x,pos_y)

        else:
            # self.result = self.resent_result
            self.first_pos = self.first_pos - self.pos
            self.V = best_vector  - self.pos
            self.target = self.temptarget 
            self.result = self.resent_result
            
        return temp_new_img

# demo --------

frames = []  
mySlam = Slam()
result = None
for x in range(0,88):
    if x!=6:
        a = cv2.imread('1/'+str(x)+'.jpg',0)
        print(x)
        # a = cv2.resize(a,(int(a.shape[1]/2),int(a.shape[0]/2)))
        # a = a[::-1]
        result = mySlam.no_T_Slam(a)
        frames.append(result)
    else:
        mySlam.set_target(20,20)


cv2.imwrite('result.png',mySlam.result)
print("temp_pos",mySlam.temp_pos)
print("target",mySlam.target)
# imageio.mimsave('demo.gif', frames, 'GIF', duration = 0.25)  






