# -*- coding: utf-8 -*-    
import cv2    
import numpy as np    
from matplotlib import pyplot as plt
import imageio
import random
from PIL import Image
from Heap import MyHeap
import math

BLOCK_RANGE = 1

class node(object):
	"""docstring for node"""
	def __init__(self, pos):
		super(node, self).__init__()
		self.pos = pos
		self.parent = None
		self.len = 0

	def __lt__(self,value):
		return(self.len < value.len)

class Search(object):
	"""docstring for Search"""
	def __init__(self):
		super(Search, self).__init__()
		self._A_STAR = 0
		self._RRT = 1
		self._BFS = 2
		self.Map = None
		self.obj_pos = np.array([0,0])
		self.target_pos = np.array([0,0])
		self.v = np.array([0,0])
		# myHeap =
		self.heap =  MyHeap(key=lambda item:item.len)


	def set_Map(self,_map):
		self.Map = _map

	def set_target(self,_target):
		self.target_pos = _target
		self.Map[_target[0]][_target[1]] = 254
		
		# 
	def set_start_pos(self,pos):
		self.obj_pos = pos

	def check_safe(self,tempnode,vx,vy):

		if (tempnode.pos[0]<0) or (tempnode.pos[1] < 0):
			return 0

		if (tempnode.pos[0]>=self.Map.shape[0]) or (tempnode.pos[1] >= self.Map.shape[1]):
			return 0

		dx = np.array([1,-1,0,0]) # < > ^ V
		dy = np.array([0,0,1,-1])
		for i in range(0,4):

			p_x = max(0,tempnode.pos[0]+dx[i])
			p_x = min(self.Map.shape[0]-1,tempnode.pos[0]+dx[i])

			p_y = max(0,tempnode.pos[1]+dy[i])
			p_y = min(self.Map.shape[1]-1,tempnode.pos[1]+dy[i])

			# print(p_x,p_y,self.Map.shape)
			if self.Map[p_x][p_y] == 255:
				return 0  

		if self.Map[tempnode.pos[0]][tempnode.pos[1]] == 255:
			return 0 

		if self.Map[tempnode.pos[0]][tempnode.pos[1]] == 100:
			return 0 

		if self.Map[tempnode.pos[0]][tempnode.pos[1]] == 253:
			return 0 

		return 1

	def get_first():
		pass

	def check_no_way_target(self):
		return 0

	def draw_path(self,pos):
		temp = pos
		print(temp.pos)
		print(temp.parent.pos)
		child = self.obj_pos
		while temp.parent != None:
			self.Map[temp.pos[0]][temp.pos[1]] = 253
			child = temp
			temp = temp.parent

		self.v = child.pos - temp.pos

	def BFS(self):
		
		queue = []
		dx = np.array([1,-1,0,0]) # < > ^ V
		dy = np.array([0,0,1,-1])
		head = 0
		tail = 1

		headnode = node(self.obj_pos)
		queue.append(headnode)
		max_len = np.fabs(self.obj_pos-self.target_pos).sum()

		current_len = 0
		self.Map[headnode.pos[0]][headnode.pos[1]] = 253
		while ((current_len <= max_len*5.2) and (head<tail)):
			
			headnode = queue[head]
			# print("head",headnode.pos)y
			for i in range(0,4):
				
				tempnode = node(np.array([dx[i],dy[i]]) + headnode.pos)


				if (self.check_safe(tempnode,dx[i],dy[i]) == 1) :
					
					tempnode.parent = headnode
					if (tempnode.pos == self.target_pos).all():
						print("find!")
						self.draw_path(tempnode)
						self.Map[tempnode.pos[0]][tempnode.pos[1]] = 253
						return tempnode.pos

					if self.Map[tempnode.pos[0]][tempnode.pos[1]] == 254:
						
						print((tempnode.pos == self.target_pos).all())
						print(tempnode.pos)
						print(self.target_pos)
					else:
						self.Map[tempnode.pos[0]][tempnode.pos[1]] = 100
					tail += 1
					queue.append(tempnode)
			current_len = np.fabs(self.obj_pos - tempnode.pos ).sum()
			head +=1   

			permutation = np.random.permutation(dx.shape[0])

			dx = dx[permutation]
			dy = dy[permutation]

			# print ("[",dx[0],dy[0],']',"[",dx[1],dy[1],']',"[",dx[2],dy[2],']',"[",dx[3],dy[3],']')

		return(np.array([0,0]))

	def RRT(self):
		pass

	def A_STAR(self):
		heap = []
		dx = np.array([1,-1,0,0]) # < > ^ V
		dy = np.array([0,0,1,-1])
		head = 0
		tail = 1

		headnode = node(self.obj_pos)
		headnode.len = 0
		NoneNode = node(np.array([0,0]))
		NoneNode.len = 1e9
		self.heap.push(headnode)
		self.heap.push(NoneNode)
		print(headnode)
		max_len = np.fabs(self.obj_pos-self.target_pos).sum()

		current_len = 0
		self.Map[headnode.pos[0]][headnode.pos[1]] = 253
		while ((current_len <= max_len*2.5) and (head<tail)):
			
			headnode = self.heap.pop()
			# print(headnode)
			for i in range(0,4):
				
				tempnode = node(np.array([dx[i],dy[i]]) + headnode.pos)


				if (self.check_safe(tempnode,dx[i],dy[i]) == 1) :
					
					tempnode.parent = headnode
					if (tempnode.pos == self.target_pos).all():
						print("find!")
						self.draw_path(tempnode)
						self.Map[tempnode.pos[0]][tempnode.pos[1]] = 253
						return tempnode.pos

					if self.Map[tempnode.pos[0]][tempnode.pos[1]] == 254:
						
						print((tempnode.pos == self.target_pos).all())
						print(tempnode.pos)
						print(self.target_pos)
					else:
						self.Map[tempnode.pos[0]][tempnode.pos[1]] = 100
						# tempnode.len = math.sqrt((tempnode.pos[0] - self.target_pos[0])**2 + (tempnode.pos[1] - self.target_pos[1])**2)
						
						min_x = min(tempnode.pos[0]-BLOCK_RANGE,1)
						min_y = min(tempnode.pos[1]-BLOCK_RANGE,1)

						max_x = max(tempnode.pos[0]+BLOCK_RANGE+1,self.Map.shape[0])
						max_y = max(tempnode.pos[1]+BLOCK_RANGE+1,self.Map.shape[1])

						
						tempnode.len += math.sqrt((tempnode.pos[0] - self.obj_pos[0])**2 + (tempnode.pos[1] - self.obj_pos[1])**2)
						# tempnode.len += math.sqrt((tempnode.pos[0] - self.target_pos[0])**2 + (tempnode.pos[1] - self.target_pos[1])**2)
						tempnode.len += np.fabs(tempnode.pos - self.target_pos ).sum()		
						tempnode.len += np.fabs(tempnode.pos - self.obj_pos ).sum()		
						

					tail += 1


					self.heap.push(tempnode)


			current_len = np.fabs(self.obj_pos - tempnode.pos ).sum()
			head +=1   

			permutation = np.random.permutation(dx.shape[0])
			
			dx = dx[permutation]
			dy = dy[permutation]

		return(np.array([0,0]))

	def get_base_move_vector(self):
		pass

	def get_next_move_vector(self,method):
		
		if method == self._A_STAR:
		 	return  self.A_STAR()

		if method == self._RRT:
			return self.RRT()

		if method == self._BFS:
			return self.BFS()

	

a = cv2.imread('result.png',0)

mySearch = Search()

mySearch.set_Map(a)
mySearch.set_target(np.array([40,80])*2)
mySearch.set_start_pos(np.array([34,63])*2)

mySearch.get_next_move_vector(mySearch._BFS)
print(mySearch.v)
cv2.imshow( "result", mySearch.Map )       
cv2.waitKey(0)
