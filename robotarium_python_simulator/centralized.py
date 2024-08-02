#!/usr/env/bin python3
#Importing libraries
import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import csv
import numpy as np
import pandas as pd
import math
import bisect
import matplotlib.pyplot as plt

class task:
    def __init__(self,N,T):
        self.tasks = T
        self.N = N
        self.si_to_uni_dyn, self.uni_to_si_states = create_si_to_uni_mapping()

        #Boundaries
        self.x_min_robotarium = -1.5
        self.x_max_robotarium = 1.5
        self.y_min_robotarium = -1
        self.y_max_robotarium = 1
        self.res = 0.1

        self.positional_controller_constant=0.9
        self.safety_radius = 0.04
        
        self.battery = [6,7,6,9,7,5,6,4,3,6][:self.N]
        self.robo_thresh = 2.0
        self.battery_station = [0,0]
        
        self.k=-1
        self.task_list = []
        self.tasks_completed = [False] * len(self.tasks)
        self.task_per_robot = [[] for _ in range(self.N)]
        self.path_per_robot = [[] for _ in range(self.N)]
        self.bid_per_robot = [[] for _ in range(self.N)]
        
    def euclidean_distance(self,p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
    def robo_visual(self):
        self.r = robotarium.Robotarium(number_of_robots=self.N, show_figure=True, sim_in_real_time=True)
        self.L = completeGL(self.N)
        self.x = self.r.get_poses()
        self.distance_travelled = np.zeros(self.N)
        self.previous_pose = np.zeros((self.N,2))
        self.CM = np.random.rand(self.N,3)
        self.safety_radius_marker_size = determine_marker_size(self.r,self.safety_radius)
        self.font_height_meters = 0.1
        self.font_height_points = determine_font_size(self.r,self.font_height_meters) 
        self.facecolor=['red','blue','green','black','yellow','plum','orange','purple','brown','pink']
        self.robot_marker_size_m = 0.15
        self.marker_size_robot = determine_marker_size(self.r, self.robot_marker_size_m)
        self.font_size_m = 0.08
        self.font_size = determine_font_size(self.r,self.font_size_m)
        self.line_width = 5
        self.follower_text = np.zeros((self.N))
        self.follower_labels = []
        
        for i in range(self.N):
            self.follower_text[i] = i+1  
            self.follower_labels = np.append(self.follower_labels, [self.r.axes.text(self.x[0,i],self.x[1,i]+0.15,self.follower_text[i],fontsize=self.font_size, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)])
        self.goal_marker_size_m = 0.03
        self.marker_size_goal = determine_marker_size(self.r,self.goal_marker_size_m)
        #Read in and scale image
        gt_img = plt.imread('b1.png')
        x_img = np.linspace(-1.5, 1.5, gt_img.shape[1])
        y_img = np.linspace(-1.5, 1.5, gt_img.shape[0])
        
        gt_img_handle = self.r.axes.imshow(gt_img, extent=(-0.075, 0.075, -0.075, 0.075))
        #self.station_markers = [self.r.axes.scatter(self.battery_station[0], self.battery_station[1], s=self.marker_size_goal, marker='s', facecolors='blue',edgecolors= 'blue',linewidth=self.line_width,zorder=-2)]
        self.r.step()

    def poses_(self):
        self.current_pos = self.r.get_poses()
        self.current_pos_si = self.uni_to_si_states(self.current_pos)
        self.current_x = self.current_pos_si[0]        
        self.current_y = self.current_pos_si[1]
        self.current_location = np.zeros((self.N,2))
        
        for i in range(len(self.current_x)):
            self.current_location[i][0] = self.current_x[i]
            self.current_location[i][1] = self.current_y[i]
        self.velocity = np.zeros((2,self.N))
        
        for i in range(self.N):    
            self.follower_labels[i].set_position([self.current_pos_si[0,i],(self.current_pos_si[1,i])+0.15])
            self.follower_labels[i].set_fontsize(determine_font_size(self.r,self.font_size_m))       
            self.distance_travelled[i] = self.euclidean_distance(self.previous_pose[i], self.current_location[i])
            self.battery[i] = self.battery[i] - self.distance_travelled[i]
            if (self.euclidean_distance(self.battery_station, self.current_location[i]) < 0.05):
                self.battery[i] = 6
        self.previous_pose = self.current_location    
        self.r.step()
    
    
    def bid_generator(self):
        self.battery_after_task = np.zeros((self.N,len(self.task_list)))
        self.bid = np.zeros((self.N,len(self.task_list)))
        for robot in range(self.N):
            for task in range(len(self.task_list)):
                if len(self.task_per_robot[robot])>0:
                    self.battery_after_task[robot][task] = self.battery[robot] - self.euclidean_distance(self.task_list[task],self.task_per_robot[robot][task])
                    if self.battery_after_task[robot][task] < self.battery_threshold[task]:
                        self.bid[robot][task] = self.euclidean_distance(self.battery_station,self.current_location[robot]) + self.euclidean_distance(self.task_list[task],self.battery_station)
                    else:
                        self.bid[robot][task] = self.euclidean_distance(self.task_list[task],self.task_per_robot[robot][task])
                else:
                    self.battery_after_task[robot][task] = self.battery[robot] - self.euclidean_distance(self.task_list[task],self.current_location[robot])
                    if self.battery_after_task[robot][task] < self.battery_threshold[task]:
                        self.bid[robot][task] = self.euclidean_distance(self.battery_station,self.current_location[robot]) + self.euclidean_distance(self.task_list[task],self.battery_station)
                    else:
                        self.bid[robot][task] = self.euclidean_distance(self.task_list[task],self.current_location[robot])
                        
    def task_allocator(self):
        self.task_allocation = np.argmin(self.bid, axis=0)
        for robot in range(self.N):
            for task in range(len(self.task_allocation)):
                if self.task_allocation[task] == robot:
                    if any(all(item in sublist for item in self.task_list[task]) for sublist in self.task_per_robot[robot]):
                        break
                    else:
                        bisect.insort(self.task_per_robot[robot], self.task_list[task])
                        
            
    def threshold_calculator(self):  
        for i in range(len(self.task_list)):
            self.goal_markers = [self.r.axes.scatter(self.task_list[i][0], self.task_list[i][1], s=self.marker_size_goal, marker='s', facecolors='red',edgecolors= 'red',linewidth=self.line_width,zorder=-2)]
        self.battery_threshold = np.zeros(len(self.task_list))
        for task in range(len(self.task_list)):
            self.battery_threshold[task] = self.euclidean_distance(self.task_list[task], self.battery_station)
            
    def list_update(self):
        for robot in range(self.N):
            distance_threshold = 0.05
            count = -1
            if len(self.task_per_robot[robot])>0:
                for task in self.task_per_robot[robot]:
                    count = count + 1
                    distance = self.euclidean_distance(self.current_location[robot],task)
                    if distance < distance_threshold:
                        self.task_per_robot[robot].remove(task)
                    
                    
    def check_task_completion(self):
        for task in range(len(self.tasks)):
            for robot in range(self.N):
                distance = self.euclidean_distance(self.current_location[robot],self.tasks[task])
                if distance < 0.05:
                    self.tasks_completed[task] = True
                    self.goal_markers = [self.r.axes.scatter(self.tasks[task][0], self.tasks[task][1], s=self.marker_size_goal, marker='s', facecolors='green',edgecolors= 'green',linewidth=self.line_width,zorder=-2)]
    
    def controller(self): 
        for robot in range(self.N):
            if self.battery[robot] < self.robo_thresh:
                self.velocity[:,robot] = [(self.battery_station[0] - self.current_x[robot]), (self.battery_station[1] - self.current_y[robot])]
            else:
                for task in range(len(self.task_per_robot[robot])):
                    self.velocity[:,robot] = [(self.task_per_robot[robot][task][0] - self.current_x[robot]), (self.task_per_robot[robot][task][1] - self.current_y[robot])]
        self.dxu = self.si_to_uni_dyn(self.velocity, self.current_pos)
        self.r.set_velocities(np.arange(self.N),self.positional_controller_constant*self.dxu)
        
    
    def main(self):
        if (self.k < (len(self.tasks)-1)):
            self.k = self.k+1
            self.task_list.append(self.tasks[self.k])
            self.threshold_calculator()
            self.bid_generator()
            self.task_allocator()
            self.task_list.remove(self.tasks[self.k])

    def final_project(self):
        self.robo_visual()
        i=-1
        while not all(self.tasks_completed):
            self.poses_()
            if (i%20 == 0):
                self.main()
            self.controller()
            self.list_update()
            self.check_task_completion()
            i=i+1
        self.r.call_at_scripts_end()

t_list= [[1.25, 0.25],[1, 0.5],[1, -0.5],[-1, -0.75],[0.1, 0.2],[0.2, -0.6],[-0.75, -0.1],[-1, 0],[-0.8, -0.25],[1.3, -0.4]]
number_of_robots = 4
Scene1 = task(number_of_robots,t_list)
Scene1.final_project()

