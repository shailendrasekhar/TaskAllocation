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
import time

class Scene:
    def __init__(self,robots,tasks):
        #Initial variables
        self.tasks = tasks
        self.N = robots
        #Robotarium variables
        self.si_to_uni_dyn, self.uni_to_si_states = create_si_to_uni_mapping()
        #Boundaries
        self.x_min_robotarium = -1.5
        self.x_max_robotarium = 1.5
        self.y_min_robotarium = -1
        self.y_max_robotarium = 1
        self.res = 0.1
        #User-defined variables
        self.positional_controller_constant=0.9
        self.safety_radius = 0.04
        self.tmp_variable = [10]*self.N
        self.battery = [6,7,6,9,7,5,6,4,3,6]
        self.battery_station = [0,0]
        #Variables for adding a task momentarily
        self.k=-1
        self.tasks_available = []
        #Variables for List-Update
        self.task_per_robot = [[] for _ in range(self.N)]
        self.battery_after_task = [[] for _ in range(self.N)]
        self.bid_per_task = [[] for _ in range(self.N)]
        self.battery_threshold = 2
        self.current_task = [[] for _ in range(self.N)]
        self.n_tasks = np.zeros((self.N))
        self.robot_status = [-1 for _ in range(self.N)]
        #Variable for Winner_Update
        self.bid_list = [[] for _ in range(self.N)]
        self.trade_list = [[[] for _ in range(len(self.tasks) )] for _ in range(self.N)]
        #Variable for task_trade
        self.j_star = [-1 for _ in range(self.N)]
        self.current_trade = [[] for _ in range(self.N)]
        self.indices = [[] for _ in range(self.N)]
        self.completed_tasks =[] 
    
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
        self.goal_marker_size_m = 0.05
        self.marker_size_goal = determine_marker_size(self.r,self.goal_marker_size_m)
        gt_img = plt.imread('b1.png')
        x_img = np.linspace(-1.5, 1.5, gt_img.shape[1])
        y_img = np.linspace(-1.5, 1.5, gt_img.shape[0])
        gt_img_handle = self.r.axes.imshow(gt_img, extent=(-0.075, 0.075, -0.075, 0.075))        
        #self.station_markers = [self.r.axes.scatter(self.battery_station[0], self.battery_station[1], s=self.marker_size_goal, marker='x', facecolors='blue',edgecolors= 'blue',linewidth=self.line_width,zorder=-2)]
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
    
    def euclidean_distance(self,p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
    def threshold_calculator(self,task):  
        return(self.euclidean_distance(task, self.battery_station))
            
    def add_task(self):
        if (self.k < (len(self.tasks)-1)):
            self.k = self.k + 1
            self.goal_markers = [self.r.axes.scatter(self.tasks[self.k][0], self.tasks[self.k][1], s=self.marker_size_goal, marker='s', facecolors='red',edgecolors= 'red',linewidth=self.line_width,zorder=-2)]
            d = np.zeros((self.N))
            for robot in range(self.N):
                    d[robot] = self.euclidean_distance(self.current_location[robot], self.tasks[self.k])
            win_robot = np.argmin(d)
            self.bid_list[win_robot].append([self.k,win_robot,10,True]) 
              
    def winner_update(self, robot):
        if len(self.bid_list[robot])>0: 
            for n in range(len(self.bid_list[robot])):
                j, k, b, auc = self.bid_list[robot][n]
                if j not in self.completed_tasks:
                    if auc==True and (self.trade_list[robot][j] ==[]):
                        bid, battery_after_task, priority = self.bid_generator(robot, self.tasks[j])
                        if robot == k:
                            self.trade_list[robot][j] = [robot, bid, battery_after_task, priority, True]
                        elif bid < b:
                            self.trade_list[robot][j] = [robot, bid, battery_after_task, priority, False]
                        else:
                            self.trade_list[robot][j] = [k, b, 0, 0, False]
                    elif (self.trade_list[robot][j] !=[]) and (b < self.trade_list[robot][j][1]):
                        self.trade_list[robot][j] = [k, b, 0, 0, self.trade_list[robot][j][4] ]
            self.bid_list[robot] = []
 
    def bid_generator(self,robot,new_task):
        threshold = self.threshold_calculator(new_task)
        bid=10
        battery_after_task = 10
        priority = 10
        if len(self.task_per_robot[robot])<1:
            distance = self.euclidean_distance(new_task, self.current_location[robot])
            if (self.battery[robot] - distance) < threshold:
                bid = self.euclidean_distance(self.battery_station,self.current_location[robot]) + self.euclidean_distance(new_task,self.battery_station)
                battery_after_task = 5 - self.euclidean_distance(new_task,self.battery_station)
            else:
                bid = self.euclidean_distance(new_task,self.current_location[robot])
                battery_after_task = self.battery[robot]- self.euclidean_distance(new_task,self.current_location[robot])
            priority = 0
        else:
            for task in range(len(self.task_per_robot[robot])):
                distance = self.euclidean_distance(new_task, self.tasks[(self.task_per_robot[robot][task])])
                if (self.battery[robot] - distance) < threshold:
                    cost = self.euclidean_distance(self.battery_station,self.tasks[self.task_per_robot[robot][task]]) + self.euclidean_distance(new_task,self.battery_station)
                    battery_after_task = 5 - self.euclidean_distance(new_task,self.battery_station)
                else:
                    cost = self.euclidean_distance(self.tasks[self.task_per_robot[robot][task]], new_task)
                    battery_after_task = self.battery_after_task[robot][task] - self.euclidean_distance(new_task,self.tasks[self.task_per_robot[robot][task]])
                if cost < self.bid_per_task[robot][task]:
                    priority = task
                    bid = cost
                    break
        return bid,battery_after_task,priority
        
    def task_trade(self,robot):
        self.current_trade[robot] = [] 
        if self.j_star[robot] == -1:
            min_bid=0
            j_star_candidates = [(j, qi_j[1]) for j, qi_j in enumerate(self.trade_list[robot]) if len(qi_j) > 0 and qi_j[0] == robot]
            if len(j_star_candidates)>0:
                self.j_star[robot] = min(j_star_candidates, key=lambda x: x[1])[0]
                min_bid = min(j_star_candidates, key=lambda x: x[1])[1]
                Ztmp = [self.j_star[robot], robot, min_bid, self.trade_list[robot][self.j_star[robot]][4]]
                for m in topological_neighbors(self.L, robot):
                    self.bid_list[m].append(Ztmp)
                self.tmp_variable = 0
        else:
            if self.trade_list[robot][self.j_star[robot]][0] == robot:
                self.current_trade[robot].append(self.trade_list[robot][self.j_star[robot]])
            if self.trade_list[robot][self.j_star[robot]][4]:
                Ztmp = [self.j_star[robot], self.trade_list[robot][self.j_star[robot]][0], self.trade_list[robot][self.j_star[robot]][1], False]
                for m in topological_neighbors(self.L, robot):
                    self.bid_list[m].append(Ztmp)
            self.trade_list[robot][self.j_star[robot]] = []
            self.indices[robot] = self.j_star[robot]
            self.j_star[robot] = -1

        
            
    def list_update(self,robot):
        if len(self.current_trade[robot]) >  0:
            for i in range(len(self.current_trade[robot])):
                priority = self.current_trade[robot][i][3]
                task = self.indices[robot]
                bid = self.current_trade[robot][i][1]
                battery_after_task = self.current_trade[robot][0][2]
                z_temp = [task,robot,10,True]
                self.task_per_robot[robot]=self.task_per_robot[robot][:priority-1]
                self.bid_per_task[robot]=self.bid_per_task[robot][:priority-1]
                self.battery_after_task[robot]=self.battery_after_task[robot][:priority-1]
                self.bid_list[robot].append(z_temp)
                self.task_per_robot[robot].append(task) 
                self.bid_per_task[robot].append(bid)
                self.battery_after_task[robot].append(battery_after_task)
                self.task_per_robot[robot] = list(set(self.task_per_robot[robot]))
                self.bid_per_task[robot] = list(set(self.bid_per_task[robot]))
                self.battery_after_task[robot] = list(set(self.battery_after_task[robot]))
        if len(self.task_per_robot[robot])>0 and (self.battery[robot]>self.battery_threshold):
            if self.current_task[robot] == []:
                self.current_task[robot] = self.task_per_robot[robot][0]
                self.n_tasks[robot] = self.n_tasks[robot] + 1
                self.robot_status[robot] = 0
            elif self.euclidean_distance(self.current_location[robot], self.tasks[self.current_task[robot]])<0.05:
                self.task_per_robot[robot] = self.task_per_robot[robot][1:]
                self.bid_per_task[robot] = self.bid_per_task[robot][1:]
                self.battery_after_task[robot] = self.battery_after_task[robot][1:]
                self.current_task[robot]= []
        elif (self.battery[robot]<self.battery_threshold) and (len(self.task_per_robot[robot])>0):
            self.current_task[robot] = []
            self.robot_status[robot] = 1
            for task in self.task_per_robot[robot]:
                for m in topological_neighbors(self.L,robot):
                    self.bid_list[robot].append([task,robot,5,True])
                    self.bid_list[m].append([task,robot,5,True])
            self.task_per_robot[robot] = []
            self.bid_per_task[robot] = []
            self.battery_after_task[robot] = [] 
        
    def controller(self):
        for robot in range(self.N):
            if self.robot_status[robot] == 0:
                if self.current_task[robot] != []:
                    self.velocity[:,robot] = [(self.tasks[self.current_task[robot]][0] - self.current_x[robot]), (self.tasks[self.current_task[robot]][1] - self.current_y[robot])]
            elif self.robot_status[robot] == 1:
                self.velocity[:,robot] = [(self.battery_station[0] - self.current_x[robot]), (self.battery_station[1] - self.current_y[robot])]
        self.dxu = self.si_to_uni_dyn(self.velocity, self.current_pos)
        self.r.set_velocities(np.arange(self.N),self.positional_controller_constant*self.dxu)
        
    def check_task_completion(self):
        for task in range(len(self.tasks)):
            for robot in range(self.N):
                distance = self.euclidean_distance(self.current_location[robot],self.tasks[task])
                if distance < 0.05:
                    self.tasks_completed[task] = True
                    self.completed_tasks.append(task)
                    self.goal_markers = [self.r.axes.scatter(self.tasks[task][0], self.tasks[task][1], s=self.marker_size_goal, marker='s', facecolors='green',edgecolors= 'green',linewidth=self.line_width,zorder=-2)]

        
    def final_project(self):
        self.robo_visual()
        self.tasks_completed = [False] * len(self.tasks)
        self.count = -1
        while not all(self.tasks_completed):    
            self.poses_()
            if self.count % 20 == 0:
                self.add_task()
            for robot in range(self.N):
                self.winner_update(robot)
                self.task_trade(robot)
                if self.tmp_variable == 0:
                    self.tmp_variable = 1
                    break
                self.list_update(robot)
            self.controller()
            self.check_task_completion()
            #print("Task_per_robot:",self.task_per_robot)
            #print("current_task:",self.current_task)
            #print(self.battery)
            #print(self.robot_status)
            self.count = self.count + 1
            #print(" ")
        self.r.call_at_scripts_end()

tasks= [[1.5, 0.25],[1, 0.5],[1, -0.5],[-1, -0.75],[0.1, 0.2],[0.2, -0.6],[-0.75, -0.1],[-1, 0],[-0.8, -0.25],[1.3, -0.4]]
robots = 4
Scene1 = Scene(robots,tasks)
Scene1.final_project()