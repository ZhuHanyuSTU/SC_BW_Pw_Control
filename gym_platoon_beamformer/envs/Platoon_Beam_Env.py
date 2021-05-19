'''
author: Zhu Hanyu
email: zhuhy@shanghaitech.edu.cn
date: 08-19-2020
'''

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.constants import *
import cmath
import math,copy
import torch
import time
import scipy.stats as stats
ROAD_TYPE = "LINE" #"LINE" # "CIRCLE"
RSU_DISTANCE = 100 # distance of RSU to the road(meters)

if ROAD_TYPE == "LINE":
    ROAD_LENGTH = 2*RSU_DISTANCE/math.tan(math.pi/6) # length of road coveraged by a RSU
elif ROAD_TYPE == "CIRCLE":
    ROAD_LENGTH = 2*math.pi*RSU_DISTANCE/3 # length of road coveraged by a RSU
else:
    print("Road Type Error!")

ROAD_LEFT_END = -ROAD_LENGTH/2


MIN_BEAMWIDTH = 0.01
MAX_BEAMWIDTH = 0.3

MIN_POWER = 10 #dBm
MAX_POWER = 23 #dBm

MIN_DESIRED_DISTANCE = 2
MAX_DESIRED_DISTANCE = 16


MAX_VELOCITY = 40 # m/s
MIN_VELOCITY = 0

MAX_ACCELERATE = 5 # (m/s^2)
MIN_ACCELERATE = -3

VEHICLE_LENGTH = 4 # vehicle length (meters)
DESIRED_VELOCITY = 25 # desired velocity (m/s) 


MDP_STEP_LENGTH = 0.025 # step length in MDP (seconds)
BASIC_TIME_LENGTH = 0.025 # step length in MDP (seconds)
HIGH_ACTION_LENGTH = 5 # seconds
HIGH_ACTION_INTERVAL = round(HIGH_ACTION_LENGTH/MDP_STEP_LENGTH)


INITIAL_SPACING = 8 # initial inter-vehicle distance (meters)
POS_ERROR_STD_list = [0.008, 0.01,0.012] # std of angle error
THRESHOLD_list = [0,5,10,15] # (dB)
Nlos_num = 2


REWARD_WEIGHT = 0.1
# CACC parameters
CACC_C1 = 0.5
CACC_xi = 1
CACC_omega_n = 0.5
ACT_LAG = 0.02 

ACCELERATE_WEIGHT = BASIC_TIME_LENGTH/(BASIC_TIME_LENGTH+ACT_LAG)


def log_train(log_str, log_file):
    f_log = open(log_file, "a")
    f_log.write(log_str)
    f_log.close()


'''
RSU class
'''


class RSU:

    def __init__(self, antenna_num):
        self.Antenna_Num = antenna_num
        self.position = np.array([0,RSU_DISTANCE,0]) # (x,y,z)
        # self.freq = 28e9  # 28 GHz
        # self.d = 0.5  # relative element space


    def beamformer_vector(self,BW_command,power_command,Est_Norm_DoA):
        # Given the beamwidth of each vehicle, then we determine the number of used antennas and beamforming vector
        desired_AntNum = np.round(4/BW_command) # np.array(vehicle_num)
        user_Num = BW_command.size # scale
        # Based on the beamwidth (required antenna number), determining the magnitude of beamforming vector
        amplitude_vec = np.zeros((self.Antenna_Num, user_Num))
        for veh_id in range(user_Num):
            active_ant_num = int(desired_AntNum[veh_id])
            amplitude_vec[0:active_ant_num,veh_id] = np.sqrt(power_command[veh_id]) *np.ones(active_ant_num)/np.sqrt(active_ant_num) # np.array(antenna_num,vehicle_num)

        BF_vector = amplitude_vec * self.array_response(Est_Norm_DoA) # np.array(antenna_num,vehicle_num)
        return BF_vector

    def array_response(self,Norm_DoA):
        user_Num = Norm_DoA.size
        array_response_matrix = np.exp( -1j*math.pi * np.dot(np.array(range(self.Antenna_Num)).reshape(self.Antenna_Num,1),Norm_DoA.reshape(1,user_Num)))
        return array_response_matrix


    def Channel(self,Norm_DoA,Platoon_pos):
        Dist=np.sqrt(np.sum((self.position)**2)+Platoon_pos**2)
        FSL = 61.4 + 20* np.log10(Dist) + 5.8*np.random.randn(Platoon_pos.size) # db, free space path loss
        channel_loss = 10 ** (-FSL / 10) 
        g_c = np.sqrt(channel_loss)
        h = np.dot(self.array_response(Norm_DoA),np.diag(g_c))  # LOS channel coefficient
        FSL_nlos = 72 + 29.2* np.log10(Dist) + 8.7*np.random.randn(Platoon_pos.size)  # db, free space path loss
        nlos_loss = 10**(-(FSL_nlos)/10)
        for path_idx in range(Nlos_num):
            h = h + np.dot(self.array_response((np.random.rand(Norm_DoA.size)-0.5)*2),np.diag(np.sqrt(nlos_loss)))
        return h

    def Calc_SINR(self, BF_vectors, Norm_DoA,Platoon_pos):
        noise_pw = 10**(-114/10) #-114 
        # Based on the real DoA, generate channel
        H = self.Channel(Norm_DoA,Platoon_pos) # np.array(antenna_num,vehicle_num)
        # calculate the SINR of each user
        receive_comp = abs(np.dot(H.conj().T, BF_vectors))**2
        signal_comp = np.diag(receive_comp)
        interf_comp = np.sum(receive_comp,axis=1)-signal_comp
        user_SINR =  signal_comp/(interf_comp+noise_pw)

        return user_SINR


class Platoon:
    def __init__(self,platoon_size):
        self.platoon_size = platoon_size
        self.platoon_left_end = ROAD_LEFT_END -  (VEHICLE_LENGTH+INITIAL_SPACING)*self.platoon_size/2
        self.position = self.platoon_left_end + (VEHICLE_LENGTH+INITIAL_SPACING)*(self.platoon_size-1-np.array(range(self.platoon_size)))# initial position of vehicles
        self.velocity = DESIRED_VELOCITY*np.ones(self.platoon_size) # initial velocity of vehicles
        self.Veh_accelerate = np.zeros(self.platoon_size)
        self.current_inter_dis = self.position[0:-1]-self.position[1:] - VEHICLE_LENGTH
        self.Veh_idxs = np.array(range(self.platoon_size))
        self.PM_idxs = self.Veh_idxs[1:self.platoon_size] # index of member vehicle
        

    def CACC(self,Spacing_Command, brake_flag):
        # Based on the current state and the given spacing command, 
        # then determining the acceleration of each vehicle and control the change of vehicle state

        vel_diff = self.velocity[1:]-self.velocity[0:-1]
        spacing_diff = Spacing_Command-self.current_inter_dis
        # calculate required accelaration
        des_veh_accelerate = np.zeros(self.platoon_size)
        # determining the acceleration of leader vehicle
        if brake_flag == 1:
            des_veh_accelerate[0] = MIN_ACCELERATE
        else:
            leader_velocity = self.velocity[0]
            if  leader_velocity < DESIRED_VELOCITY:
                des_veh_accelerate[0] = MAX_ACCELERATE
        for veh_id in self.PM_idxs:
            des_veh_accelerate[veh_id] = (1-CACC_C1) * des_veh_accelerate[veh_id-1] + CACC_C1 * des_veh_accelerate[0]\
                -(2*CACC_xi-CACC_C1*(CACC_xi+math.sqrt(CACC_xi**2-1))) * CACC_omega_n * vel_diff[veh_id-1]\
                    -(CACC_xi+math.sqrt(CACC_xi**2-1))* CACC_omega_n*CACC_C1*(self.velocity[veh_id]-self.velocity[0])\
                        -CACC_omega_n**2 * spacing_diff[veh_id-1]
            
        # Based on the current acceleration and the desired acceleration, then determining real acceleration
        self.Veh_accelerate = np.clip(ACCELERATE_WEIGHT * des_veh_accelerate + (1-ACCELERATE_WEIGHT) * self.Veh_accelerate, MIN_ACCELERATE, MAX_ACCELERATE)
        # Based on the real acceleration, updating velocity
        self.velocity = np.clip(self.velocity + self.Veh_accelerate * BASIC_TIME_LENGTH, MIN_VELOCITY,MAX_VELOCITY)
        # Based on the real velocity, updating position
        self.position = self.position + self.velocity * BASIC_TIME_LENGTH
        self.current_inter_dis = self.position[0:-1]-self.position[1:] - VEHICLE_LENGTH
        center_veh_pos = self.position[int(np.round(self.platoon_size/2))]
        if center_veh_pos > abs(ROAD_LEFT_END):
            self.position = self.position-ROAD_LENGTH


''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
PlatoonBeamEnv - Platoon Beamformer Environment

Model Characteristics:
- Considers a platoon including 1+V vehicles. These vehicles are served by RSU with M antennas simultaneouslt.
- Given two parameters: inter-vehicle distance, beamwidth, the model control the platooning and the RSU beamforming. 
'''


class PlatoonBeamEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.Veh_Num = 3
        # low level action space：
        # Dim：Veh_Num*2 
        self.low_action_dim = self.Veh_Num*2 # beamwidth and power
        self.low_obs_dim = self.Veh_Num +2  #  DoA 

        # high level action space：
        # Dim：Veh_Num-1
        # target inter-vehicle distance
        self.high_action_dim = self.Veh_Num-1 # target inter-vehicle distance
        self.high_obs_dim = self.Veh_Num*4-1 + 2 # AoD, velocity, acceleration, current distance

        self.high_action_interval = HIGH_ACTION_INTERVAL
        self.Antenna_Num = 500 
        self.weight = 0
        self.pro_threa = 0.1
        
        self.seed()
    
    def action_sample(self):
        random_action_low = np.clip(np.random.randn(self.low_action_dim),-1,1)
        random_action={"low_action":random_action_low}
        if self.high_level_action == True:
            random_action_high = np.clip(np.random.randn(self.high_action_dim),-1,1)
            random_action["high_action"]=random_action_high
        return random_action

    def cal_AoD(self,RSU_pos,Platoon_pos):
        if ROAD_TYPE == "LINE":
            phy_AoD = np.arcsin(Platoon_pos/np.sqrt(np.sum((RSU_pos)**2)+Platoon_pos**2))
        elif ROAD_TYPE == "CIRCLE":
            phy_AoD = Platoon_pos/RSU_DISTANCE
        else:
            print("Road Type Error!")

        norm_AoD = np.sin(phy_AoD) 
        
        return norm_AoD # return AoD

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self):

        mu, sigma = 0, self.POS_ERROR_STD*100
        lower, upper = mu - 10, mu + 10 
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
        err_AoD = self.cal_AoD(self.RSU.position,self.Platoon.position+ X.rvs(self.Veh_Num))
        obs = copy.deepcopy(self.state)
        obs[0:self.Veh_Num] = np.clip(err_AoD,-1,1) # POS_ERROR_STD*np.random.randn(self.Veh_Num),-1,1)
        obs_dict = {"low_obs":np.concatenate((obs[0:self.Veh_Num],np.array(obs[-2:]))),"high_obs":obs}
        return obs_dict

    def cal_reward_low(self,sinr):
        if ROAD_TYPE=="LINE":
            reward = np.mean(np.min((10*np.log10(sinr))-self.threshold,axis=0)) # 
        elif ROAD_TYPE == "CIRCLE":
            reward = np.mean(np.min((10*np.log10(sinr))-self.threshold,axis=0)) # 
        return reward

    def cal_reward_high(self,sinr_unsatif_pr,inter_distance,weight):
        if ROAD_TYPE=="LINE":
            self.reward_part = - REWARD_WEIGHT*np.mean(inter_distance)
            self.constraint_part = (sinr_unsatif_pr-self.pro_threa)
            if self.constraint_part<=0:
                reward = REWARD_WEIGHT*(20-np.mean(inter_distance))
            else:
                reward = -self.constraint_part
        elif ROAD_TYPE == "CIRCLE":
            self.reward_part = - REWARD_WEIGHT*np.mean(inter_distance)
            self.constraint_part = (sinr_unsatif_pr-self.pro_threa)
            if self.constraint_part<=0:
                reward = REWARD_WEIGHT*(20-np.mean(inter_distance))
            else:
                reward = -self.constraint_part
        return reward


    def step(self, action):


        if self.high_level_action == True:
            self.spacing_command = action["high_action"]*(MAX_DESIRED_DISTANCE-MIN_DESIRED_DISTANCE)/2+(MAX_DESIRED_DISTANCE+MIN_DESIRED_DISTANCE)/2
            self.SINR_rec_pstep_high = []
            self.inter_distance_rec_pstep_high = []
            self.power_rec_pstep_high = []
            self.beamwidth_rec_pstep_high = []
            self.spacing_command_rec_pstep_high = []
        
        beamwidth_command = action["low_action"][0:self.Veh_Num]*(MAX_BEAMWIDTH-MIN_BEAMWIDTH)/2+(MAX_BEAMWIDTH+MIN_BEAMWIDTH)/2 #
        power_command = action["low_action"][self.Veh_Num:2*self.Veh_Num]*(MAX_POWER-MIN_POWER)/2+(MAX_POWER+MIN_POWER)/2

        time_count_pstep = 0
        SINR_rec_pstep = []
        inter_distance_rec_pstep = []
        if self.count %(2*500)==0:
            rnd_num = np.random.uniform()
            if rnd_num<0.2: # brake
                self.brake_flag = 1
        while time_count_pstep < MDP_STEP_LENGTH:
            # 
            Norm_DoA = self.state[0:self.Veh_Num]
            Est_Norm_DoA = self.observation["low_obs"][0:self.Veh_Num] # 
            # 
            BF_vectors = self.RSU.beamformer_vector(beamwidth_command,power_command,Est_Norm_DoA) #np.array(antenna_num,vehicle_num)
            # 
            SINR = self.RSU.Calc_SINR(BF_vectors, Norm_DoA,self.Platoon.position)
            SINR_rec_pstep.append(SINR) #
            self.SINR_rec_pstep_high.append(SINR) #

            self.Platoon.CACC(self.spacing_command, self.brake_flag)
            inter_distance_rec_pstep.append(self.Platoon.current_inter_dis)
            self.inter_distance_rec_pstep_high.append(self.Platoon.current_inter_dis)
            self.power_rec_pstep_high.append(power_command)
            self.beamwidth_rec_pstep_high.append(beamwidth_command)
            self.spacing_command_rec_pstep_high.append(self.spacing_command)

            self.update_state() # update state
            self.observation = self.get_obs()# 
            time_count_pstep = time_count_pstep + BASIC_TIME_LENGTH

        # 
        reward_low = self.cal_reward_low(np.array(SINR_rec_pstep).T)
        reward={"low_reward":reward_low}
        if ((self.count+1) % self.high_action_interval)==0:
            self.unsatisfied_pro = np.max(np.mean((10*np.log10(np.array(self.SINR_rec_pstep_high).T))-self.threshold<0,axis=1),axis=0)
            reward_high = self.cal_reward_high(self.unsatisfied_pro,\
                np.array(self.inter_distance_rec_pstep_high).T, self.weight)
            reward["high_reward"] = reward_high
            reward["high_constraint"] = self.constraint_part
        
        if self.brake_flag ==1:
            self.brake_count += 1
            if self.brake_count>(2*60):
                self.brake_flag = 0
                self.brake_count = 0

        self.count+=1

        if ((self.count%(2*5000))==0) or (np.sum(self.Platoon.current_inter_dis<0)>0 or np.sum(self.Platoon.current_inter_dis>100)>0):
            done = True
        else:
            done = False

        return self.observation, reward, done, {"SINR":np.mean(np.array(SINR_rec_pstep),axis=0),"IV_Dis":np.mean(np.array(inter_distance_rec_pstep),axis=0)}


    def reset(self):
        self.high_level_action = True # flag whether high level action

        self.threshold = np.random.choice(THRESHOLD_list)
        self.POS_ERROR_STD = np.random.choice(POS_ERROR_STD_list)
        
        self.RSU = RSU(self.Antenna_Num)
        self.Platoon = Platoon(self.Veh_Num)
        self.count=0
        self.brake_flag = 0
        self.brake_count = 0
        self.update_state()
        self.observation = self.get_obs()
        return self.observation
    def update_state(self):
        tem_AoD = self.cal_AoD(self.RSU.position,self.Platoon.position) #
        self.state = np.concatenate((tem_AoD,self.Platoon.velocity/MAX_VELOCITY,self.Platoon.Veh_accelerate/MAX_ACCELERATE,self.Platoon.current_inter_dis/20,(np.array([self.threshold]))/20,(np.array([self.POS_ERROR_STD]))/0.015))
    def render(self, mode='human', close=False):
        pass


