import gym
from gym_platoon_beamformer.envs.Platoon_Beam_Env import  PlatoonBeamEnv
from gym.spaces import Box
import argparse
import shutil
import os
import utils
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from sys import platform
import xml.etree.ElementTree as ET
import math
import shutil
import scipy.io as scio

import multiprocessing
import timeit
import operator


ROAD_TYPE = "LINE" #"LINE" # "CIRCLE"
RSU_DISTANCE = 100 # distance of RSU to the road(meters)

if ROAD_TYPE == "LINE":
    ROAD_LENGTH = 2*RSU_DISTANCE/math.tan(math.pi/6) # length of road coveraged by a RSU
elif ROAD_TYPE == "CIRCLE":
    ROAD_LENGTH = 2*math.pi*RSU_DISTANCE/3 # length of road coveraged by a RSU
else:
    print("Road Type Error!")

ROAD_LEFT_END = -ROAD_LENGTH/2
VEHICLE_LENGTH = 4 # vehicle length (meters)


MIN_BEAMWIDTH = 0.01
MAX_BEAMWIDTH = 0.3

MIN_POWER = 10 #dBm
MAX_POWER = 23 #dBm

MIN_DESIRED_DISTANCE = 2
MAX_DESIRED_DISTANCE = 16

seed=1000
curr_dir = os.path.dirname(os.path.realpath(__file__))
exe_name = f"ES_{seed}_{time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))}"
exe_dir = os.path.join(curr_dir,"results","ES_5_15_Ini8_bw0.01_pw1_sc0.5",exe_name)
process = "pal"  #"pal"
print("---------------------------------------")
print(f"Policy: ES")
print("---------------------------------------")

if not os.path.exists(exe_dir):
	os.makedirs(exe_dir)
file_name_sample = os.path.join(exe_dir, "log_sample.txt")


def log_record(log_str, log_file):
    f_log = open(log_file, "a")
    f_log.write(log_str)
    f_log.close()


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

def eval_policy_one_episode(para):
    beamwidth_fix=para[0][0]
    spacing_command_fix=para[0][1]
    seed=para[0][2]
    threshold=para[0][3]
    DOA_ERROR_STD=para[0][4]
    power_fix = para[0][5]
    eval_env = PlatoonBeamEnv()
    eval_env.seed(seed + 100)
    eval_episodes=1
    weigth = eval_env.weight
    low_action =np.zeros(eval_env.Veh_Num*2)
    low_action[0:eval_env.Veh_Num]=(beamwidth_fix*np.ones(eval_env.Veh_Num)-(MAX_BEAMWIDTH+MIN_BEAMWIDTH)/2)/((MAX_BEAMWIDTH-MIN_BEAMWIDTH)/2)
    low_action[eval_env.Veh_Num:eval_env.Veh_Num*2]=(power_fix*np.ones(eval_env.Veh_Num)-(MAX_POWER+MIN_POWER)/2)/((MAX_POWER-MIN_POWER)/2)
    action = {"low_action":low_action}
    action["high_action"] = (spacing_command_fix*np.ones(eval_env.Veh_Num-1)-(MAX_DESIRED_DISTANCE+MIN_DESIRED_DISTANCE)/2)/((MAX_DESIRED_DISTANCE-MIN_DESIRED_DISTANCE)/2)
    INITIAL_SPACING = 8 # spacing_command_fix # initial inter-vehicle distance (meters)

    avg_reward_low = 0.
    avg_reward_high = 0.
    avg_reward_part = 0
    avg_abs_cons_high = 0
    avg_cons_high = 0
    cons_rec = []
    unsatisfied_pro_rec = []
    reward_part_rec = []
    reward_low_rec = []
    beamwidth_rec = []
    power_rec = []
    spacing_command_rec = []
    inter_distance_rec = []
    threshold_rec = []
    DoA_err_STD_rec = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        eval_env.threshold = threshold
        eval_env.DOA_ERROR_STD = DOA_ERROR_STD
        eval_env.Platoon.platoon_left_end = ROAD_LEFT_END -  (VEHICLE_LENGTH+INITIAL_SPACING)*eval_env.Platoon.platoon_size/2
        eval_env.Platoon.position = eval_env.Platoon.platoon_left_end + (VEHICLE_LENGTH+INITIAL_SPACING)*(eval_env.Platoon.platoon_size-1-np.array(range(eval_env.Platoon.platoon_size)))# initial position of vehicles
        eval_env.update_state()
        eval_env.observation = eval_env.get_obs()
        state = eval_env.observation
        while not done:
            state, reward, done, _ = eval_env.step(action)
            avg_reward_low += reward["low_reward"]
            reward_low_rec.append(reward["low_reward"])
                
            if (eval_env.count % eval_env.high_action_interval)==0:
                eval_env.high_level_action = True
                avg_reward_high += reward["high_reward"]
                avg_reward_part += eval_env.reward_part
                avg_abs_cons_high += abs(eval_env.constraint_part)
                avg_cons_high += eval_env.constraint_part
                cons_rec.append(eval_env.constraint_part)
                unsatisfied_pro_rec.append(eval_env.unsatisfied_pro)
                reward_part_rec.append(eval_env.reward_part)
                beamwidth_rec.append(np.mean(np.array(eval_env.beamwidth_rec_pstep_high),axis=0)) 
                power_rec.append(np.mean(np.array(eval_env.power_rec_pstep_high),axis=0))
                spacing_command_rec.append(np.mean(np.array(eval_env.spacing_command_rec_pstep_high),axis=0))
                inter_distance_rec.append(np.mean(np.array(eval_env.inter_distance_rec_pstep_high),axis=0))
                threshold_rec.append(eval_env.threshold)
                DoA_err_STD_rec.append(eval_env.DOA_ERROR_STD)
            else:
                eval_env.high_level_action = False

    avg_reward_low /= eval_episodes
    avg_reward_high /= eval_episodes
    avg_reward_part /= eval_episodes
    avg_abs_cons_high /= eval_episodes
    avg_cons_high /= eval_episodes


    return avg_reward_low, avg_reward_high, avg_abs_cons_high, avg_cons_high, avg_reward_part, weigth, eval_env.count, beamwidth_rec, spacing_command_rec, inter_distance_rec, cons_rec, unsatisfied_pro_rec,reward_part_rec,reward_low_rec,threshold_rec,DoA_err_STD_rec,power_rec


DOA_ERROR_STD_list = [0.008, 0.01,0.012] # std of angle error
THRESHOLD_list = [0, 5,10,15] # (dB)

def eval_policy(beamwidth_fix,spacing_fix,power_fix, seed):
    eval_episodes = 12
    Combin_list = []

    avg_reward_low = 0.
    avg_reward_high = 0.
    avg_reward_part = 0
    avg_abs_cons_high = 0
    avg_cons_high = 0
    count = []
    cons_rec = []
    unsatisfied_pro_rec = []
    reward_part_rec = []
    reward_low_rec = []
    beamwidth_rec = []
    power_rec = []
    spacing_command_rec = []
    inter_distance_rec = []
    threshold_rec = []
    DoA_err_STD_rec = []
    com_idx=0
    for std_idx in range(len(DOA_ERROR_STD_list)):
        for thre_idx in range(len(THRESHOLD_list)):
            Combin_list.append(( [beamwidth_fix,spacing_fix, seed,THRESHOLD_list[thre_idx],DOA_ERROR_STD_list[std_idx],power_fix], ))
            evaluations = eval_policy_one_episode(Combin_list[com_idx])
            avg_reward_low += evaluations[0]
            avg_reward_high += evaluations[1]
            avg_abs_cons_high += evaluations[2]
            avg_cons_high += evaluations[3] 
            avg_reward_part += evaluations[4]
            weigth = evaluations[5]
            count.append(evaluations[6])

            beamwidth_rec.extend(evaluations[7])
            spacing_command_rec.extend(evaluations[8])
            inter_distance_rec.extend(evaluations[9])
            cons_rec.extend(evaluations[10])
            unsatisfied_pro_rec.extend(evaluations[11])
            reward_part_rec.extend(evaluations[12])
            reward_low_rec.extend(evaluations[13])
            threshold_rec.extend(evaluations[14])
            DoA_err_STD_rec.extend(evaluations[15]) 
            power_rec.extend(evaluations[16])
            com_idx += 1

    avg_reward_low /= eval_episodes
    avg_reward_high /= eval_episodes
    avg_reward_part /= eval_episodes
    avg_abs_cons_high /= eval_episodes
    avg_cons_high /= eval_episodes
    print("------------------------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: low {avg_reward_low:.3f} high {avg_reward_high:.3f} reward_part {avg_reward_part:.3f} abs_cons {avg_abs_cons_high:.3f} cons {avg_cons_high:.3f} weight {weigth:.3f}")
    print("------------------------------------------------------")
    return avg_reward_low, avg_reward_high, avg_abs_cons_high, avg_cons_high, avg_reward_part, weigth, count, beamwidth_rec, spacing_command_rec, inter_distance_rec, cons_rec, unsatisfied_pro_rec,reward_part_rec,reward_low_rec,threshold_rec,DoA_err_STD_rec,power_rec

def eval_policy_Fix_pal(combin_command):
    
    evalue = eval_policy(combin_command[1], combin_command[0],combin_command[2],seed=seed)
    return evalue



if __name__ == "__main__":


    total_time = time.time()
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}) 
    beamwidth = np.linspace(0.02,0.15,num=14)
    power = np.linspace(10,23,num=14)
    spacing_command = np.linspace(2,16,num=29)
    print((len(spacing_command)*len(beamwidth)))

    if process =="pal":
        Combin_list = []
        for sc_idx in range(len(spacing_command)):
            for bw_idx in range(len(beamwidth)):
                for pw_idx in range(len(power)):
                    Combin_list.append([spacing_command[sc_idx],beamwidth[bw_idx],power[pw_idx]])

        # revise to parallel
        items = [x for x in Combin_list]
        p = multiprocessing.Pool(18)
        results = p.map(eval_policy_Fix_pal, items)
        p.close()
        p.join()
 

    evaluations_low = []
    evaluations_high = []
    evaluations_abs_cons = []
    evaluations_cons = []
    evaluations_reward_part = []
    evaluations_weight = []
    evaluations_steps = []
    evaluations_BW = []
    evaluations_Pw = []
    evaluations_SC = []
    evaluations_Dist = []

    evaluations_cons_rec = []
    evaluations_unsatisfied_pro_rec = []
    evaluations_reward_part_rec = []
    evaluations_reward_low_rec = []
    evaluations_threshold_rec = []
    evaluations_DoA_err_STD_rec = []

    comb_command_idx = 0
    for sc_idx in range(len(spacing_command)):
        evaluations_low_sc = []
        evaluations_high_sc = []
        evaluations_abs_cons_sc = []
        evaluations_cons_sc = []
        evaluations_reward_part_sc = []
        evaluations_weight_sc = []
        evaluations_steps_sc = []
        evaluations_BW_sc = []
        evaluations_Pw_sc = []
        evaluations_SC_sc = []
        evaluations_Dist_sc = []

        evaluations_cons_rec_sc = []
        evaluations_unsatisfied_pro_rec_sc = []
        evaluations_reward_part_rec_sc = []
        evaluations_reward_low_rec_sc = []
        evaluations_threshold_rec_sc = []
        evaluations_DoA_err_STD_rec_sc = []
        for bw_idx in range(len(beamwidth)):
            evaluations_low_sc_bw = []
            evaluations_high_sc_bw = []
            evaluations_abs_cons_sc_bw = []
            evaluations_cons_sc_bw = []
            evaluations_reward_part_sc_bw = []
            evaluations_weight_sc_bw = []
            evaluations_steps_sc_bw = []
            evaluations_BW_sc_bw = []
            evaluations_Pw_sc_bw = []
            evaluations_SC_sc_bw = []
            evaluations_Dist_sc_bw = []

            evaluations_cons_rec_sc_bw = []
            evaluations_unsatisfied_pro_rec_sc_bw = []
            evaluations_reward_part_rec_sc_bw = []
            evaluations_reward_low_rec_sc_bw = []
            evaluations_threshold_rec_sc_bw = []
            evaluations_DoA_err_STD_rec_sc_bw = []
            for pw_idx in range(len(power)):
                start_time = time.time()
                if process == "pal":
                    evaluations = results[comb_command_idx]
                    comb_command_idx += 1
                else:
                    evaluations = eval_policy(beamwidth[bw_idx], spacing_command[sc_idx],power[pw_idx],seed=seed)

                evaluations_low_sc_bw.append(evaluations[0])
                evaluations_high_sc_bw.append(evaluations[1])
                evaluations_abs_cons_sc_bw.append(evaluations[2])
                evaluations_cons_sc_bw.append(evaluations[3])
                evaluations_reward_part_sc_bw.append(evaluations[4])
                evaluations_weight_sc_bw.append(evaluations[5])
                evaluations_steps_sc_bw.append(evaluations[6])
                evaluations_BW_sc_bw.append(evaluations[7])
                evaluations_SC_sc_bw.append(evaluations[8])
                evaluations_Dist_sc_bw.append(evaluations[9])

                evaluations_cons_rec_sc_bw.append(evaluations[10])
                evaluations_unsatisfied_pro_rec_sc_bw.append(evaluations[11])
                evaluations_reward_part_rec_sc_bw.append(evaluations[12])
                evaluations_reward_low_rec_sc_bw.append(evaluations[13])
                evaluations_threshold_rec_sc_bw.append(evaluations[14])
                evaluations_DoA_err_STD_rec_sc_bw.append(evaluations[15])
                evaluations_Pw_sc_bw.append(evaluations[16])



                max_constraint = np.max(np.array(evaluations[10]))
                max_unsatis_pro = np.max(np.array(evaluations[11]))
                mean_constraint = np.mean(np.array(evaluations[10]))
                mean_unsatis_pro = np.mean(np.array(evaluations[11]))
                eval_ave_BeamWidth = np.mean(np.array(evaluations[7]),axis=0)
                eval_ave_spacing = np.mean(np.array(evaluations[8]),axis=0)
                eval_ave_dist = np.mean(np.array(evaluations[9]),axis=0)
                end_time = time.time()
                time_len = end_time - start_time
                sample_str = f"sc: {spacing_command[sc_idx]:.4f}, bw: {beamwidth[bw_idx]:.4f}, pw: {power[pw_idx]:.4f}, low {evaluations[0]:.3f} high {evaluations[1]:.3f} \
BW {eval_ave_BeamWidth} SC {eval_ave_spacing} Dist {eval_ave_dist} reward_part {evaluations[4]:.3f}, max_unsatis_pro: {max_unsatis_pro:.4f}, \
mean_unsatis_pro: {mean_unsatis_pro:.4f}, max_constraint: {max_constraint:.4f}, mean_constraint: {mean_constraint:.4f}, time: {time_len:.5f} \n"
                log_record(sample_str, file_name_sample) 

            evaluations_low_sc.append(evaluations_low_sc_bw)
            evaluations_high_sc.append(evaluations_high_sc_bw)
            evaluations_abs_cons_sc.append(evaluations_abs_cons_sc_bw)
            evaluations_cons_sc.append(evaluations_cons_sc_bw)
            evaluations_reward_part_sc.append(evaluations_reward_part_sc_bw)
            evaluations_weight_sc.append(evaluations_weight_sc_bw)
            evaluations_steps_sc.append(evaluations_steps_sc_bw)
            evaluations_BW_sc.append(evaluations_BW_sc_bw)
            evaluations_Pw_sc.append(evaluations_Pw_sc_bw)
            evaluations_SC_sc.append(evaluations_SC_sc_bw)
            evaluations_Dist_sc.append(evaluations_Dist_sc_bw)

            evaluations_cons_rec_sc.append(evaluations_cons_rec_sc_bw)
            evaluations_unsatisfied_pro_rec_sc.append(evaluations_unsatisfied_pro_rec_sc_bw)
            evaluations_reward_part_rec_sc.append(evaluations_reward_part_rec_sc_bw)
            evaluations_reward_low_rec_sc.append(evaluations_reward_low_rec_sc_bw)
            evaluations_threshold_rec_sc.append(evaluations_threshold_rec_sc_bw)
            evaluations_DoA_err_STD_rec_sc.append(evaluations_DoA_err_STD_rec_sc_bw)


        evaluations_low.append(evaluations_low_sc)
        evaluations_high.append(evaluations_high_sc)
        evaluations_abs_cons.append(evaluations_abs_cons_sc)
        evaluations_cons.append(evaluations_cons_sc)
        evaluations_reward_part.append(evaluations_reward_part_sc)
        evaluations_weight.append(evaluations_weight_sc)
        evaluations_steps.append(evaluations_steps_sc)
        evaluations_BW.append(evaluations_BW_sc)
        evaluations_Pw.append(evaluations_Pw_sc)
        evaluations_SC.append(evaluations_SC_sc)
        evaluations_Dist.append(evaluations_Dist_sc)

        evaluations_cons_rec.append(evaluations_cons_rec_sc)
        evaluations_unsatisfied_pro_rec.append(evaluations_unsatisfied_pro_rec_sc)
        evaluations_reward_part_rec.append(evaluations_reward_part_rec_sc)
        evaluations_reward_low_rec.append(evaluations_reward_low_rec_sc)
        evaluations_threshold_rec.append(evaluations_threshold_rec_sc)
        evaluations_DoA_err_STD_rec.append(evaluations_DoA_err_STD_rec_sc)
                
    np.save(exe_dir+"/evaluations_low", evaluations_low)
    np.save(exe_dir+"/evaluations_high", evaluations_high)
    np.save(exe_dir+"/evaluations_abs_cons", evaluations_abs_cons)
    np.save(exe_dir+"/evaluations_cons", evaluations_cons)
    np.save(exe_dir+"/evaluations_reward_part", evaluations_reward_part)
    np.save(exe_dir+"/evaluations_weight", evaluations_weight)
    np.save(exe_dir+"/evaluations_steps", evaluations_steps)
    np.save(exe_dir+"/evaluations_BW", evaluations_BW)
    np.save(exe_dir+"/evaluations_Pw", evaluations_Pw)
    np.save(exe_dir+"/evaluations_SC", evaluations_SC)
    np.save(exe_dir+"/evaluations_Dist", evaluations_Dist)
    np.save(exe_dir+"/evaluations_cons_rec", evaluations_cons_rec)
    np.save(exe_dir+"/evaluations_unsatisfied_pro_rec", evaluations_unsatisfied_pro_rec)
    np.save(exe_dir+"/evaluations_reward_part_rec", evaluations_reward_part_rec)
    np.save(exe_dir+"/evaluations_reward_low_rec", evaluations_reward_low_rec)
    np.save(exe_dir+"/evaluations_threshold_rec", evaluations_threshold_rec)
    np.save(exe_dir+"/evaluations_DoA_err_STD_rec", evaluations_DoA_err_STD_rec)
    print("total_time:",time.time()-total_time)
             