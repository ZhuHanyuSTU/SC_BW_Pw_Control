import gym
from gym_platoon_beamformer.envs.Platoon_Beam_Env import  PlatoonBeamEnv
from gym.spaces import Box
import SAC_alg
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

import shutil
import scipy.io as scio
import multiprocessing
import timeit
import operator
def log_record(log_str, log_file):
    f_log = open(log_file, "a")
    f_log.write(log_str)
    f_log.close()

MIN_DESIRED_DISTANCE = 2
MAX_DESIRED_DISTANCE = 16
POS_ERROR_STD_list = [0.008, 0.01,0.012] # std of position error
THRESHOLD_list = [0, 5,10,15] # (dB)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy_one_episode(para):
    policy_low=para[0][0]
    policy_high=para[0][1]
    seed=para[0][2]
    threshold=para[0][3]
    POS_ERROR_STD=para[0][4]
    weigth=0
    eval_env = PlatoonBeamEnv()
    eval_env.seed(seed + 100)
    eval_episodes=1
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
        eval_env.POS_ERROR_STD = POS_ERROR_STD
        eval_env.update_state()
        eval_env.observation = eval_env.get_obs()
        state = eval_env.observation
        while not done:
            action_low = policy_low.select_action(np.array(state["low_obs"]),evaluate=True)
            action={"low_action":action_low}
            if eval_env.high_level_action == True:
                action_high = policy_high.select_action(np.array(state["high_obs"]),evaluate=True)
                action["high_action"] = action_high
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
                DoA_err_STD_rec.append(eval_env.POS_ERROR_STD)
            else:
                eval_env.high_level_action = False

    avg_reward_low /= eval_episodes
    avg_reward_high /= eval_episodes
    avg_reward_part /= eval_episodes
    avg_abs_cons_high /= eval_episodes
    avg_cons_high /= eval_episodes

    print("------------------------------------------------------")
    print(f"Threshold {threshold:.1f}, ERROR_STD {POS_ERROR_STD:.3f}: low {avg_reward_low:.3f} high {avg_reward_high:.3f} reward_part {avg_reward_part:.3f} abs_cons {avg_abs_cons_high:.3f} cons {avg_cons_high:.3f}")
    print("------------------------------------------------------")
    return avg_reward_low, avg_reward_high, avg_abs_cons_high, avg_cons_high, avg_reward_part, weigth, eval_env.count, beamwidth_rec, spacing_command_rec, inter_distance_rec, cons_rec, unsatisfied_pro_rec,reward_part_rec,reward_low_rec,threshold_rec,DoA_err_STD_rec,power_rec


def eval_policy(policy_low,policy_high, seed):
    eval_episodes = 12
    Combin_list = []
    for std_idx in range(len(POS_ERROR_STD_list)):
        for thre_idx in range(len(THRESHOLD_list)):
            Combin_list.append(( [policy_low, policy_high, seed,THRESHOLD_list[thre_idx],POS_ERROR_STD_list[std_idx]], ))

    # revise to parallel
    items = [x for x in Combin_list]
    p = multiprocessing.Pool(eval_episodes)
    results = p.map(eval_policy_one_episode, items)
    p.close()
    p.join()

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

    for idx in range(eval_episodes):
        evaluations = results[idx]
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

    avg_reward_low /= eval_episodes
    avg_reward_high /= eval_episodes
    avg_reward_part /= eval_episodes
    avg_abs_cons_high /= eval_episodes
    avg_cons_high /= eval_episodes

    return avg_reward_low, avg_reward_high, avg_abs_cons_high, avg_cons_high, avg_reward_part, weigth, count, beamwidth_rec, spacing_command_rec, inter_distance_rec, cons_rec, unsatisfied_pro_rec,reward_part_rec,reward_low_rec,threshold_rec,DoA_err_STD_rec,power_rec

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                  # Policy name
    parser.add_argument("--seed", default=1000, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=20e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=20e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e7, type=int)   # Max time steps to run environment
    parser.add_argument("--alpha", default=0.2)                # Std of Gaussian exploration noise
    parser.add_argument("--final_scale", default=0.1)
    parser.add_argument("--batch_size", default=64, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--discount_low", default=0.0)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    start_timesteps_high = args.start_timesteps*10
    start_timesteps_low = args.start_timesteps
    end_timesteps = int(args.max_timesteps/5)
    low_train_interval = 20
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    buffer_size = 2e4

    exe_name = f"{args.policy}_MaxStep{args.max_timesteps}_{args.seed}\
_{time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))}"
    exe_dir = os.path.join(curr_dir,"results","TS_5_15_DySCBWPw_SAC_buffer2e4",exe_name)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Seed: {args.seed}")
    print("---------------------------------------")
    modle_dir = os.path.join(exe_dir,"models")
    eval_dir = os.path.join(exe_dir,"eval")
    if not os.path.exists(exe_dir):
        os.makedirs(exe_dir)
    args.save_model = True
    if args.save_model and not os.path.exists(modle_dir):
        os.makedirs(modle_dir)
    file_name_reward = os.path.join(exe_dir, "log_epi_reward.txt")
    file_name_step = os.path.join(exe_dir, "log_step.txt")
    file_name_eval = os.path.join(exe_dir, "log_eval.txt")
    

    shutil.copy(
        os.path.join(curr_dir, 'main_SAC.py'),
        os.path.join(exe_dir, 'main_SAC.py'))
    shutil.copy(
        os.path.join(curr_dir,'gym_platoon_beamformer/envs', 'Platoon_Beam_Env.py'),
        os.path.join(exe_dir, 'Platoon_Beam_Env.py'))

    env = PlatoonBeamEnv()

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim_low = env.low_obs_dim
    action_dim_low = env.low_action_dim
    max_action = 1

    state_dim_high = env.high_obs_dim
    action_dim_high = env.high_action_dim

    action_space_low = np.zeros((action_dim_low,2))
    for ac_idx in range(action_dim_low):
        action_space_low[ac_idx, 0] = -1
        action_space_low[ac_idx, 1] = 1
    policy_low = SAC_alg.SAC(state_dim_low,action_space_low,args.discount_low,args.alpha,args.tau)

    action_space = np.zeros((action_dim_high,2))
    for ac_idx in range(action_dim_high):
        action_space[ac_idx, 0] = -1
        action_space[ac_idx, 1] = 1
    policy_high = SAC_alg.SAC(state_dim_high,action_space,args.discount,args.alpha,args.tau)


    replay_buffer_low = utils.ReplayBuffer(state_dim_low, action_dim_low)
    replay_buffer_high = utils.ReplayBuffer(state_dim_high, action_dim_high,max_size=int(buffer_size))

    # Evaluate untrained policy
    t1=time.time()
    eval_count = 0
    evaluations = eval_policy(policy_low, policy_high, args.seed)
    print("time:",time.time()-t1)

    eval_ave_BeamWidth = np.mean(np.array(evaluations[7]),axis=0)
    eval_ave_spacing = np.mean(np.array(evaluations[8]),axis=0)
    eval_ave_dist = np.mean(np.array(evaluations[9]),axis=0)
    max_constraint = np.max(np.array(evaluations[10]))
    eval_ave_Power = np.mean(np.array(evaluations[16]),axis=0)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    eval_str = f"Evaluation over {1} episodes {np.mean(evaluations[6])} steps: low {evaluations[0]:.3f} high {evaluations[1]:.3f} \
BW {eval_ave_BeamWidth} Pw{eval_ave_Power} SC {eval_ave_spacing} Dist {eval_ave_dist} reward_part {evaluations[4]:.3f} \
abs_cons {evaluations[2]:.3f} cons {evaluations[3]:.3f} max_cons {max_constraint:.3f}\n"
    log_record(eval_str, file_name_eval)
    evaluations_low = [evaluations[0]]
    evaluations_high = [evaluations[1]]
    evaluations_abs_cons = [evaluations[2]]
    evaluations_cons = [evaluations[3]]
    evaluations_reward_part = [evaluations[4]]
    evaluations_weight = [evaluations[5]]
    evaluations_steps = [evaluations[6]]
    evaluations_BW = [evaluations[7]]
    evaluations_SC = [evaluations[8]]
    evaluations_Dist = [evaluations[9]]

    evaluations_cons_rec = [evaluations[10]]
    evaluations_unsatisfied_pro_rec = [evaluations[11]]
    evaluations_reward_part_rec = [evaluations[12]]
    evaluations_reward_low_rec = [evaluations[13]]
    evaluations_threshold_rec = [evaluations[14]]
    evaluations_DoA_err_STD_rec = [evaluations[15]]
    evaluations_Pw = [evaluations[16]]

    state, done = env.reset(), False # state 
    episode_reward_low = 0 
    episode_timesteps_low = 0
    episode_reward_high = 0 
    episode_reward_part = 0 
    episode_abs_cons_high = 0
    episode_cons_high = 0
    episode_timesteps_high = 0
    episode_num = 0
    high_level_step = 0
    t_high = 0
    start_time = time.time()
    ini_scale=1
    for t in range(int(args.max_timesteps)):
        # print("t:",t)
        episode_timesteps_low += 1
        if env.high_level_action == True:
            episode_timesteps_high += 1
            ave_BeamWidth = np.zeros(env.Veh_Num)
            ave_power = np.zeros(env.Veh_Num)
            ave_spacing = np.zeros(env.Veh_Num-1)
            ave_dist = np.zeros(env.Veh_Num-1)
        # Select action randomly or according to policy
        
        if t < args.start_timesteps:
            action = env.action_sample()
        else:
            if t>end_timesteps:
                start_timesteps_low = end_timesteps
                start_timesteps_high = end_timesteps
                end_timesteps += int(args.max_timesteps/5)
                ini_scale = ini_scale*0.5
            scale_low = (ini_scale-args.final_scale)/(start_timesteps_low-end_timesteps)*t+(ini_scale-((ini_scale-args.final_scale)/(start_timesteps_low-end_timesteps)*start_timesteps_low))
            action_low = policy_low.select_action(np.array(state["low_obs"])).clip(-max_action, max_action)
            action={"low_action":action_low}
            if env.high_level_action == True:
                if t > start_timesteps_high:
                    scale_high = (ini_scale-args.final_scale)/(start_timesteps_high-end_timesteps)*t+(ini_scale-((ini_scale-args.final_scale)/(start_timesteps_high-end_timesteps)*start_timesteps_high))
                    origi_action = policy_high.select_action(np.array(state["high_obs"]))
                    action_high = origi_action.clip(-max_action, max_action)
                else:
                    action_high = env.action_sample()["high_action"]
                action["high_action"] = action_high
        if env.high_level_action == True:
            pre_state_high = state["high_obs"]
            pre_action_high = action["high_action"]
        # Perform action
        next_state, reward, done, _ = env.step(action) 
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer_low.add(state["low_obs"], action["low_action"], next_state["low_obs"], reward["low_reward"], done)

        state = next_state
        episode_reward_low += reward["low_reward"]
   

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            if t % low_train_interval ==0:
                policy_low.train(replay_buffer_low, args.batch_size)
        
        if env.high_level_action == True:
            if t >= start_timesteps_high: # 
                policy_high.train(replay_buffer_high, args.batch_size)
            t_high += 1



        if (env.count % env.high_action_interval)==0:
            ave_BeamWidth += np.mean(np.array(env.beamwidth_rec_pstep_high),axis=0)
            ave_power += np.mean(np.array(env.power_rec_pstep_high),axis=0)
            ave_spacing += np.mean(np.array(env.spacing_command_rec_pstep_high),axis=0)
            ave_dist += np.mean(np.array(env.inter_distance_rec_pstep_high),axis=0)
            env.high_level_action = True # means the high level action will be executed in next slot, so store the current state for last high level action
            if t>0:
                new_dis = state["high_obs"][-2:]
                high_r = reward["high_reward"]
                step_str = f"{env.count} Pre_dis: {pre_state_high[-2]:.1f},{pre_state_high[-1]:.1f} Action: {pre_action_high[0]:.1f},\
{pre_action_high[1]:.1f} New_dis: {new_dis[0]:.1f},{new_dis[1]:.1f} Reward: {high_r:.1f} Reward_part: {env.reward_part:.3f} Constraint: {env.constraint_part:.3f} Weight: {env.weight:.3f} \n"
                log_record(step_str, file_name_step)
                episode_reward_high += reward["high_reward"]
                episode_reward_part += env.reward_part
                episode_abs_cons_high += abs(env.constraint_part)
                episode_cons_high += env.constraint_part
                replay_buffer_high.add(pre_state_high, pre_action_high, state["high_obs"], reward["high_reward"], done)
        else:
            env.high_level_action = False
        if done==True: 
            stop_time = time.time()
            ave_BeamWidth /= 50
            ave_power /= 50
            ave_spacing /= 50
            ave_dist /= 50
            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
            rec_str = f"Total T: {t+1} Episode Num: {episode_num+1} STD: {env.POS_ERROR_STD} threshold: {env.threshold} Episode T_low: {episode_timesteps_low} \
Reward_low: {episode_reward_low:.3f} Episode T_high: {episode_timesteps_high} Reward_high: {episode_reward_high:.3f} \
Reward_part: {episode_reward_part:.3f} abs_Constraint: {episode_abs_cons_high:.3f} Constraint: {episode_cons_high:.3f} \
Beamwdith: {ave_BeamWidth} Power: {ave_power} \
SpacingCommand: {ave_spacing} InterDistance: \
{ave_dist} RunTime: {(stop_time-start_time):.1f}\n"
            log_record(rec_str, file_name_reward)
            # Reset environment
            state, done = env.reset(), False
            start_time = time.time()
            episode_reward_low = 0
            episode_timesteps_low = 0
            episode_reward_high = 0
            episode_reward_part = 0
            episode_abs_cons_high = 0
            episode_cons_high = 0
            episode_timesteps_high = 0
            episode_num += 1 
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_count += 1
            if args.save_model: policy_low.save_model("PBENV",suffix="",actor_path=modle_dir+f"/models_low_actor{t}",critic_path=modle_dir+f"/models_low_critic{eval_count}")
            if args.save_model: policy_high.save_model("PBENV",suffix="",actor_path=modle_dir+f"/models_high_actor{t}",critic_path=modle_dir+f"/models_high_critic{eval_count}")
            evaluations=eval_policy(policy_low,policy_high, args.seed)
            eval_ave_BeamWidth = np.mean(np.array(evaluations[7]),axis=0)
            eval_ave_spacing = np.mean(np.array(evaluations[8]),axis=0)
            eval_ave_dist = np.mean(np.array(evaluations[9]),axis=0)
            max_constraint = np.max(np.array(evaluations[10]))
            eval_ave_Power = np.mean(np.array(evaluations[16]),axis=0)
            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
            eval_str = f"Evaluation over {1} episodes {np.mean(evaluations[6])} steps: low {evaluations[0]:.3f} high {evaluations[1]:.3f} \
BW {eval_ave_BeamWidth} Pw {eval_ave_Power} SC {eval_ave_spacing} Dist {eval_ave_dist} reward_part {evaluations[4]:.3f} \
abs_cons {evaluations[2]:.3f} cons {evaluations[3]:.3f} max_cons {max_constraint:.3f}\n"
            log_record(eval_str, file_name_eval)

            evaluations_low.append(evaluations[0])
            evaluations_high.append(evaluations[1])
            evaluations_abs_cons.append(evaluations[2])
            evaluations_cons.append(evaluations[3])
            evaluations_reward_part.append(evaluations[4])
            evaluations_weight.append(evaluations[5])
            evaluations_steps.append(evaluations[6])
            evaluations_BW.append(evaluations[7])
            evaluations_SC.append(evaluations[8])
            evaluations_Dist.append(evaluations[9])

            evaluations_cons_rec.append(evaluations[10])
            evaluations_unsatisfied_pro_rec.append(evaluations[11])
            evaluations_reward_part_rec.append(evaluations[12])
            evaluations_reward_low_rec.append(evaluations[13])
            evaluations_threshold_rec.append(evaluations[14])
            evaluations_DoA_err_STD_rec.append(evaluations[15])
            evaluations_Pw.append(evaluations[16])

            np.save(exe_dir+"/evaluations_low", evaluations_low)
            np.save(exe_dir+"/evaluations_high", evaluations_high)
            np.save(exe_dir+"/evaluations_abs_cons", evaluations_abs_cons)
            np.save(exe_dir+"/evaluations_cons", evaluations_cons)
            np.save(exe_dir+"/evaluations_reward_part", evaluations_reward_part)
            np.save(exe_dir+"/evaluations_weight", evaluations_weight)
            np.save(exe_dir+"/evaluations_steps", evaluations_steps)
            np.save(exe_dir+"/evaluations_BW", evaluations_BW)
            np.save(exe_dir+"/evaluations_SC", evaluations_SC)
            np.save(exe_dir+"/evaluations_Dist", evaluations_Dist)
            np.save(exe_dir+"/evaluations_cons_rec", evaluations_cons_rec)
            np.save(exe_dir+"/evaluations_unsatisfied_pro_rec", evaluations_unsatisfied_pro_rec)
            np.save(exe_dir+"/evaluations_reward_part_rec", evaluations_reward_part_rec)
            np.save(exe_dir+"/evaluations_reward_low_rec", evaluations_reward_low_rec)
            np.save(exe_dir+"/evaluations_threshold_rec", evaluations_threshold_rec)
            np.save(exe_dir+"/evaluations_DoA_err_STD_rec", evaluations_DoA_err_STD_rec)
            np.save(exe_dir+"/evaluations_Pw", evaluations_Pw)
        if (t>1e6) and (t%1e6==0):
            np.save(exe_dir+"/replay_buffer_low", {"state":replay_buffer_low.state,"action":replay_buffer_low.action,"next_state":replay_buffer_low.next_state,"reward":replay_buffer_low.reward,"done":replay_buffer_low.not_done})
            np.save(exe_dir+"/replay_buffer_high", {"state":replay_buffer_high.state,"action":replay_buffer_high.action,"next_state":replay_buffer_high.next_state,"reward":replay_buffer_high.reward,"done":replay_buffer_high.not_done})
    np.save(exe_dir+"/replay_buffer_low", {"state":replay_buffer_low.state,"action":replay_buffer_low.action,"next_state":replay_buffer_low.next_state,"reward":replay_buffer_low.reward,"done":replay_buffer_low.not_done})
    np.save(exe_dir+"/replay_buffer_high", {"state":replay_buffer_high.state,"action":replay_buffer_high.action,"next_state":replay_buffer_high.next_state,"reward":replay_buffer_high.reward,"done":replay_buffer_high.not_done})