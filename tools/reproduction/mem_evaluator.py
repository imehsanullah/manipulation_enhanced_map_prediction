"""Recovered eval_rss_method.py metric harness, adapted to official MEM.

Provenance and limitations: see README.md in this directory.
Invoke run_mem.py; this module is not a standalone launcher.
"""
import pandas as pd
from shelf_gym.scripts.data_generation.pushing_collection import PushingCollection
from shelf_gym.scripts.model_training.train_ycb_push_prediction import PushPredictor
import os
from shelf_gym.utils.result_visualization_utils import get_my_cmap
from shelf_gym.utils.information_gain_utils import InfoGainEval
import torch
from shelf_gym.utils.learning_utils.datasets import MapDatasetH5py
from tqdm import tqdm
import cupy as cp
import numpy as np
from glob import glob
import argparse
from shelf_gym.utils.map_calibration_utils import Cumulative_mIoU,mECE_Calibration_calc_3D
from shelf_gym.utils.model_evaluation_utils import EvaluationHelper, get_valid_choices
from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(40)


def mean_std_ignore_zeros(arr):
    arr = np.asarray(arr)
    non_zero_vals = arr[arr != 0]
    if non_zero_vals.size > 0:
        return np.mean(non_zero_vals), np.std(non_zero_vals)
    else:
        return 0, 0

def eval_and_process_policy(env, scenes, name):
    occ_ious = []
    sem_ious = []
    occ_cals = []
    sem_cals = []
    stop_steps = int(MAX_ACTIONS/EVAL_INTERVAL)
    for stop_stage in range(stop_steps):
        occ_ious.append(Cumulative_mIoU(n_classes = 2))
        sem_ious.append(Cumulative_mIoU(n_classes = env.n_classes))
        occ_cals.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = 2))
        sem_cals.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = env.n_classes))


    pos_diffs_all = np.zeros((1,MAX_ACTIONS))
    pos_diffs_collision_all = np.zeros((1,MAX_ACTIONS))
    step_times_all = np.zeros((1,MAX_ACTIONS))
    push_times_all = np.zeros((1,MAX_ACTIONS))
    vpp_times_all = np.zeros((1,MAX_ACTIONS))
    push_matrix = []
    push_summaries = []
    for challenge_n,challenge in tqdm(enumerate(scenes)):
        output = env.run(challenge, debug=False)
        semantic_gts = output['semantic_gt']
        semantic_maps = output['semantic_map']
        occupancy_maps = output['occupancy_map']
        occupancy_gts = output['occupancy_gt']
        pos_diffs = np.asarray(output['pos_diffs'])
        pos_diffs_collision = np.asarray(output['pos_diffs_without_collision'])
        step_times = np.asarray(output['step_time'])
        push_times = np.asarray(output['push_time'])
        vpp_times = np.asarray(output['vpp_time'])
        pushes = np.asarray(output['pushes'])
        push_matrix.append(pushes.tolist())  # assume 0,1,2 values
        pos_diffs_all = np.vstack((pos_diffs_all,pos_diffs))
        pos_diffs_collision_all = np.vstack((pos_diffs_collision_all,pos_diffs_collision))
        step_times_all = np.vstack((step_times_all,step_times))
        push_times_all = np.vstack((push_times_all,push_times))
        vpp_times_all = np.vstack((vpp_times_all,vpp_times))
        num_pushes = int(((pushes == 1) | (pushes == 2)).sum())  # Count both 1s and 2s
        summary_text = f"In scene {challenge_n} there have been {num_pushes} pushes"
        summary_text += "; collision status unavailable from official action labels"
        push_summaries.append(summary_text)

        for stop_stage,data in enumerate(zip(occupancy_maps,semantic_maps,occupancy_gts,semantic_gts)):
            occupancy_map,semantic_map,occupancy_gt,semantic_gt = data
            occupancy_gt = cp.asarray(occupancy_gt).astype(int)
            occupancy_map = cp.array(occupancy_map)
            semantic_gt = cp.asarray(semantic_gt)
            oc = occupancy_map.flatten()
            not_oc = 1-oc
            oc_probs = cp.stack([not_oc,oc],axis = 1)
            semantic_map = cp.array(semantic_map)
            occ_ious[stop_stage].update_counts(occupancy_map.flatten()>0.5,occupancy_gt.flatten())
            occ_cals[stop_stage].update_bins(oc_probs,occupancy_gt.flatten())
            sem_ious[stop_stage].update_counts(semantic_map.argmax(axis = -1),semantic_gt)
            sem_cals[stop_stage].update_bins(semantic_map.reshape(-1,env.n_classes),semantic_gt.reshape(-1))


        dfs = []
        summary_dfs = []
        for stop_step,data in enumerate(zip(sem_cals,sem_ious,occ_ious,occ_cals)):
            pos_diff = np.mean(pos_diffs_all[1:, stop_step])  # Skip dummy row
            pos_diff_collision = np.mean(pos_diffs_collision_all[1:, stop_step])

            step_mean, step_std = mean_std_ignore_zeros(step_times_all[1:, stop_step])
            push_mean, push_std = mean_std_ignore_zeros(push_times_all[1:, stop_step])
            vpp_mean, vpp_std = mean_std_ignore_zeros(vpp_times_all[1:, stop_step])

            sem_cal,sem_iou,occ_iou,occ_cal = data

            sem_eces = cp.array((np.nan_to_num(cp.asnumpy(sem_cal.get_ECEs()),0))).get()
            sem_eces = sem_eces.tolist()
            sem_eces.append(np.mean(sem_eces[:-1]))
            oc_eces = cp.array(occ_cal.get_ECEs()).get()[-1]
            iou_occ = np.mean(np.nan_to_num(cp.asnumpy(occ_iou.get_IoUs()),0).tolist())
            iou_sem = np.nan_to_num(cp.asnumpy(sem_iou.get_IoUs()),0).tolist()
            iou_sem.append(np.mean(iou_sem))
            experiment_name = ['MSDM (Ours)']
            semantic_ECE_titles = []
            semantic_iou_titles = []
            for c in range(15):
                semantic_ECE_titles.append('semantic_ECE_class_{}'.format(c))
                semantic_iou_titles.append('semantic_iou_class_{}'.format(c))
            semantic_ECE_titles.append('semantic_ECE_aggregate')
            semantic_ECE_titles.append('semantic_mECE')
            semantic_iou_titles.append('semantic_miou')

            occupancy_ECE_title = 'occupancy_ECE'
            occupancy_iou_title = 'occupancy_iou'


            input_dict = {}

            for title,ece in zip(semantic_ECE_titles,sem_eces):
                input_dict.update({title:[ece]})
            input_dict.update({occupancy_ECE_title:[oc_eces]})

            for title,iou in zip(semantic_iou_titles,iou_sem):
                input_dict.update({title:[iou]})

            input_dict.update({occupancy_iou_title:[iou_occ]})
            input_dict.update({'experiment':experiment_name})
            input_dict['pos_diff'] = [pos_diff]
            input_dict['pos_diff_without_collision'] = [pos_diff_collision]
            input_dict['step_time'] = [step_mean]
            input_dict['push_time'] = [push_mean]
            input_dict['vpp_time'] = [vpp_mean]
            input_dict['step_std'] = [step_std]
            input_dict['push_std'] = [push_std]
            input_dict['vpp_std'] = [vpp_std]

            df =  pd.DataFrame(input_dict)
            df.loc[:,'stop_step'] = (stop_step+1)*EVAL_INTERVAL

            #     df.to_csv('results_dengler_et_al.csv',sep = '|',index = False)
            summary_df = df.loc[:,
                         ['experiment', 'stop_step', 'semantic_mECE', 'semantic_miou', 'occupancy_ECE', 'occupancy_iou','pos_diff', 'pos_diff_without_collision', 'step_time', 'push_time', 'vpp_time']]
            dfs.append(df)
            summary_dfs.append(summary_df)
        df = pd.concat(dfs)
        summary_df = pd.concat(summary_dfs)
        summary_df.loc[:,'partial_i'] = challenge_n

        summary_df.to_csv('./results/{}/clean_start/{}/partial_{}_summary.csv'.format(experiment_type,name,name),sep = '|',index = False)
        df.to_csv('./results/{}/clean_start/{}/partial_{}.csv'.format(experiment_type,name,name),sep = '|',index = False)

        # Create and save push matrix DataFrame
        push_df = pd.DataFrame(push_matrix)
        push_df.columns = [f"step_{(i + 1) * EVAL_INTERVAL}" for i in range(push_df.shape[1])]
        push_df.index.name = "scene"
        push_df.to_csv('./results/{}/clean_start/{}/partial_push_data_{}.csv'.format(experiment_type,name,name),sep = '|',index = False)

        # Create and save push summary DataFrame
        push_summary_df = pd.DataFrame({
            "scene": list(range(len(push_summaries))),
            "summary": push_summaries
        })
        push_summary_df.to_csv('./results/{}/clean_start/{}/partial_push_data_{}_summary.csv'.format(experiment_type,name,name),sep = '|',index = False)

    return df, summary_df, push_df, push_summary_df
