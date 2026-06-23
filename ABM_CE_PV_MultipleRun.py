# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 12:43 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Run - one or several simulations with all states of outputs
"""

from ABM_CE_PV_Model import *
import matplotlib.pyplot as plt
import time
import os
from itertools import product
from utils import TIMESTEP
from ABM_CE_PV_ConsumerAgents import Consumers


# Calibrated parameters (new values stay in ranges reported in the
# literature):
# w_sn_eol=0.23, (previously 0.27)
# w_a_eol=0.5, (previously 0.39)
# recycling_learning_shape_factor=-0.3, (previously -0.39)
# att_distrib_param_eol=[0.805, 0.10]) (previously [0.544, 0.09])
# Default values have been modified, no need to modify them in
# functions below.

def run_model(number_run, number_steps, timestep=TIMESTEP.ANNUAL):
    """
    Run model several times and collect outputs at each time steps. Creates
    a new file for each run. Use a new seed for random generation at each
    run.
    Args:
        number_run (int): Number of runs to perform.
        number_steps (int): Number of steps in years.
        timestep (TIMESTEP): Time step of the simulation, default is annual.
        """
    number_steps = get_number_of_steps(number_steps, timestep)
    for j in range(number_run):
        # Reinitialize model
        # j = j + 43
        t0 = time.time()
        if j < 10:
            model = ABM_CE_PV(
                seed=(j),
                # att_distrib_param_eol=[0.425, 0.1], 
                last_step=number_steps,
                hazardous_waste_regulation_enabled=True,
                landfill_solar_waste_acceptance_ratio=1.0,
                calculate_distances=False,
                # model_states= ['TX', 'AZ', 'NV', 'NM'],
                solar_cycle=False,
                rtn=False,
                # landfill_data_params = {
                #     "landfill_volume_column": "Waste Business Journal Costs ($/metric tons)",
                #     "landfill_name_column": "Landfill Name"},
                # file_name={
                #     'Landfill data': "LandfillCostsbyYearAllLandfills.csv",
                #     'Recycling data': "RecyclingCostsbyYearAllLandfills.csv",
                #     'Hazardous landfill data': "Landfills_data_SA.csv"
                #     },
                timestep=timestep,
                )  # baseline
        elif j < 20:
            model = ABM_CE_PV(
                seed=(j - 10), last_step=number_steps,
                att_distrib_param_eol=[0.65, 0.1])  # this can be used for scenario analysis
        elif j < 30:
            model = ABM_CE_PV(
                seed=(j - 20), last_step=number_steps,
                att_distrib_param_eol=[0.5, 0.1])
        elif j < 4:
            model = ABM_CE_PV(
                seed=(j - 3), last_step=number_steps,
                sa_landfill_costs=(True, 0.0000),
                file_name={'Landfill data': "Landfills_data_SA.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances_SA.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1000-1E-6, 1000+1E-6, 1000],  # 0.0077 $/W → $/ton
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 5:
            model = ABM_CE_PV(
                seed=(j - 4), last_step=number_steps,
                sa_landfill_costs=(True, 1000),  # 0.0077 $/W → $/ton
                file_name={'Landfill data': "Landfills_data_SA.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances_SA.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1000-1E-6, 1000+1E-6, 1000],  # 0.0077 $/W → $/ton
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 6:
            model = ABM_CE_PV(
                seed=(j - 5), last_step=number_steps,
                sa_landfill_costs=(True, 1740),  # 0.0134 $/W → $/ton
                file_name={'Landfill data': "Landfills_data_SA.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances_SA.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1000-1E-6, 1000+1E-6, 1000],  # 0.0077 $/W → $/ton
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 7:
            model = ABM_CE_PV(
                seed=(j - 6), last_step=number_steps,
                sa_landfill_costs=(True, 0.0000),  # 0.0 $/W (no change)
                file_name={'Landfill data': "Landfills_data_SA.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances_SA.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],  # near-zero, no conversion needed
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 8:
            model = ABM_CE_PV(
                seed=(j - 7), last_step=number_steps,
                sa_landfill_costs=(True, 1000),  # 0.0077 $/W → $/ton
                file_name={'Landfill data': "Landfills_data_SA.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances_SA.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 9:
            model = ABM_CE_PV(
                seed=(j - 8), last_step=number_steps,
                sa_landfill_costs=(True, 1740),  # 0.0134 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data_SA.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances_SA.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 10:
            model = ABM_CE_PV(
                seed=(j - 9), last_step=number_steps,
                sa_landfill_costs=(True, 0.0000),
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1740-1E-6, 1740+1E-6, 1740],  # 0.0134 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1E-6)
                # transportation_cost=0.25)
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 11:
            model = ABM_CE_PV(
                seed=(j - 10), last_step=number_steps,
                sa_landfill_costs=(True, 1000),  # 0.0077 $/W -> $/ton,
                # sa_landfill_costs=(True, 0.0038),
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1740-1E-6, 1740+1E-6, 1740],  # 0.0134 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=0.5,
                # transportation_cost=0.75)
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 12:
            model = ABM_CE_PV(
                seed=(j - 11), last_step=number_steps,
                sa_landfill_costs=(True, 1740),  # 0.0134 $/W -> $/ton,
                # sa_landfill_costs=(True, 0.0077),
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1740-1E-6, 1740+1E-6, 1740],  # 0.0134 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1,
                # transportation_cost=1.25)
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 13:
            model = ABM_CE_PV(
                seed=(j - 12), last_step=number_steps,
                sa_landfill_costs=(True, 0.0000),
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1000-1E-6, 1000+1E-6, 1000],  # 0.0077 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 14:
            model = ABM_CE_PV(
                seed=(j - 13), last_step=number_steps,
                sa_landfill_costs=(True, 1000),  # 0.0077 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1000-1E-6, 1000+1E-6, 1000],  # 0.0077 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 15:
            model = ABM_CE_PV(
                seed=(j - 14), last_step=number_steps,
                sa_landfill_costs=(True, 1740),  # 0.0134 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                original_recycling_cost=[1000-1E-6, 1000+1E-6, 1000],  # 0.0077 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 16:
            model = ABM_CE_PV(
                seed=(j - 15), last_step=number_steps,
                sa_landfill_costs=(True, 0.0000),
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 17:
            model = ABM_CE_PV(
                seed=(j - 16), last_step=number_steps,
                sa_landfill_costs=(True, 1000),  # 0.0077 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 18:
            model = ABM_CE_PV(
                seed=(j - 17), last_step=number_steps,
                sa_landfill_costs=(True, 1740),  # 0.0134 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.25,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 19:
            model = ABM_CE_PV(
                seed=(j - 18), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1.5,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 20:
            model = ABM_CE_PV(
                seed=(j - 19), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 21:
            model = ABM_CE_PV(
                seed=(j - 20), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=0.5,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 22:
            model = ABM_CE_PV(
                seed=(j - 21), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                init_eol_rate={"repair": 1E-6, "sell": 1E-6,
                                "recycle": 1E-6, "landfill": 1,
                                "hoard": 1E-6},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[0.064-1E-6, 0.064+1E-6, 0.064],
                original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1E-12,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 33:
            model = ABM_CE_PV(
                seed=(j - 22), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                original_recycling_cost=[8312-1E-6, 8312+1E-6, 8312],  # 0.064 $/W -> $/ton,
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1.5,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep,)
        elif j < 43:
            model = ABM_CE_PV(
                seed=(j - 33), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                original_recycling_cost=[8312-1E-6, 8312+1E-6, 8312],  # 0.064 $/W -> $/ton,
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1E-12,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep)
        elif j < 53:
            model = ABM_CE_PV(
                seed=(j - 43), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                original_recycling_cost=[11039-1E-6, 11039+1E-6, 11039],  # 0.085 $/W -> $/ton,
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1.5,
                w_sn_eol=0.27,
                w_pbc_eol=0.44,
                w_a_eol=0.39,
                timestep=timestep)
        elif j < 63:
            model = ABM_CE_PV(
                seed=(j - 53), last_step=number_steps,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                recycling_learning_shape_factor=-0.0,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                original_recycling_cost=[11039-1E-6, 11039+1E-6, 11039],  # 0.085 $/W -> $/ton,
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                transportation_cost=1E-12,
                w_sn_eol=0.27,
                w_pbc_eol=0.44,
                w_a_eol=0.39,
                timestep=timestep)
        elif j < 140:
            model = ABM_CE_PV(
                seed=(j - 120), last_step=number_steps,
                sa_landfill_costs=(True, 1494),  # 0.0115 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                original_recycling_cost=[11039-1E-6, 11039+1E-6, 11039],  # 0.085 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.5,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep)
        elif j < 160:
            model = ABM_CE_PV(
                seed=(j - 140), last_step=number_steps,
                sa_landfill_costs=(True, 1740),  # 0.0134 $/W -> $/ton,
                file_name={'Landfill data': "Landfills_data.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv"},
                original_recycling_cost=[11039-1E-6, 11039+1E-6, 11039],  # 0.085 $/W -> $/ton,
                # original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                # original_recycling_cost=[1E-12, 3E-12, 2E-12],
                # transportation_cost=1.5,
                w_sn_eol=0,
                w_pbc_eol=1,
                w_a_eol=0,
                timestep=timestep)
        elif j < 240:
            model = ABM_CE_PV(seed=(j - 210),
                              dynamic_lifetime_model={
                                  "Dynamic lifetime": True,
                                  "d_lifetime_intercept": 15.9,
                                  "d_lifetime_reg_coeff": 0.87,
                                  "Seed": False, "Year": 5,
                                  "avg_lifetime": 50})
        elif j < 270:
            model = ABM_CE_PV(seed=(j - 240),
                              all_EoL_pathways={"repair": True, "sell": True,
                                                "recycle": True,
                                                "landfill": False,
                                                "hoard": True},
                                                timestep=timestep)
        elif j < 300:
            model = ABM_CE_PV(seed=(j - 270),
                              seeding={"Seeding": True,
                                       "Year": 5, "number_seed": 50})
        elif j < 330:
            model = ABM_CE_PV(seed=(j - 300),
                              repairability=1,
                              init_purchase_choice={"new": 0, "used": 1,
                                                    "certified": 0},
                              w_sn_eol=0,
                              w_pbc_eol=0.44,
                              w_a_eol=0,
                              w_sn_reuse=0.497,
                              w_pbc_reuse=0.382,
                              w_a_reuse=0,
                              original_repairing_cost=[0.0001, 0.00045,
                                                       0.00028],
                              all_EoL_pathways={"repair": False, "sell": True,
                                                "recycle": False,
                                                "landfill": True,
                                                "hoard": True},
                                                timestep=timestep)
        else:
            model = ABM_CE_PV(seed=(j - 330),
                              calibration_n_sensitivity_3=0.65,
                              recovery_fractions={
                "Product": np.nan, "Aluminum": 0.994, "Glass": 0.98,
                "Copper": 0.97, "Insulated cable": 1., "Silicon": 0.97,
                "Silver": 0.94},
                timestep=timestep)
        for i in range(number_steps):
            model.step()
        # Get results in a pandas DataFrame
        results_model = model.datacollector.get_model_vars_dataframe()
        # results_agents = model.datacollector.get_agent_vars_dataframe()
        results_agents_consumers = model.datacollector.get_agenttype_vars_dataframe(agent_type=Consumers)
        results_agents_consumers.reset_index(inplace=True)
        results_agents_consumers.drop(columns=['Step', 'AgentID'], inplace=True)
        # Draw figures
        # draw_graphs(False, False, model, results_agents, results_model)
        print("Run", j+1, "out of", number_run)
        t1 = time.time()
        print(t1 - t0)
        os.chdir('../../../')
        if not os.path.exists("results"):
            os.makedirs("results")
        results_model.to_csv(os.path.join(
            "results", "Results_model_run_%s.csv" % j))
        results_agents_consumers.to_csv(os.path.join(
            "results", "Results_agents_consumers_run_%s.csv" % j), index=False)
        # results_agents.to_csv("results\\Results_agents_run_%s.csv" % j)


def run_batch(number_run, number_steps, **kwargs):
    """
    Run model several times and collect outputs at each time steps. Creates
    a new file for each run. Use a new seed for random generation at each
    run.
    """
    list_param_values = kwargs.values()
    list_param_names = kwargs.keys()
    combinations = list(product(*list_param_values))
    print(combinations)

    for j in range(len(combinations * number_run)):
        # Reinitialize model
        t0 = time.time()
        if j < 15:
            model = ABM_CE_PV(
                seed=(j), last_step=number_steps,
                att_distrib_param_eol=[0.5, 1E-6])
                # transportation_cost=0)
        elif j < 30:
            model = ABM_CE_PV(
                seed=(j - 40), last_step=number_steps,
                att_distrib_param_eol=[0.6, 1E-6])
                # transportation_cost=0)
        elif j < 120:
            model = ABM_CE_PV(
                seed=(j - 80), last_step=number_steps,
                transportation_cost=0)
        elif j < 160:
            model = ABM_CE_PV(
                seed=(j - 120), last_step=number_steps,
                transportation_cost=0)
        elif j < 200:
            model = ABM_CE_PV(
                seed=(j - 160), last_step=number_steps,
                transportation_cost=0)
        elif j < 240:
            model = ABM_CE_PV(seed=(j - 150),
                              recycling_learning_shape_factor=-0.6)
        elif j < 210:
            model = ABM_CE_PV(seed=(j - 180),
                              recycling_learning_shape_factor=-1E-6)
        elif j < 240:
            model = ABM_CE_PV(seed=(j - 210),
                              dynamic_lifetime_model={
                                  "Dynamic lifetime": True,
                                  "d_lifetime_intercept": 15.9,
                                  "d_lifetime_reg_coeff": 0.87,
                                  "Seed": False, "Year": 5,
                                  "avg_lifetime": 50})
        elif j < 270:
            model = ABM_CE_PV(seed=(j - 240),
                              all_EoL_pathways={"repair": True, "sell": True,
                                                "recycle": True,
                                                "landfill": False,
                                                "hoard": True})
        elif j < 300:
            model = ABM_CE_PV(seed=(j - 270),
                              seeding={"Seeding": True,
                                       "Year": 5, "number_seed": 50})
        elif j < 330:
            model = ABM_CE_PV(seed=(j - 300),
                              repairability=1,
                              init_purchase_choice={"new": 0, "used": 1,
                                                    "certified": 0},
                              w_sn_eol=0,
                              w_pbc_eol=0.44,
                              w_a_eol=0,
                              w_sn_reuse=0.497,
                              w_pbc_reuse=0.382,
                              w_a_reuse=0,
                              original_repairing_cost=[0.0001, 0.00045,
                                                       0.00028],
                              all_EoL_pathways={"repair": False, "sell": True,
                                                "recycle": False,
                                                "landfill": True,
                                                "hoard": True})
        else:
            model = ABM_CE_PV(seed=(j - 330),
                              calibration_n_sensitivity_3=0.65,
                              recovery_fractions={
                "Product": np.nan, "Aluminum": 0.994, "Glass": 0.98,
                "Copper": 0.97, "Insulated cable": 1., "Silicon": 0.97,
                "Silver": 0.94})
        for i in range(number_steps):
            model.step()
        # Get results in a pandas DataFrame
        results_model = model.datacollector.get_model_vars_dataframe()
        results_agents = model.datacollector.get_agent_vars_dataframe()
        # Draw figures
        draw_graphs(False, False, model, results_agents, results_model)
        print("Run", j+1, "out of", number_run)
        t1 = time.time()
        print(t1 - t0)
        os.chdir('../../../')
        if not os.path.exists("results"):
            os.makedirs("results")
        results_model.to_csv(os.path.join(
            "results", "Results_model_run_%s.csv" % j))
        # results_agents.to_csv("results\\Results_agents_run_%s.csv" % j)


def color_agents(step, column, condition1, condition2, model, results_agents):
    """
    Color figure of the network.
    """
    color_map = []
    for node in model.H1:
        agents_df = results_agents.loc[step, column]
        if agents_df[node] == condition1:
            color_map.append('green')
        elif agents_df[node] == condition2:
            color_map.append('red')
        else:
            color_map.append('grey')
    return color_map


def draw_graphs(network, figures, model, results_agents, results_model):
    """
    Draw different figures.
    """
    if network:
        plt.figure(figsize=(12, 12))
        nx.draw(model.H1, node_color=color_agents(
            1, "Recycling", "recycle", "landfill", model, results_agents),
                node_size=5, with_labels=False)
        # Draw other networks:
        # nx.draw(model.H1, node_color="lightskyblue")
        # nx.draw(model.H2, node_color="purple")
        # nx.draw(model.H3, node_color="chocolate", edge_color="white")
        # nx.draw(model.G, with_labels=False)
    if figures:
        results_model[results_model.columns[2:7]].plot()
        results_model[results_model.columns[15:20]].plot()
        plt.text(0.6, 0.7, 'Landfilling').set_color("red")
        plt.text(0.6, 0.8, 'Recycling').set_color("green")
        plt.text(0.6, 0.9, 'Other behavior').set_color("grey")
    if network or figures:
        plt.show()  # draw graph as desired and plot outputs

def get_number_of_steps(number_steps: int, timestep: TIMESTEP):
    """
    Get the number of steps based on the timestep.
    """
    if timestep == TIMESTEP.ANNUAL:
        return number_steps
    elif timestep == TIMESTEP.MONTHLY:
        return number_steps * 12
    elif timestep == TIMESTEP.QUARTERLY:
        return number_steps * 4
    else:
        raise ValueError("Unsupported timestep: {}".format(timestep))

run_model(10, 11, timestep=TIMESTEP.QUARTERLY)
# run_batch(40, 31, list1=['a', 'b', 'c'],list2=['d', 'e', 'f'],
#          list3=['x', 'y', 'z'])
