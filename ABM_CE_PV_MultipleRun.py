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


def run_model(number_run, number_steps):
    """
    Run model several times and collect outputs at each time steps. Creates
    a new file for each run. Use a new seed for random generation at each
    run.
    """
    for j in range(number_run):
        # Reinitialize model
        t0 = time.time()
        if j < 30:
            model = ABM_CE_PV(
                seed=j)
        elif j < 60:
            model = ABM_CE_PV(
                seed=(j - 30), w_sn_eol=0)
        elif j < 90:
            model = ABM_CE_PV(
                seed=(j - 60), seeding_recyc={
                    "Seeding": True, "Year": 1, "number_seed": 100,
                    "discount": 0.35})
        elif j < 120:
            model = ABM_CE_PV(seed=(j - 90), seeding_recyc={
                "Seeding": True, "Year": 1, "number_seed": 200,
                "discount": 0.35})
        elif j < 150:
            model = ABM_CE_PV(seed=(j - 120),
                              calibration_n_sensitivity_4=2)
        elif j < 180:
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
        #results_model.to_csv(
        #    "C:/Users/jwalzber/PycharmProjects/ABM_CE_PV_model/Results_model_run%s.csv" % j)
        #results_agents.to_csv("results\\Results_agents.csv")
        # Draw figures
        draw_graphs(False, False, model, results_agents, results_model)
        print("Run", j+1, "out of", number_run)
        t1 = time.time()
        print(t1 - t0)
        os.chdir('../../../')



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


run_model(30, 3)
