# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Recycler
"""

from mesa import Agent
import numpy as np
import pandas as pd


class Recyclers(Agent):
    """
    A recycler which sells recycled materials and improve its processes.

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_CE_PV_Model)
        original_recycling_cost (a list for a triangular distribution) ($/fu) (
            default=[0.106, 0.128, 0.117]). From EPRI 2018.
        init_eol_rate (dictionary with initial end-of-life (EOL) ratios),
            (default={"repair": 0.005, "sell": 0.02, "recycle": 0.1,
            "landfill": 0.4375, "hoard": 0.4375}). From Monteiro Lunardi
            et al 2018 and European Commission (2015).
        recycling_learning_shape_factor, (default=-0.39). From Qiu & Suh 2019.

    """

    def __init__(self, unique_id, model, original_recycling_cost,
                 init_eol_rate, recycling_learning_shape_factor):
        """
        Creation of new recycler agent
        """
        super().__init__(unique_id, model)
        self.original_recycling_cost = np.random.triangular(
            original_recycling_cost[0], original_recycling_cost[2],
            original_recycling_cost[1])
        self.original_fraction_recycled_waste = init_eol_rate["recycle"]
        self.recycling_learning_shape_factor = recycling_learning_shape_factor
        self.recycling_cost = self.original_recycling_cost
        self.init_recycling_cost = self.original_recycling_cost
        self.recycler_total_volume = 0
        self.recycling_volume = 0
        self.repairable_volume = 0
        self.total_repairable_volume = 0
        #  Original recycling volume is based on previous years EoL volume
        # (from 2000 to 2019)
        yearly_waste_file = pd.read_csv("all_pca_dataOut_95-by-35.Adv.csv")
        yearly_waste = yearly_waste_file[
            yearly_waste_file['year'] <= 2020]
        yearly_waste = sum(
            yearly_waste['Yearly_Sum_Power_atEOL'].tolist())
        self.original_recycling_volume = \
            (1 - self.model.repairability) * \
            self.original_fraction_recycled_waste * yearly_waste
        self.symbiosis = False
        self.agent_i = self.unique_id - self.model.num_consumers
        self.recycler_costs = 0
        self.recycler_name = self.model.recycler_names.pop()

    def update_transport_recycling_costs(self):
        """
        Update transportation costs according to the (evolving) mass of waste.
        Here, an average distance between all origins and targets is assumed.
        """
        self.recycling_cost = \
            self.recycling_cost + \
            (self.model.dynamic_product_average_wght -
             self.model.product_average_wght) * \
            self.model.transportation_cost / 1E3 * \
            self.model.mn_mx_av_distance_to_recycler[2]

    def update_recycled_waste(self):
        """
        Update consumers' amount of recycled waste.
        """
        if self.unique_id == self.model.num_consumers:
            for agent in self.model.schedule.agents:
                if agent.unique_id < self.model.num_consumers:
                    agent.update_yearly_recycled_waste(False)

    def triage(self):
        """
        Evaluate amount of products that can be refurbished
        """
        self.recycler_total_volume = 0
        self.recycling_volume = 0
        self.repairable_volume = 0
        self.total_repairable_volume = 0
        tot_waste_sold = 0
        new_installed_capacity = 0
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers and \
                    agent.EoL_pathway == "sell":
                tot_waste_sold += agent.number_product_EoL
            if agent.unique_id < self.model.num_consumers and \
                    agent.purchase_choice == "used":
                new_installed_capacity += agent.number_product[-1]
        used_vol_purchased = self.model.consumer_used_product \
            / self.model.num_consumers * new_installed_capacity
        tot_waste_sold += self.model.yearly_repaired_waste
        if tot_waste_sold < used_vol_purchased:
            for agent in self.model.schedule.agents:
                if agent.unique_id < self.model.num_consumers and \
                        agent.recycling_facility_id == self.unique_id:
                    self.recycler_total_volume += agent.yearly_recycled_waste
                    if self.model.yearly_repaired_waste < \
                            self.model.repairability * self.model.total_waste:
                        self.recycling_volume = \
                            (1 - self.model.repairability) * \
                            self.recycler_total_volume
                        self.repairable_volume = self.recycler_total_volume - \
                            self.recycling_volume
                    else:
                        self.recycling_volume = self.recycler_total_volume
                        self.repairable_volume = 0
        else:
            for agent in self.model.schedule.agents:
                if agent.unique_id < self.model.num_consumers and \
                        agent.recycling_facility_id == self.unique_id:
                    self.recycler_total_volume += agent.yearly_recycled_waste
                    self.recycling_volume = self.recycler_total_volume
                    self.repairable_volume = 0
        self.model.recycler_repairable_waste += self.repairable_volume
        self.total_repairable_volume += self.repairable_volume
        self.model.yearly_repaired_waste += self.repairable_volume

    def learning_curve_function(self, original_volume, volume, original_cost,
                                shape_factor):
        """
        Account for the learning effect: recyclers and refurbishers improve
        their recycling and repairing processes respectively
        """
        if volume > 0:
            potential_recycling_cost = original_cost * \
                                       (volume / original_volume) ** \
                                       shape_factor
            if potential_recycling_cost < original_cost:
                return potential_recycling_cost
            else:
                return original_cost
        return original_cost

    def compute_recycler_costs(self):
        """
        Compute societal costs of recyclers. Only account for the material
        recovered and the costs of recycling processes. Sales revenue of
        repairable products are not included.
        """
        revenue = 0
        for agent in self.model.schedule.agents:
            if self.model.num_consumers + self.model.num_recyclers <= \
                    agent.unique_id < self.model.num_consumers + \
                    self.model.num_prod_n_recyc:
                if not np.isnan(agent.yearly_recycled_material_volume) and \
                        not np.isnan(agent.recycled_mat_price):
                    revenue += agent.yearly_recycled_material_volume * \
                               agent.recycled_mat_price
        revenue /= self.model.num_recyclers
        self.recycler_costs += \
            ((self.recycling_volume + self.model.installer_recycled_amount) *
             self.recycling_cost - revenue)

    def step(self):
        """
        Evolution of agent at each step
        """
        self.update_recycled_waste()
        self.triage()
        self.recycling_cost = self.learning_curve_function(
            self.original_recycling_volume, self.recycling_volume,
            self.original_recycling_cost,
            self.recycling_learning_shape_factor)
        # ! we do not use this function anymore, distance now use the
        # ! pca-recycler data frame
        # self.update_transport_recycling_costs()
