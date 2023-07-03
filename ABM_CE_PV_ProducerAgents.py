# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Producer
"""

from mesa import Agent
import numpy as np


class Producers(Agent):
    """
    A producer which buys recycled materials, following the model from Ghali
    et al. 2017. The description of IS in Mathur et al.  2020 is also used.
    It is assumed that there is no volume threshold to establish IS as
    recyclers may keep materials until they have enough to ship
    to producer. For materials like Al and glass, industries already
    incorporate recycled materials. Thus the market is established for those
    materials and the IS model from Ghali et al. 2017 is bypassed, considering
    those materials are directly reused in established markets. For materials
    like Si or Ag we may assume that other industries would accept
    recycled materials following the IS model from Ghali et al. 2017
    (see Mathur et al. 2020). The IS model may also be bypassed entirely as
    most parameters' values for this model are uncertain.

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_CE_PV_Model)
        scd_mat_prices (dictionary containing lists for triangular
            distributions of secondary materials prices) ($/fu), (default={
            "Product": [np.nan, np.nan, np.nan], "Aluminum": [0.66, 1.98,
            1.32], "Glass": [0.01, 0.06, 0.035], "Copper": [3.77, 6.75, 5.75],
            "Insulated cable": [3.22, 3.44, 3.33], "Silicon": [2.20, 3.18,
            2.69], "Silver": [453, 653, 582]}). From www.Infomine.com (2019),
            copper.org (2019), USGS (2017), Bureau of Labor Statistics (2018),
            www.recyclingproductnews.com (all websites accessed 03/2020).
        virgin_mat_prices (dictionary containing lists for triangular
            distributions of virgin materials prices) ($/fu), (default={
            "Product": [np.nan, np.nan, np.nan], "Aluminum": [1.76, 2.51,
            2.14], "Glass": [0.04, 0.07, 0.055], "Copper": [4.19, 7.50, 6.39],
            "Insulated cable": [3.22, 3.44, 3.33], "Silicon": [2.20, 3.18,
            2.69], "Silver": [453, 653, 582]}). From Butler et al. (2005),
            Newlove  (2017), USGS (2017), www.infomine.com (2019), expert
            opinions (for insulated cables) (all websites accessed 03/2020).

    """

    def __init__(self, unique_id, model, scd_mat_prices, virgin_mat_prices):
        """
        Creation of new producer agent
        """
        super().__init__(unique_id, model)
        self.agent_i = self.unique_id - self.model.num_consumers
        self.material_produced = self.producer_type()
        self.recycled_material_volume = 0
        self.yearly_recycled_material_volume = 0
        self.recycling_volume = 0
        self.recycled_mat_price = np.random.triangular(
            scd_mat_prices[self.material_produced][0], scd_mat_prices[
                self.material_produced][2], scd_mat_prices[
                self.material_produced][1])
        self.virgin_mat_prices = np.random.triangular(
            virgin_mat_prices[self.material_produced][0], virgin_mat_prices[
                self.material_produced][2], virgin_mat_prices[
                self.material_produced][1])
        self.all_virgin_mat_prices = virgin_mat_prices
        self.recycled_material_value = 0
        self.industrial_waste = {
            k: self.model.material_waste_ratio[k] *
            self.model.product_mass_fractions[k] *
            self.model.dynamic_product_average_wght
            for k in self.model.material_waste_ratio}
        self.industrial_waste_ratio = {
            k: self.model.material_waste_ratio[k] *
            self.model.product_mass_fractions[k] for k in
            self.model.material_waste_ratio}
        self.industrial_waste_generated = 0
        self.yearly_industrial_waste_generated = 0
        self.producer_costs = 0
        self.transport_cost_industrial_waste = 0
        self.avoided_costs_virgin_materials = 0

    def producer_type(self):
        """
        Distribute producers' types (what materials are produced by each
        producer agent) among the producers.
        """
        for i in range(len(list(self.model.product_mass_fractions.keys()))):
            if round(self.unique_id - (
                self.model.num_consumers + self.model.num_recyclers) <=
                     (i + 1) * self.model.num_producers / len(
                     list(self.model.product_mass_fractions.keys()))):
                return list(self.model.product_mass_fractions.keys())[i]

    def count_producer_type(self, producer_type):
        """
        Count the number of producers according to their types.
        """
        count = 0
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'material_produced'):
                if agent.material_produced == producer_type:
                    count += 1
        return count

    def industrial_waste_generation(self):
        """
        Generate industrial waste.
        """
        if self.material_produced == "Product":
            num_product_producer = self.count_producer_type("Product")
            ind_waste = sum(self.industrial_waste_ratio.values()) * \
                self.model.total_yearly_new_products / \
                num_product_producer
            self.industrial_waste_generated += ind_waste
            self.yearly_industrial_waste_generated = ind_waste

    def add_installer_recycled_volumes(self):
        """
        Account recycled.
        """
        tot_recycled = 0
        amount_recyclers = 0
        self.model.installer_recycled_amount = 0
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers:
                tot_recycled += agent.yearly_recycled_waste
            if self.model.num_consumers <= agent.unique_id < \
                    self.model.num_consumers + self.model.num_recyclers:
                amount_recyclers += agent.recycling_volume
        self.model.installer_recycled_amount = \
            (tot_recycled - amount_recyclers) / self.model.num_prod_n_recyc

    def recovered_volume_n_value(self):
        """
        Compute exchanged volumes from industrial synergy. Mathematical
        model adapted from Ghali et al. 2017.
        """
        self.yearly_recycled_material_volume = 0
        num_neighbors_producer = 0
        tot_recycling_volume = 0
        for agent in self.model.schedule.agents:
            if self.model.num_consumers <= agent.unique_id < \
                    self.model.num_consumers + self.model.num_prod_n_recyc:
                if agent.unique_id >= self.model.num_recyclers + \
                        self.model.num_consumers and \
                        agent.material_produced == self.material_produced:
                    num_neighbors_producer += 1
                tot_recycling_volume += agent.recycling_volume + \
                    self.model.installer_recycled_amount

        # ! TODO: replace code below with PV_ICE material waste value - START

        # ! Multiply self.pv_ice_yearly_waste by the ratio of waste (in W)
        # ! between what is recycled and the total amount of waste:
        # ! tot_recycling_volume / self.model.total_waste
        # ! use the material value from PV_ICE corresponding with the material
        # ! type of the producer (self.material_produced). Replace recl_vol
        # ! with this new value.

        recl_vol = \
            self.model.product_mass_fractions[self.material_produced] \
            * tot_recycling_volume / num_neighbors_producer * \
            self.model.dynamic_product_average_wght \
            * self.model.recovery_fractions[self.material_produced]

        # ! TODO: replace code below with PV_ICE material waste value - STOP

        self.recycled_material_volume += recl_vol
        self.yearly_recycled_material_volume = recl_vol
        self.recycled_material_value = self.recycled_mat_price * \
            self.recycled_material_volume

    def costs_producer(self):
        """
        Compute societal costs of producers. Only account for the
        transportation and end of life costs of industrial waste as well as the
        avoided costs from using recovered materials. Does not account for the
        sales of materials and products.
        """
        self.avoided_costs_virgin_materials = 0
        self.transport_cost_industrial_waste = 0
        avd_costs_industrial_waste = 0
        if not np.isnan(self.virgin_mat_prices):
            self.avoided_costs_virgin_materials = \
                self.yearly_recycled_material_volume * (
                        self.recycled_mat_price - self.virgin_mat_prices)
        if not self.model.epr_business_model:
            self.transport_cost_industrial_waste = \
                self.yearly_industrial_waste_generated * \
                ((self.model.yearly_product_wght *
                  self.model.transportation_cost / 1E3 *
                  self.model.mean_distance_within_state) +
                 self.model.average_landfill_cost)
        elif self.material_produced == "Product":
            self.transport_cost_industrial_waste = 0
            for key, value in self.industrial_waste_ratio.items():
                virgin_mat_p = self.all_virgin_mat_prices[key]
                if not np.isnan(virgin_mat_p[2]):
                    avd_costs_industrial_waste += \
                        -1 * self.yearly_industrial_waste_generated * value * \
                        virgin_mat_p[2]
        self.producer_costs += (self.avoided_costs_virgin_materials +
                                avd_costs_industrial_waste +
                                self.transport_cost_industrial_waste)

    def step(self):
        """
        Evolution of agent at each step
        """
        self.industrial_waste_generation()
