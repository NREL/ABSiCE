# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Producer
"""

from mesa import Agent
import numpy as np
import networkx as nx
import random


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
        social_influencability_boundaries (from Ghali et al. 2017)
        self_confidence_boundaries (from Ghali et al. 2017)

    """

    def __init__(self, unique_id, model, scd_mat_prices, virgin_mat_prices,
                 social_influencability_boundaries,
                 self_confidence_boundaries):
        """
        Creation of new producer agent
        """
        super().__init__(unique_id, model)
        self.trust_history = np.copy(self.model.trust_prod)
        self.social_influencability = np.random.uniform(
            social_influencability_boundaries[0],
            social_influencability_boundaries[1])
        self.agent_i = self.unique_id - self.model.num_consumers
        self.knowledge = np.random.random()
        self.social_interactions = np.random.random()
        self.knowledge_learning = np.random.random()
        self.knowledge_t = self.knowledge
        self.acceptance = 0
        self.symbiosis = False
        self.self_confidence = np.random.uniform(
            self_confidence_boundaries[0], self_confidence_boundaries[1])
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

    def update_trust(self):
        """
        Update trust of agents in one another within the industrial symbiosis
        network. Mathematical model adapted from Ghali et al. 2017.
        """
        random_social_event = np.asmatrix(
            np.random.uniform(self.model.social_event_boundaries[0],
                              self.model.social_event_boundaries[1],
                              (self.model.num_prod_n_recyc,
                               self.model.num_prod_n_recyc)))
        for agent in self.model.schedule.agents:
            if self.model.num_consumers <= agent.unique_id < \
                    self.model.num_consumers + self.model.num_prod_n_recyc:
                agent_j = agent.unique_id - self.model.num_consumers
                common_neighbors = list(
                    nx.common_neighbors(self.model.G, self.unique_id,
                                        agent.unique_id))
                if common_neighbors:
                    trust_neighbors = \
                        [self.model.trust_prod[self.agent_i, i -
                                               self.model.num_consumers]
                         for i in common_neighbors]
                    avg_trust_neighbors = self.social_influencability * (
                            sum(trust_neighbors) / len(trust_neighbors) -
                            self.trust_history[self.agent_i, agent_j])
                # Slight modification from Ghali et al.: if no common contact
                # there is no element for reputation
                else:
                    avg_trust_neighbors = 0
                trust_ij = self.trust_history[self.agent_i, agent_j] + \
                    avg_trust_neighbors + random_social_event[
                               self.agent_i, agent_j]
                if trust_ij < -1:
                    trust_ij = -1
                if trust_ij > 1:
                    trust_ij = 1
                self.model.trust_prod[self.agent_i, agent_j] = trust_ij
        self.trust_history = ((self.trust_history * (self.model.clock + 1)) +
                              self.model.trust_prod) / (self.model.clock + 2)

    def update_knowledge(self):
        """
        Update knowledge of agents about industrial symbiosis. Mathematical
        model adapted from Ghali et al. 2017.
        """
        self.knowledge_learning = np.random.random()
        knowledge_neighbors = 0
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            self.social_interactions = np.random.random()
            agent_j = agent.unique_id - self.model.num_consumers
            if self.model.trust_prod[self.agent_i, agent_j] >= \
                    self.model.trust_threshold:
                knowledge_neighbors += self.social_interactions * (
                        agent.knowledge - self.knowledge)
        self.knowledge_t = self.knowledge
        self.knowledge += self.social_influencability * knowledge_neighbors + \
            self.knowledge_learning
        if self.knowledge < 0:
            self.knowledge = 0
        if self.knowledge > 1:
            self.knowledge = 1

    def update_acceptance(self):
        """
        Update agents' acceptance of industrial symbiosis. Mathematical model
        adapted from Ghali et al. 2017.
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        neighbors_influence = \
            len([agent for agent in self.model.grid.get_cell_list_contents(
                neighbors_nodes) if agent.symbiosis]) / \
            len([agent for agent in self.model.grid.get_cell_list_contents(
                neighbors_nodes)])
        self.acceptance += self.social_influencability * neighbors_influence \
            + self.self_confidence * (self.knowledge - self.knowledge_t)
        self.knowledge_t = self.knowledge
        if self.acceptance < 0:
            self.acceptance = 0
        if self.acceptance > 1:
            self.acceptance = 1

    def update_willingness(self):
        """
        Update willingness to form an industrial synergy. Mathematical
        model adapted from Ghali et al. 2017.
        """
        number_synergies = 0
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            agent_j = agent.unique_id - self.model.num_consumers
            if self.model.trust_prod[self.agent_i, agent_j] >= \
                    self.model.trust_threshold and self.knowledge > \
                    self.model.knowledge_threshold:
                self.model.willingness[self.agent_i, agent_j] = self.acceptance
                number_synergies += 1
        if number_synergies > 0:
            self.symbiosis = True

    def add_installer_recycled_volumes(self):
        """
        Update willingness to form an industrial synergy. Mathematical
        model adapted from Ghali et al. 2017.
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
        if self.model.industrial_symbiosis:
            neighbors_nodes = \
                self.model.grid.get_neighbors(self.pos, include_center=False)
            for agent in \
                    self.model.grid.get_cell_list_contents(neighbors_nodes):
                agent_j = agent.unique_id - self.model.num_consumers
                num_neighbors_producer = 0
                agent_neighbors = \
                    agent.model.grid.get_neighbors(agent.pos,
                                                   include_center=False)
                for agent2 in self.model.grid.get_cell_list_contents(
                        agent_neighbors):
                    if agent2.unique_id >= self.model.num_recyclers + \
                            self.model.num_consumers and \
                            agent.recycling_volume > 0 and \
                            agent2.material_produced == self.material_produced:
                        num_neighbors_producer += 1
                if num_neighbors_producer == 0:
                    num_neighbors_producer = 1
                recl_vol = \
                    self.model.product_mass_fractions[self.material_produced]\
                    * (agent.recycling_volume +
                       self.model.installer_recycled_amount) / \
                    num_neighbors_producer * \
                    self.model.dynamic_product_average_wght \
                    * self.model.recovery_fractions[self.material_produced]
                if self.model.established_scd_mkt[self.material_produced]:
                    self.recycled_material_volume += recl_vol
                    self.yearly_recycled_material_volume += recl_vol
                else:
                    if self.model.willingness[self.agent_i, agent_j] >= \
                            self.model.willingness_threshold:
                        self.recycled_material_volume += recl_vol
                        self.yearly_recycled_material_volume += recl_vol
        else:
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
            recl_vol = \
                self.model.product_mass_fractions[self.material_produced] \
                * tot_recycling_volume / num_neighbors_producer * \
                self.model.dynamic_product_average_wght \
                * self.model.recovery_fractions[self.material_produced]
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
        if not all(self.model.established_scd_mkt.values()) or not \
                self.model.industrial_symbiosis:
            self.update_trust()
            self.update_knowledge()
            self.update_acceptance()
            self.update_willingness()
