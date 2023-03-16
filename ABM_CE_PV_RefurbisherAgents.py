# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Refurbisher
"""

from mesa import Agent
import numpy as np
from ABM_CE_PV_RecyclerAgents import Recyclers
import operator
from scipy.stats import truncnorm


class Refurbishers(Agent):
    """
    A refurbisher which repairs modules (and eventually discard them), improve
    its processes and act as an intermediary between other actors.

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_CE_PV_Model)
        original_repairing_cost (a list for a triangular distribution) ($/fu),
            (default=[0.1, 0.45, 0.28]). From Tsanakas et al. 2019.
        init_eol_rate (dictionary with initial end-of-life (EOL) ratios),
            (default={"repair": 0.005, "sell": 0.02, "recycle": 0.1,
            "landfill": 0.4375, "hoard": 0.4375}). From Monteiro Lunardi
            et al 2018 and European Commission (2015).
        repairing_learning_shape_factor, (default=-0.31). Estimated with data
            on repairing costs at different scales from JRC 2019.
        scndhand_mkt_pric_rate (a list for a triangular distribution) (ratio),
            (default=[0.4, 1, 0.7]). From unpublished study Wang et al.
        refurbisher_margin (ratio), (default=[0.03, 0.45, 0.24]). From Duvan
            & Aykaç 2008 and www.investopedia.com (accessed 03/2020).
        max_storage (a list for a triangular distribution) (years), (default=
            [1, 8, 4]). From Wilson et al. 2017.

    """

    def __init__(self, unique_id, model, original_repairing_cost,
                 init_eol_rate, repairing_learning_shape_factor,
                 scndhand_mkt_pric_rate, refurbisher_margin, max_storage):
        """
        Creation of new refurbisher agent
        """
        super().__init__(unique_id, model)
        self.original_repairing_cost = \
            np.random.triangular(original_repairing_cost[0],
                                 original_repairing_cost[2],
                                 original_repairing_cost[1])
        original_reused_volumes = [x / model.num_refurbishers * 1E6 for x
                                   in model.original_num_prod]
        #  Original repairing volume is based on previous years EoL volume
        # (from 2000 to 2019)
        self.original_repairing_volume = \
            self.model.repairability * (
                    init_eol_rate["repair"] + init_eol_rate["sell"]) * \
            sum(self.model.waste_generation(self.model.d_product_lifetimes,
                                            self.model.avg_failure_rate[2],
                                            original_reused_volumes))
        self.repairing_cost = self.original_repairing_cost
        self.refurbished_volume = 0
        self.repairing_shape_factor = repairing_learning_shape_factor
        #self.scndhand_mkt_pric_rate = \
         #   np.random.triangular(scndhand_mkt_pric_rate[0],
          #                       scndhand_mkt_pric_rate[2],
           #                      scndhand_mkt_pric_rate[1])
        self.scndhand_mkt_pric_rate = \
            float(truncnorm((0.11 - scndhand_mkt_pric_rate[0]) /
                            scndhand_mkt_pric_rate[1],
                            (1.14 - scndhand_mkt_pric_rate[0]) /
                            scndhand_mkt_pric_rate[1],
                            scndhand_mkt_pric_rate[0],
                            scndhand_mkt_pric_rate[1]).rvs(1))
        # attitude_level = float(distribution.rvs(1))
        self.refurbisher_margin = np.random.triangular(
            refurbisher_margin[0], refurbisher_margin[2],
            refurbisher_margin[1])
        self.scd_hand_price = self.scndhand_mkt_pric_rate * \
            self.model.fsthand_mkt_pric
        self.count_consumers = 0
        self.count_consumers_tot = 0
        self.storage_decision = False
        self.storage_yr = 0
        self.storage_yr_recycle = 0
        self.max_storage_ref = np.random.triangular(
            max_storage[0], max_storage[2], max_storage[1])
        self.hoarded_waste = 0
        self.hoarded_waste_mass = 0
        self.hoarded_to_other = 0
        self.repaired_then_sold = 0
        self.refurbished_volume_n_sold = 0
        self.ref_hoarded_waste = 0
        self.ref_hoarded_waste_mass = 0
        self.sold_waste_recycler = 0
        self.hoarded_waste_recycle = 0
        self.hoarded_waste_recycle_mass = 0
        self.prod_sold = 0
        self.prod_recycled = 0
        self.prod_landfilled = 0
        self.prod_hoarded = 0
        self.refurbisher_costs = 0
        self.refurbisher_costs_w_margins = 0
        self.revenue = 0

    def repairable_volumes(self):
        """
        Compute amount of waste that can be repaired (and thus sold).
        """
        self.refurbished_volume = 0
        total_volume_recyler = 0
        for agent in self.model.schedule.agents:
            if self.model.num_consumers <= agent.unique_id < \
                    self.model.num_consumers + self.model.num_recyclers:
                total_volume_recyler += agent.repairable_volume
            if agent.unique_id < self.model.num_consumers:
                if agent.refurbisher_id == self.unique_id and \
                        agent.EoL_pathway == "repair":
                    self.refurbished_volume += agent.number_product_EoL
        self.refurbished_volume += total_volume_recyler / \
            self.model.num_refurbishers
        self.refurbished_volume_n_sold = self.refurbished_volume + \
            self.repaired_then_sold
        self.repaired_then_sold = 0

    def refurbisher_learning_curve_function(self):
        """
        Run the learning curve function from recyclers with refurbishers
        parameters.
        """
        self.repairing_cost = \
            Recyclers.learning_curve_function(
                self, self.original_repairing_volume,
                self.refurbished_volume_n_sold, self.original_repairing_cost,
                self.repairing_shape_factor)

    def count_consumers_refurbisher(self):
        """
        Count the refurbisher's number of consumers (refurbisher's customers)
        that sells products as well as the total number of consumers assigned
        to that refurbisher.
        """
        self.count_consumers = 0
        self.count_consumers_tot = 0
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers:
                if agent.refurbisher_id == self.unique_id and \
                        agent.EoL_pathway == "sell":
                    self.count_consumers += 1
                if agent.refurbisher_id == self.unique_id:
                    self.count_consumers_tot += 1

    def refurbisher_landfill_storage(self):
        """
        Choose what to do with products that were sold by consumers. After
        adding repair and transportation costs, if it's still profitable
        products are sold. Otherwise, divert waste that cannot be repaired or
        sold (either due to insufficient demand or repairs that are too costly)
        to the landfill, storage, and recycle pathways.
        """
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers:
                if agent.refurbisher_id == self.unique_id and \
                        agent.EoL_pathway == "sell":
                    eol_refurbisher = \
                        self.economic_rationale_tpb(agent, False)
                    if eol_refurbisher == "hoard":
                        self.storage_yr += 1
                    volume = (agent.number_product_EoL +
                              agent.product_storage_to_other_ref)
                    mass_volume = agent.mass_per_function_model(agent.waste) \
                        + agent.weighted_average_mass_watt * \
                        agent.product_storage_to_other_ref
                    self.update_volumes_eol(agent, eol_refurbisher,
                                            volume, mass_volume, False)

    def storage_to_other_pathway(self):
        """
        Divert waste from consumers that cannot be repaired or sold (either
        due to insufficient demand or repairs that are too costly) to the
        landfill, and recycle pathways after they have been stored.
        """
        if self.storage_yr > self.max_storage_ref:
            self.storage_yr = 0
            self.ref_hoarded_waste = 0
            hoarded_waste_copy = self.hoarded_waste
            hoarded_waste_copy_mass = self.hoarded_waste_mass
            for agent in self.model.schedule.agents:
                if agent.unique_id < self.model.num_consumers:
                    if agent.refurbisher_id == self.unique_id and \
                            agent.EoL_pathway == "sell":
                        self.ref_hoarded_waste = min(
                            agent.number_product_hoarded,
                            hoarded_waste_copy / self.count_consumers)
                        self.ref_hoarded_waste_mass = min(
                            agent.number_new_prod_hoarded,
                            hoarded_waste_copy_mass / self.count_consumers)
                        eol_refurbisher_stored = \
                            self.economic_rationale_tpb(agent, True)
                        self.update_volumes_eol(
                            agent, eol_refurbisher_stored,
                            self.ref_hoarded_waste,
                            self.ref_hoarded_waste_mass, True)
                        self.hoarded_waste -= self.ref_hoarded_waste
                        self.hoarded_waste_mass -= self.ref_hoarded_waste_mass

    def economic_rationale_tpb(self, agent, storage):
        """
        Model refurbisher decision to sell or divert waste to landfill,
        storage, or recycling according to the costs of each pathways (choose
        the cheapest pathway).
        """
        tpb_scores = agent.copy_perceived_behavioral_control.copy()
        tpb_scores[1] = -1 * self.scd_hand_price + \
            self.repairing_cost + \
            agent.random_interstate_distance * \
            self.model.transportation_cost / 1E3 * \
            self.model.dynamic_product_average_wght
        pathways_and_BI = {
            list(self.model.all_EoL_pathways.keys())[i]:
                tpb_scores[i] for i in range(
                len(list(self.model.all_EoL_pathways.keys())))}
        if storage:
            pathways_and_BI.pop("hoard")
        conditions = False
        removed_choice = None
        while not conditions:
            if removed_choice is not None:
                pathways_and_BI.pop(removed_choice)
            key = min(pathways_and_BI.items(),
                      key=operator.itemgetter(1))[0]
            if self.model.all_EoL_pathways.get(key) and \
                    key != "repair":
                return key
            else:
                removed_choice = key

    def update_volumes_eol(self, agent, eol_path, volume, mass_volume,
                           storage):
        """
        Update volumes of waste from consumers in each end of life pathway
        according to the decision made above.
        """
        if eol_path == "recycle":
            agent.number_product_recycled += volume
            agent.number_new_prod_recycled += mass_volume
            self.prod_recycled += volume
            if storage:
                agent.number_product_hoarded -= volume
                agent.number_new_prod_hoarded -= mass_volume
                self.prod_hoarded -= volume
            else:
                agent.number_product_sold -= volume
                agent.number_new_prod_sold -= mass_volume
        elif eol_path == "landfill":
            agent.number_product_landfilled += volume
            agent.number_new_prod_landfilled += mass_volume
            self.prod_landfilled += volume
            if storage:
                agent.number_product_hoarded -= volume
                agent.number_new_prod_hoarded -= mass_volume
                self.prod_hoarded -= volume
            else:
                agent.number_product_sold -= volume
                agent.number_new_prod_sold -= mass_volume
        elif eol_path == "sell":
            if storage:
                agent.number_product_sold += volume
                agent.number_product_hoarded -= volume
                agent.number_new_prod_sold += mass_volume
                agent.number_new_prod_hoarded -= mass_volume
                self.prod_sold += volume
            else:
                self.repaired_then_sold += volume
                self.prod_sold += volume
        elif eol_path == "hoard":
            agent.number_product_hoarded += volume
            agent.number_new_prod_hoarded += mass_volume
            agent.number_product_sold -= volume
            agent.number_new_prod_sold -= mass_volume
            self.hoarded_waste += volume
            self.hoarded_waste_mass += mass_volume
            self.prod_hoarded += volume

    def product_from_recycler(self):
        """
        Choose what to do with products that were sold by recyclers. After
        adding repair and transportation costs, if it's still profitable
        products are sold. Otherwise, divert waste that cannot be repaired or
        sold (either due to insufficient demand or repairs that are too costly)
        to the landfill, storage, and recycle pathways.
        """
        self.sold_waste_recycler = self.model.yearly_repaired_waste / \
                                   self.model.num_consumers
        mass_volume_recycler = self.sold_waste_recycler * \
                               self.model.dynamic_product_average_wght
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers:
                if agent.refurbisher_id == self.unique_id:
                    eol_ref_recycled_vol = \
                        self.economic_rationale_tpb(agent, False)
                    if eol_ref_recycled_vol == "hoard":
                        self.storage_yr_recycle += 1
                    self.update_volumes_eol_recycled(
                        agent, eol_ref_recycled_vol, self.sold_waste_recycler,
                        mass_volume_recycler, False)

    def storage_to_other_pathway_recycler(self):
        """
        Divert waste from recyclers that cannot be repaired or sold (either
        due to insufficient demand or repairs that are too costly) to the
        landfill, and recycle pathways after they have been stored.
        """
        if self.storage_yr_recycle > self.max_storage_ref:
            self.storage_yr = 0
            self.ref_hoarded_waste = 0
            hoarded_waste_copy = self.hoarded_waste_recycle
            hoarded_waste_copy_mass = self.hoarded_waste_recycle_mass
            for agent in self.model.schedule.agents:
                if agent.unique_id < self.model.num_consumers:
                    if agent.refurbisher_id == self.unique_id:
                        self.ref_hoarded_waste = \
                            hoarded_waste_copy / self.count_consumers_tot
                        self.ref_hoarded_waste_mass = \
                            hoarded_waste_copy_mass / self.count_consumers_tot
                        eol_ref_recycled_vol_stored = \
                            self.economic_rationale_tpb(agent, True)
                        self.update_volumes_eol_recycled(
                            agent, eol_ref_recycled_vol_stored,
                            self.ref_hoarded_waste,
                            self.ref_hoarded_waste_mass, True)
                        self.hoarded_waste_recycle -= self.ref_hoarded_waste
                        self.hoarded_waste_recycle_mass -= \
                            self.ref_hoarded_waste_mass

    def update_volumes_eol_recycled(self, agent, eol_path, volume,
                                    mass_volume_recycler, storage):
        """
        Update volumes of waste from recyclers in each end of life pathway
        according to the decision made above.
        """
        if eol_path == "recycle" and storage:
            agent.number_product_recycled += volume
            agent.number_product_hoarded -= volume
            agent.number_new_prod_recycled += mass_volume_recycler
            agent.number_new_prod_hoarded -= mass_volume_recycler
            self.prod_recycled += volume
            self.prod_hoarded -= volume
        elif eol_path == "landfill":
            agent.number_product_landfilled += volume
            agent.number_new_prod_landfilled += mass_volume_recycler
            self.prod_landfilled += volume
            if storage:
                agent.number_product_hoarded -= volume
                agent.number_new_prod_hoarded -= mass_volume_recycler
                self.prod_hoarded -= volume
            else:
                agent.number_product_recycled -= volume
                agent.number_new_prod_recycled -= mass_volume_recycler
        elif eol_path == "sell":
            agent.number_product_sold += volume
            agent.number_new_prod_sold += mass_volume_recycler
            self.prod_sold += volume
            if storage:
                agent.number_product_hoarded -= volume
                agent.number_new_prod_hoarded -= mass_volume_recycler
                self.prod_hoarded -= volume
            else:
                agent.number_product_recycled -= volume
                agent.number_new_prod_recycled -= mass_volume_recycler
        elif eol_path == "hoard":
            agent.number_product_hoarded += volume
            agent.number_new_prod_hoarded += mass_volume_recycler
            agent.number_product_recycled -= volume
            agent.number_new_prod_recycled -= mass_volume_recycler
            self.hoarded_waste_recycle += volume
            self.hoarded_waste_recycle_mass += mass_volume_recycler
            self.prod_hoarded += volume

    def compute_refurbisher_costs(self):
        """
        Compute societal costs of refurbishers. Assume an average of all
        refurbisher's customers' characteristics when computing costs.
        """
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers and \
                    agent.refurbisher_id == self.unique_id and \
                    agent.EoL_pathway == "sell":
                revenue = \
                    -1 * self.scd_hand_price + self.repairing_cost + \
                    agent.random_interstate_distance * \
                    self.model.transportation_cost / 1E3 * \
                    self.model.dynamic_product_average_wght
                cost_recycling = agent.copy_perceived_behavioral_control[2]
                cost_landfilling = \
                    agent.copy_perceived_behavioral_control[3]
                cost_hoarding = agent.copy_perceived_behavioral_control[4]
                self.refurbisher_costs += \
                    (revenue * self.prod_sold + cost_recycling *
                     self.prod_recycled + cost_landfilling *
                     self.prod_landfilled + cost_hoarding *
                     self.prod_hoarded) / self.count_consumers
                self.refurbisher_costs_w_margins += \
                    (revenue * self.prod_sold * self.refurbisher_margin +
                     cost_recycling * self.prod_recycled + cost_landfilling *
                     self.prod_landfilled + cost_hoarding *
                     self.prod_hoarded) / self.count_consumers

    def recovered_material_volumes(self):
        """
        Update amount recycled after refurbishers make their decision regarding
        waste sold to them.
        """
        # Last refurbisher calls producers to update amount of waste recycled
        if self.unique_id == self.model.num_consumers + \
                self.model.num_prod_n_recyc + self.model.num_refurbishers - 1:
            for agent in self.model.schedule.agents:
                if agent.unique_id < self.model.num_consumers:
                    agent.update_yearly_recycled_waste(True)
                if self.model.num_consumers + self.model.num_recyclers \
                        <= agent.unique_id < self.model.num_prod_n_recyc + \
                        self.model.num_consumers:
                    agent.add_installer_recycled_volumes()
                    agent.recovered_volume_n_value()
                    agent.costs_producer()
            for agent in self.model.schedule.agents:
                if self.model.num_consumers <= agent.unique_id < \
                        self.model.num_consumers + self.model.num_recyclers:
                    agent.compute_recycler_costs()

    def update_price_scdhand(self):
        self.scd_hand_price = self.scndhand_mkt_pric_rate * \
                              self.model.fsthand_mkt_pric

    def step(self):
        """
        Evolution of agent at each step
        """
        self.prod_hoarded = 0
        self.prod_landfilled = 0
        self.prod_recycled = 0
        self.prod_sold = 0
        self.repairable_volumes()
        self.refurbisher_learning_curve_function()
        self.count_consumers_refurbisher()
        self.refurbisher_landfill_storage()
        self.storage_to_other_pathway()
        self.product_from_recycler()
        self.storage_to_other_pathway_recycler()
        self.recovered_material_volumes()
        self.compute_refurbisher_costs()
        self.update_price_scdhand()
