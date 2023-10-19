# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Consumer
"""

from mesa import Agent
import numpy as np
import pandas as pd
import random
from collections import OrderedDict
from scipy.stats import truncnorm
import operator
from math import e



class Consumers(Agent):
    """
    A residential (or non-residential) owner of a product (e.g. PV,
    electronics) which dispose of it at its end of life and buy a first-hand
    or a second-hand product according to the Theory of Planned Behavior (TPB).

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_CE_PV_Model)
        product_growth (a list for a piecewise function) (ratio), (default=
            [0.166, 0.045]). From IRENA-IEA 2016
        failure_rate_alpha (a list for a triangular distribution), (default=
            [2.4928, 5.3759, 3.93495]). From IRENA-IEA 2016.
        perceived_behavioral_control (a list containing costs of each end of
            life (EoL) pathway)
        w_sn_eol (the weight of subjective norm in the agents' decisions as
            modeled with the theory of planned behavior), (default=0.33). From
            Geiger et al. 2019.
        w_pbc_eol (the weight of perceived behavioral control in the agents'
            decisions as modeled with the theory of planned behavior), (
            default=0.39). From Geiger et al. 2019.
        w_a_eol (the weight of attitude in the agents' decisions as modeled
            with the theory of planned behavior), (default=0.34). From
            Geiger et al. 2019.
        w_sn_reuse (same as above but for remanufactured product purchase
            decision), (default=0.497). From Singhal et al. 2019.
        w_pbc_reuse (same as above but for remanufactured product purchase
            decision), (default=0.382). From Singhal et al. 2019.
        w_a_reuse (same as above but for remanufactured product purchase
            decision), (default=0.464). From Singhal et al. 2019.
        product_lifetime (years), (default=30). From IRENA-IEA 2016.
        landfill_cost (a list for a triangular distribution) ($/fu), (default=
            [0.003, 0.009, 0.006]). From EPRI 2018.
        hoarding_cost (a list for a triangular distribution) ($/fu), (default=
            [0, 0.001, 0.0005]). From www.cisco-eagle.com (accessed 12/2019).
        used_product_substitution_rate (a list for a triangular distribution)
            (ratio), (default=[0.6, 1, 0.8]). From unpublished study Wang et
            al.
        att_distrib_param_eol (a list for a bounded normal distribution), (
            default=[0.53, 0.12]). From model's calibration step (mean),
            Saphores 2012 (standard deviation).
        att_distrib_param_eol (a list for a bounded normal distribution), (
            default=[0.35, 0.2]). From model's calibration step (mean),
            Abbey et al. 2016 (standard deviation).
        max_storage (a list for a triangular distribution) (years), (default=
            [1, 8, 4]). From Wilson et al. 2017.
        consumers_distribution (allocation of different types of consumers),
            (default={"residential": 1, "commercial": 0., "utility": 0.}).
            (Other possible values based on EIA, 2019 and SBE council, 2019:
            residential=0.75, commercial=0.2 and utility=0.05).
        product_distribution (ratios of product among consumer types), (default
            ={"residential": 1, "commercial": 0., "utility": 0.}). (Other
            possible values based on Bolinger et al. 2018: residential=0.21,
            commercial=0.18 and utility=0.61).

    """

    def __init__(self, unique_id, model, product_growth, failure_rate_alpha,
                 perceived_behavioral_control, w_sn_eol, w_pbc_eol, w_a_eol,
                 w_sn_reuse, w_pbc_reuse, w_a_reuse, landfill_cost,
                 hoarding_cost, used_product_substitution_rate,
                 att_distrib_param_eol, att_distrib_param_reuse, max_storage,
                 consumers_distribution, product_distribution):
        """
        Creation of new consumer agent
        """
        super().__init__(unique_id, model)
        self.breed = "residential"
        self.consumers_distribution = consumers_distribution
        self.trust_levels = []
        self.number_product_EoL = 0
        self.number_used_product_EoL = 0
        self.tot_prod_EoL = 0
        self.number_product_repaired = 0
        self.number_product_sold = 0
        self.number_product_recycled = 0
        self.number_product_landfilled = 0
        self.number_product_hoarded = 0
        self.number_new_prod_repaired = 0
        self.number_new_prod_sold = 0
        self.number_new_prod_recycled = 0
        self.number_new_prod_landfilled = 0
        self.number_new_prod_hoarded = 0
        self.number_used_prod_repaired = 0
        self.number_used_prod_sold = 0
        self.number_used_prod_recycled = 0
        self.number_used_prod_landfilled = 0
        self.number_used_prod_hoarded = 0
        self.product_storage_to_other = 0
        self.product_years_storage = []
        self.max_storage = np.random.triangular(max_storage[0], max_storage[2],
                                                max_storage[1])
        self.number_product_new = 0
        self.number_product_used = 0
        self.number_product_certified = 0
        self.EoL_pathway = self.initial_choice(self.model.init_eol_rate)
        self.used_EoL_pathway = self.EoL_pathway
        self.purchase_choice = self.initial_choice(
            self.model.init_purchase_choice)

        # ! This increases model resolution nothing to do here for now
        self.pca = self.model.agents[self.unique_id][0]
        self.state = self.model.agents[self.unique_id][1]
        self.agents_per_pca = self.model.agents[self.unique_id][2]
        pca_recyc_transp_dist = self.model.recycler_distance_df.copy()
        pca_recyc_transp_dist = pca_recyc_transp_dist[self.pca]
        pca_recyc_transp_dist = pca_recyc_transp_dist.to_list()
        self.pca_recyc_transp_dist = min(pca_recyc_transp_dist)
        self.pca_recyc_transp_cost = self.pca_recyc_transp_dist * \
            self.model.transportation_cost / 1E3 * \
            self.model.dynamic_product_average_wght
        # ! TODO: change landfill costs
        pca_landfill_transp_dist = self.model.landfill_distance_df.copy()
        pca_landfill_transp_dist = pca_landfill_transp_dist[self.pca]
        pca_landfill_transp_dist = pca_landfill_transp_dist.to_list()
        self.pca_landfill_transp_dist = min(pca_landfill_transp_dist)
        self.pca_landfill_transp_cost = self.pca_landfill_transp_dist * \
            self.model.transportation_cost / 1E3 * \
            self.model.dynamic_product_average_wght
        landfill_name = self.model.landfill_distance_df.loc[
            self.model.landfill_distance_df[self.pca] ==
            self.pca_landfill_transp_dist, 'Facility Name'].iloc[0]
        landfills_data = self.model.landfill_cost_df.copy()
        self.landfill_cost = landfills_data.loc[
            landfills_data['Facility Name'] == landfill_name,
            '$/ Ton'].iloc[0]  # in $/ton
        self.landfill_cost = self.landfill_cost / 1E3 * \
            self.model.dynamic_product_average_wght  # $/W
        # self.init_landfill_cost = self.landfill_cost

        # ! prepare pvice waste outputs
        self.data_out_pca = pd.read_csv(
            "dataOut_95-by-35.Adv_" + self.pca + "_.csv")
        self.data_out_pca['Yearly_Sum_Power_atEOL'] /= self.agents_per_pca
        self.data_out_pca['Yearly_Sum_Area_atEOL'] /= self.agents_per_pca
        # ! modified the initial number of products & prepare pvice outputs
        self.data_in_pca = pd.read_csv(
            "datain_95-by-35.Adv_" + self.pca + "_.csv")
        self.data_in_pca['new_Installed_Capacity_[MW]'] /= self.agents_per_pca
        self.data_in_pca['new_Installed_Capacity_[MW]'] *= 1E6
        subset_df_cap = self.data_in_pca.copy()
        subset_df_cap = subset_df_cap[
            subset_df_cap['year'] < (2020 + self.model.clock)]
        self.number_product = subset_df_cap[
            'new_Installed_Capacity_[MW]'].to_list()

        self.number_product_hard_copy = self.number_product.copy()
        self.product_distribution = product_distribution
        self.new_products = self.number_product.copy()
        self.new_products_hard_copy = self.new_products.copy()
        self.new_products_mass = \
            self.mass_per_function_model(self.new_products_hard_copy)
        self.used_products = [0] * len(self.number_product)
        self.used_products_hard_copy = self.used_products.copy()
        self.used_products_mass = \
            self.mass_per_function_model(self.used_products_hard_copy)
        self.product_growth_list = product_growth
        self.used_product_substitution_rate = \
            np.random.triangular(used_product_substitution_rate[0],
                                 used_product_substitution_rate[2],
                                 used_product_substitution_rate[1])
        self.product_growth = self.product_growth_list[0]
        self.failure_rate_alpha = \
            np.random.triangular(failure_rate_alpha[0], failure_rate_alpha[2],
                                 failure_rate_alpha[1])
        self.perceived_behavioral_control = perceived_behavioral_control
        self.copy_perceived_behavioral_control = \
            self.perceived_behavioral_control.copy()
        self.w_sn_eol = w_sn_eol
        self.w_pbc_eol = w_pbc_eol
        self.w_a_eol = w_a_eol
        self.w_sn_reuse = w_sn_reuse
        self.w_pbc_reuse = w_pbc_reuse
        self.w_a_reuse = w_a_reuse
        if self.EoL_pathway == "landfill" or self.EoL_pathway == "hoard":
            model.color_map.append('blue')
        else:
            model.color_map.append('green')
        self.recycling_facility_id = model.num_consumers + random.randrange(
            model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
            random.randrange(model.num_refurbishers)
        # self.landfill_cost = random.choice(landfill_cost)
        # self.landfill_cost = np.random.triangular(
        #   landfill_cost[0], landfill_cost[2], landfill_cost[1])

        # HERE
        self.hoarding_cost = np.random.triangular(
            hoarding_cost[0], hoarding_cost[2], hoarding_cost[1]) * \
            self.max_storage

        # self.hoarding_cost = \
        #    float(truncnorm((0 - hoarding_cost[0]) /
        #                    hoarding_cost[1],
        #                    (0.02 - hoarding_cost[0]) /
        #                    hoarding_cost[1],
        #                    hoarding_cost[0],
        #                    hoarding_cost[1]).rvs(1)) * self.max_storage
        # HERE

        self.attitude_level = \
            self.attitude_level_distribution((0 - att_distrib_param_eol[0]) /
                                             att_distrib_param_eol[1],
                                             (1 - att_distrib_param_eol[0]) /
                                             att_distrib_param_eol[1],
                                             att_distrib_param_eol[0],
                                             att_distrib_param_eol[1])
        self.attitude_levels_pathways = [0] * len(self.model.all_EoL_pathways)
        self.attitude_level_reuse = \
            self.attitude_level_distribution((0 - att_distrib_param_reuse[0]) /
                                             att_distrib_param_reuse[1],
                                             (1 - att_distrib_param_reuse[0]) /
                                             att_distrib_param_reuse[1],
                                             att_distrib_param_reuse[0],
                                             att_distrib_param_reuse[1])
        self.purchase_choices = list(self.model.purchase_options.keys())
        self.attitude_levels_purchase = [0] * len(self.purchase_choices)
        self.pbc_reuse = [self.model.fsthand_mkt_pric, np.nan,
                          self.model.fsthand_mkt_pric]
        self.distances_to_customers = []
        self.distances_to_customers = self.model.shortest_paths(
            [random.choice(self.model.all_states)],
            self.distances_to_customers)
        self.random_interstate_distance = random.choice(
            self.distances_to_customers)
        self.agent_breed()
        self.product_storage_to_other_ref = 0
        self.weighted_average_mass_watt = 0
        self.consumer_costs = 0
        self.past_recycled_waste = 0
        self.yearly_recycled_waste = 0
        self.sold_waste = 0
        self.convenience = self.extended_tpb_convenience()
        self.knowledge = self.extended_tpb_knowledge()

    def update_transport_costs(self):
        """
        Update transportation costs according to the (evolving) mass of waste.
        """
        self.pca_recyc_transp_cost = self.pca_recyc_transp_dist * \
            self.model.transportation_cost / 1E3 * \
            self.model.dynamic_product_average_wght
        self.pca_landfill_transp_cost = self.pca_landfill_transp_dist * \
            self.model.transportation_cost / 1E3 * \
            self.model.dynamic_product_average_wght
        # self.landfill_cost = \
        #    self.init_landfill_cost + \
        #    (self.model.dynamic_product_average_wght -
        #     self.model.product_average_wght) * \
        #    self.model.transportation_cost / 1E3 * \
        #    self.model.mean_distance_within_state

    def attitude_level_distribution(self, a, b, loc, scale):
        """
        Distribute pro-environmental attitude level toward the decision in the
        population.
        """
        distribution = truncnorm(a, b, loc, scale)
        attitude_level = float(distribution.rvs(1))
        return attitude_level

    def extended_tpb_convenience(self):
        """
        Compute the convenience factor of the theory of planned behavior as a
        function of the distance necessary to perform the behavior.
        All pathways are assumed to be within the states  except for recycling
        which is approximated to the distance to the nearest recycler
        (and assumed to be independent from the recycling costs).
        """
        # A small constant is added to avoid np.random.triangular error
        recyc_dist = np.random.triangular(
            self.model.mn_mx_av_distance_to_recycler[0],
            self.model.mn_mx_av_distance_to_recycler[2],
            self.model.mn_mx_av_distance_to_recycler[1] + 0.001)
        convenience = [0, 0, (recyc_dist -
                              self.model.mn_mx_av_distance_to_recycler[0]) /
                       self.model.mn_mx_av_distance_to_recycler[1], 0, 0]
        convenience = [self.model.extended_tpb["w_convenience"] * x for x in
                       convenience]
        return convenience

    def extended_tpb_knowledge(self):
        """
        Distribute end-of-life management knowledge among agents.
        """
        loc = self.model.extended_tpb["knowledge_distrib"][0]
        scale = self.model.extended_tpb["knowledge_distrib"][1]
        distribution = truncnorm((0 - loc) / scale, (1 - loc) / scale,
                                 loc, scale)
        knowledge_level = float(distribution.rvs(1))
        knowledge_eol = [knowledge_level, knowledge_level, knowledge_level,
                         0, 0]
        knowledge_eol = [self.model.extended_tpb["w_knowledge"] * x for x in
                         knowledge_eol]
        return knowledge_eol

    def initial_choice(self, list_choice):
        """
        Initiate the EoL pathway and purchase choice chosen by agents.
        """
        total = 0
        u_id = self.model.list_consumer_id[self.unique_id]
        for key, value in list_choice.items():
            total += value * self.model.num_consumers
            if u_id <= (total - 1):
                return key

    def agent_breed(self):
        """
        Distribute the agent type (residential, non-residential).
        """
        u_id = self.model.list_consumer_id[self.unique_id]
        if u_id < round(self.model.num_consumers *
                        self.consumers_distribution["commercial"]):
            self.breed = "commercial"
        elif u_id < \
                round(self.model.num_consumers *
                      (self.consumers_distribution["commercial"] +
                       self.consumers_distribution["utility"])):
            self.breed = "utility"
        self.number_product = [x / self.consumers_distribution[self.breed] *
                               self.product_distribution[self.breed] for x in
                               self.number_product]
        if not self.model.theory_of_planned_behavior[self.breed]:
            self.w_sn_eol = 0
            self.w_a_eol = 0

    def update_product_stock(self):
        """
        Update stock according to product growth and product failure
        Product failure is modeled with the Weibull function
        """

        # ! TODO: replace code below with PV_ICE installed cap value - START

        # ! use the "Installed_Capacity_[W] [W] : float" output found in
        # ! https://pv-ice.readthedocs.io/en/latest/data.html#pv-ice-outputs
        # ! We should check what this variable means. Is it with or without
        # ! waste? Looks like it should be minus the waste. If it is minus the
        # ! waste we should add waste (Yearly_Sum_Power_disposed [W] : float)
        # ! to get cumulative installed capacity. Then subtracting previous
        # ! year should give you the value for additional_capacity.
        # ! Only "additional_capacity" should be changed below.
        # ! Should we just take the list from pv_ice data frame directly?
        # ! Create a new column in the data frame that has the cumulative
        # ! capacities (not subtracting waste)?

        # at t=0
        # previous_year = x
        # current_year = Installed_Capacity_[W] [W] + \
        #                   Yearly_Sum_Power_disposed [W]
        # additional_cap = current_year - previous_year
        # previous_year = current_year

        #                              2000, 2001, 2002, ..., 2020
        # self.model.initial_capacity = [5, 5, 5, 5, 52, 67, 45, 120] # MWp

        # ! Scratch all above: the easiest is to use PV_ICE inputs:

        # subset_df_init_cap = self.model.df0[
        #     self.model.df0['year'] == 2020 + self.model.clock]

        subset_df_cap = self.data_in_pca.copy()
        subset_df_cap = subset_df_cap[
            subset_df_cap['year'] == (2020 + self.model.clock)]
        additional_capacity = subset_df_cap[
            'new_Installed_Capacity_[MW]'].iloc[0]
        # ! Old code
        # additional_capacity = sum(self.number_product_hard_copy) * \
        #    self.product_growth

        self.number_product_hard_copy.append(additional_capacity)
        self.number_product.append(self.number_product_hard_copy[-1])
        self.new_products.append(self.number_product[-1])
        self.used_products.append(0)
        self.new_products_hard_copy.append(self.number_product[-1])
        self.used_products_hard_copy.append(0)

        # ! TODO: how to deal with used capacity and new capacity?
        # ! Nothing should be changed... The code below determines the share
        # ! of used vs new product based on new capacity and agents' decisions

        if self.purchase_choice == "used":
            product_substituted = (1 - self.model.imperfect_substitution) * \
                                  self.model.sold_repaired_waste / \
                                  self.model.consumer_used_product
            self.used_products[-1] = product_substituted
            self.used_products_hard_copy[-1] = product_substituted
            if self.new_products[-1] > product_substituted:
                self.new_products[-1] -= product_substituted
                self.new_products_hard_copy[-1] -= product_substituted
                self.model.sold_repaired_waste -= product_substituted
            else:
                self.new_products[-1] = 0
                self.new_products_hard_copy[-1] = 0
                self.model.sold_repaired_waste -= product_substituted

        # ! TODO: replace code below with PV_ICE installed cap value - STOP

        # ! TODO: replace code below with PV_ICE waste value - START

        # ! use the "Yearly_Sum_Power_disposed [W] : float" output found in
        # ! https://pv-ice.readthedocs.io/en/latest/data.html#pv-ice-outputs
        # ! self.waste and self.used_waste should be changed. A list the size
        # ! of self.number_product should be created from the
        # ! "Yearly_Sum_Power_disposed [W] : float" column in pv_ice output df
        # ! if self.number_product is longer (goes back further in time than
        # ! 1995) add zeros. If the list is shorter (does not go back as far)
        # ! add the additional years to make a single first year that
        # ! correspond to the first element (year) of self.number_product
        # ! self.used_waste proportional to the amount of used / new panels for
        # ! each year

        # sel.waste = df1.subset(row_index_for_year0, row_index_for_year_now,
        #                       column1, column1).to_list()
        # sum the first five years for PV_ICE

        # ! Implementation is below. Left to do: check with Silvana that
        # ! Yearly_Sum_Power_atEOL is the total waste generated in a year
        # ! expressed in W

        # ! Old code
        # self.number_product_EoL = sum(self.waste)
        # self.number_used_product_EoL = sum(self.used_waste)
        if type(self.used_products[-1]) != float and \
                type(self.used_products[-1]) != int:
            self.used_products[-1] = 0
        if self.new_products[-1] + self.used_products[-1] == 0:
            self.used_new_ratio = 0
        else:
            self.used_new_ratio = self.used_products[-1] / (
                self.new_products[-1] + self.used_products[-1])

        yearly_waste_file = self.data_out_pca.copy()
        if self.model.clock == 0:
            yearly_waste = yearly_waste_file[
                yearly_waste_file['year'] <= (2020 + self.model.clock)]
            yearly_waste = sum(
                yearly_waste['Yearly_Sum_Power_atEOL'].tolist())
            self.number_product_EoL = yearly_waste * (
                1 - self.used_new_ratio)
            self.number_used_product_EoL = yearly_waste * self.used_new_ratio
            yearly_waste_m2 = yearly_waste_file[
                yearly_waste_file['year'] <= (2020 + self.model.clock)]
            yearly_waste_m2 = sum(
                yearly_waste_m2['Yearly_Sum_Area_atEOL'].tolist())
            self.number_product_EoL_m2 = yearly_waste_m2 * (
                1 - self.used_new_ratio)
            self.number_used_product_EoL_m2 = yearly_waste_m2 * \
                self.used_new_ratio
        else:
            yearly_waste = yearly_waste_file[
                yearly_waste_file['year'] == (2020 + self.model.clock)]
            self.number_product_EoL = yearly_waste[
                'Yearly_Sum_Power_atEOL'].iloc[0] * (1 - self.used_new_ratio)
            self.number_used_product_EoL = yearly_waste[
                'Yearly_Sum_Power_atEOL'].iloc[0] * self.used_new_ratio
            yearly_waste_m2 = yearly_waste_file[
                yearly_waste_file['year'] == (2020 + self.model.clock)]
            self.number_product_EoL_m2 = yearly_waste_m2[
                'Yearly_Sum_Area_atEOL'].iloc[0] * (1 - self.used_new_ratio)
            self.number_used_product_EoL_m2 = yearly_waste[
                'Yearly_Sum_Area_atEOL'].iloc[0] * self.used_new_ratio
        self.tot_prod_EoL = self.number_product_EoL + \
            self.number_used_product_EoL
        self.tot_prod_EoL_m2 = self.number_product_EoL_m2 + \
            self.number_used_product_EoL_m2

        subset_df_remaining_cap = self.data_out_pca.copy()
        subset_df_remaining_cap = subset_df_remaining_cap[
            subset_df_remaining_cap['year'] <= (2020 + self.model.clock)]
        self.new_products = [
            x * (1 - self.used_new_ratio) for x in subset_df_remaining_cap[
                'Effective_Capacity_[W]'].tolist()]
        self.used_products = [
            x * self.used_new_ratio for x in subset_df_remaining_cap[
                'Effective_Capacity_[W]'].tolist()]
        self.number_product = [x + y for x, y in zip(
            self.new_products, self.used_products)]

        # ! TODO: replace code below with PV_ICE waste value - STOP

    def tpb_subjective_norm(self, decision, list_choices, weight_sn):
        """
        Calculate subjective norm (peer pressure) component of EoL TPB rule
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        proportions_choices = []
        for i in range(len(list_choices)):
            proportion_choice = len([
                agent for agent in
                self.model.grid.get_cell_list_contents(neighbors_nodes)
                if getattr(agent, decision) == list_choices[i]]) / \
                                len([agent for agent in
                                     self.model.grid.get_cell_list_contents(
                                                   neighbors_nodes)])
            proportions_choices.append(proportion_choice)
        return [weight_sn * x for x in proportions_choices]

    def tpb_perceived_behavioral_control(self, decision, pbc_choice,
                                         weight_pbc):
        """
        Calculate perceived behavioral control component of EoL TPB rule.
        Following Ghali et al. 2017 and Labelle et al. 2018, perceived
        behavioral control is understood as a function of financial costs.
        """
        max_cost = max(abs(i) for i in pbc_choice)
        pbc_choice = [i / max_cost for i in pbc_choice]
        if decision == "EoL_pathway":
            self.repairable_modules(pbc_choice)
            if self.model.extended_tpb["Extended tpb"]:
                pbc_choice = \
                    [self.convenience[i] + self.knowledge[i] + pbc_choice[i]
                     for i in range(len(pbc_choice))]
                max_cost = max(abs(i) for i in pbc_choice)
                pbc_choice = [i / max_cost for i in pbc_choice]
        return [weight_pbc * -1 * max(i, 0) for i in pbc_choice]

    def tpb_attitude(self, decision, att_levels, att_level, weight_a):
        """
        Calculate pro-environmental attitude component of EoL TPB rule. Options
        considered pro environmental get a higher score than other options.
        """
        for i in range(len(att_levels)):
            if decision == "EoL_pathway":
                if list(self.model.all_EoL_pathways.keys())[i] == "repair" or \
                        list(self.model.all_EoL_pathways.keys())[i] == "sell" \
                        or list(self.model.all_EoL_pathways.keys())[i] == \
                        "recycle":
                    att_levels[i] = att_level
                    # HERE modification for encouraging recycling
                    # if list(self.model.all_EoL_pathways.keys())[i] ==
                    # "recycle":
                    # att_levels[i] = att_level * 1.0
                else:
                    att_levels[i] = 1 - att_level
            elif decision == "purchase_choice":
                if self.purchase_choices[i] == "used" or \
                        self.purchase_choices[i] == "certified":
                    att_levels[i] = att_level
                else:
                    att_levels[i] = 1 - att_level
        return [weight_a * x for x in att_levels]

    def yearly_prod_n_waste(self):
        """
        Update total waste generated an yearly production.
        """
        self.model.total_waste += self.tot_prod_EoL
        self.model.total_yearly_new_products += self.new_products[-1]

    def repairable_modules(self, pbc_choice):
        """
        Account for the fact that some panels cannot be repaired
        (and thus sold).
        """
        total_waste = 0
        self.sold_waste = 0
        total_volume_refurbished = 0
        for agent in self.model.schedule.agents:
            if self.model.num_consumers + self.model.num_prod_n_recyc <= \
                    agent.unique_id:
                total_volume_refurbished += agent.refurbished_volume
            if agent.unique_id < self.model.num_consumers:
                total_waste += agent.number_product_EoL
                if agent.EoL_pathway == "sell":
                    self.sold_waste += agent.number_product_EoL
        if self.sold_waste + total_volume_refurbished > \
                self.model.repairability * total_waste:
            pbc_choice[0] = 1
            pbc_choice[1] = 1

    def tpb_decision(self, decision, list_choices, avl_paths, weight_sn,
                     pbc_choice, weight_pbc, att_levels, att_level, weight_a):
        """
        Select the decision with highest behavioral intention following the
        Theory of Planned Behavior (TPB). Behavioral intention is a function
        of the subjective norm, the perceived behavioral control and attitude.
        """
        sn_values = self.tpb_subjective_norm(
            decision, list_choices, weight_sn)
        pbc_values = self.tpb_perceived_behavioral_control(
            decision, pbc_choice, weight_pbc)
        a_values = self.tpb_attitude(decision, att_levels, att_level, weight_a)
        self.behavioral_intentions = [(pbc_values[i]) + sn_values[i] +
                                      a_values[i] for i in
                                      range(len(pbc_values))]
        self.pathways_and_BI = {list_choices[i]: self.behavioral_intentions[i]
                                for i in
                                range(len(list_choices))}
        shuffled_dic = list(self.pathways_and_BI.items())
        random.shuffle(shuffled_dic)
        self.pathways_and_BI = OrderedDict(shuffled_dic)
        for key, value in self.pathways_and_BI.items():
            if value == np.nan:
                return self.EoL_pathway
        conditions = False
        removed_choice = None
        while not conditions:
            if removed_choice is not None:
                self.pathways_and_BI.pop(removed_choice)
            if decision == "purchase_choice":
                key = max(self.pathways_and_BI.items(),
                          key=operator.itemgetter(1))[0]
                if self.model.purchase_options.get(key):
                    return key
                else:
                    removed_choice = key
            else:
                key = max(self.pathways_and_BI.items(),
                          key=operator.itemgetter(1))[0]
                if avl_paths.get(key) and key != "sell":
                    return key
                else:
                    new_installed_capacity = 0
                    for agent in self.model.schedule.agents:
                        if agent.unique_id < self.model.num_consumers:
                            new_installed_capacity += agent.number_product[-1]
                    used_volume_purchased = self.model.consumer_used_product \
                        / self.model.num_consumers * new_installed_capacity
                if avl_paths.get(key) and key == "sell" and \
                        self.sold_waste < used_volume_purchased:
                    return key
                else:
                    removed_choice = key

    def volume_used_products_purchased(self):
        """
        Count amount of remanufactured product that are bought by consumers
        """
        self.purchase_choice = \
            self.tpb_decision(
                "purchase_choice", list(self.model.purchase_options.keys()),
                self.model.all_EoL_pathways, self.w_sn_reuse, self.pbc_reuse,
                self.w_pbc_reuse, self.attitude_levels_purchase,
                self.attitude_level_reuse, self.w_a_reuse)
        if self.model.seeding["Seeding"] and self.model.clock >= \
                self.model.seeding["Year"]:
            for consumer in range(self.model.seeding["number_seed"]):
                if self.unique_id == \
                        self.model.list_consumer_id_seed[consumer]:
                    second_hand_p = 0
                    repair_c = 0
                    for agent in self.model.schedule.agents:
                        if agent.unique_id == self.refurbisher_id:
                            second_hand_p = agent.scd_hand_price
                            repair_c = agent.repairing_cost
                    self.purchase_choice = "used"
                    self.model.cost_seeding += second_hand_p + repair_c + \
                        self.random_interstate_distance * \
                        self.model.transportation_cost / 1E3 * \
                        self.model.dynamic_product_average_wght
        if self.purchase_choice == "new":
            self.number_product_new += self.number_product[-1]
        elif self.EoL_pathway == "used":
            self.number_product_used += self.number_product[-1]
        else:
            self.number_product_certified += self.number_product[-1]

    def update_product_eol(self, product_type):
        """
        The amount of waste generated is taken from "update_product_stock"
        and attributed to the chosen EoL pathway
        """
        limited_paths = self.model.all_EoL_pathways.copy()
        if self.model.seeding_recyc["Seeding"] and self.model.clock >= \
                self.model.seeding_recyc["Year"]:
            for consumer in range(self.model.seeding_recyc["number_seed"]):
                if self.unique_id == \
                        self.model.list_consumer_id_seed[consumer]:
                    self.perceived_behavioral_control[2] *= \
                        self.model.seeding_recyc["discount"]
        if product_type == "new":
            self.storage_management(limited_paths)
            self.EoL_pathway = \
                self.tpb_decision(
                    "EoL_pathway", list(self.model.all_EoL_pathways.keys()),
                    limited_paths, self.w_sn_eol,
                    self.perceived_behavioral_control, self.w_pbc_eol,
                    self.attitude_levels_pathways, self.attitude_level,
                    self.w_a_eol)
            # HERE: self.number_product_EoL + self.product_storage_to_other
            self.update_eol_volumes(self.EoL_pathway,
                                    self.number_product_EoL +
                                    self.product_storage_to_other,
                                    product_type,
                                    self.product_storage_to_other)
        else:
            limited_paths["repair"] = False
            limited_paths["sell"] = False
            limited_paths["hoard"] = False
            self.used_EoL_pathway = \
                self.tpb_decision(
                    "EoL_pathway", list(self.model.all_EoL_pathways.keys()),
                    limited_paths, self.w_sn_eol,
                    self.perceived_behavioral_control, self.w_pbc_eol,
                    self.attitude_levels_pathways, self.attitude_level,
                    self.w_a_eol)
            self.update_eol_volumes(self.used_EoL_pathway,
                                    self.number_used_product_EoL,
                                    product_type,
                                    self.product_storage_to_other)

    def update_eol_volumes(self, eol_pathway, managed_waste, product_type,
                           storage):
        """"
        Assumes an average storage time for product stored. Also compute
        consumers' societal costs. Compute the costs paid by consumers. It
        uses end-of-life costs in $/functional_unit. Thus it assumes that
        end-of-life costs do not decrease with the decrease in
        mass/functional_unit.
        """
        # yearly_converting_factor_list = [
        #               pv_ice_waste_in_kg / pv_ice_waste_in_w, ..., year_n]
        # average_converting_factor = \
        # [yearly_pv_ice_waste_in_kg / pv_ice_waste_in_w, ..., year_n].mean()
        # # in kg/W
        # then instead of using:
        # self.mass_per_function_model(self.waste) +
        # self.weighted_average_mass_watt * storage we use:
        # self.number_new_prod_repaired = sum([x * y for x in
        # yearly_converting_factor_list and y in self.waste]) +
        # average_converting_factor * storage
        past_storage = max(0, (2020 + self.model.clock - self.max_storage))
        pv_ice_mat_subset_stored_years = self.model.pvice_mat_factor[
            (self.model.pvice_mat_factor['year'] >= past_storage) &
            (self.model.pvice_mat_factor['year'] <= 2020 + self.model.clock)]
        avg_weight_factor_stored_pv = pv_ice_mat_subset_stored_years[
            'total_massperm2'].mean()

        original_df = self.data_out_pca.copy()
        original_df = original_df[
            (original_df['year'] >= past_storage) &
            (original_df['year'] <= 2020 + self.model.clock)]
        waste_in_w = original_df['Yearly_Sum_Power_atEOL'].mean()
        waste_in_m2 = original_df['Yearly_Sum_Area_atEOL'].mean()
        waste_w_to_m2_factor = waste_in_m2 / waste_in_w

        if eol_pathway == "repair":
            self.number_product_repaired += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[0]
            if product_type == "new":
                self.number_new_prod_repaired += \
                    self.number_product_EoL_m2 * self.model.weight_factor + \
                    avg_weight_factor_stored_pv * storage * \
                    waste_w_to_m2_factor
            else:
                self.number_used_prod_repaired += \
                    self.number_used_product_EoL_m2 * self.model.weight_factor
            self.model.pca_outputs[self.pca][eol_pathway] = (
                self.number_new_prod_repaired +
                self.number_used_prod_repaired)
        elif eol_pathway == "sell":
            self.number_product_sold += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[1]
            if product_type == "new":
                self.number_new_prod_sold += \
                    self.number_product_EoL_m2 * self.model.weight_factor + \
                    avg_weight_factor_stored_pv * storage * \
                    waste_w_to_m2_factor
            else:
                self.number_used_prod_sold += \
                    self.number_used_product_EoL_m2 * self.model.weight_factor
            self.model.pca_outputs[self.pca][eol_pathway] = (
                self.number_new_prod_sold +
                self.number_used_prod_sold)
        elif eol_pathway == "recycle":
            self.number_product_recycled += managed_waste
            if not self.model.epr_business_model:
                self.consumer_costs += managed_waste * \
                                       self.perceived_behavioral_control[2]
            if product_type == "new":
                self.number_new_prod_recycled += \
                    self.number_product_EoL_m2 * self.model.weight_factor + \
                    avg_weight_factor_stored_pv * storage * \
                    waste_w_to_m2_factor
            else:
                self.number_used_prod_recycled += \
                    self.number_used_product_EoL_m2 * self.model.weight_factor
            self.model.pca_outputs[self.pca][eol_pathway] = (
                self.number_new_prod_recycled +
                self.number_used_prod_recycled)
        elif eol_pathway == "landfill":
            self.number_product_landfilled += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[3]
            if product_type == "new":
                self.number_new_prod_landfilled += \
                    self.number_product_EoL_m2 * self.model.weight_factor + \
                    avg_weight_factor_stored_pv * storage * \
                    waste_w_to_m2_factor
            else:
                self.number_used_prod_landfilled += \
                    self.number_used_product_EoL_m2 * self.model.weight_factor
            self.model.pca_outputs[self.pca][eol_pathway] = (
                self.number_new_prod_landfilled +
                self.number_used_prod_landfilled)
        else:
            # managed_waste = self.number_product_EoL or
            # self.number_product_EoL + storage_to_others
            self.number_product_hoarded += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[4]
            if product_type == "new":
                self.number_new_prod_hoarded += \
                    self.number_product_EoL_m2 * self.model.weight_factor
            else:
                self.number_used_prod_hoarded += \
                    self.number_used_product_EoL_m2 * self.model.weight_factor
            self.model.pca_outputs[self.pca][eol_pathway] = (
                self.number_new_prod_hoarded +
                self.number_used_prod_hoarded)

    def update_yearly_recycled_waste(self, installer):
        """
        Update consumers' amount of recycled waste.
        """
        self.yearly_recycled_waste = self.number_product_recycled - \
            self.past_recycled_waste
        if installer:
            self.past_recycled_waste = self.number_product_recycled

    def mass_per_function_model(self, product_as_function):
        """
        Convert end-of-life volume in Wp to kg. Account for the year the
        module was manufactured and the average weight-to-power ratio at that
        time. The model from IRENA-IEA 2016 is used.
        """
        past_storage = max(0, (2020 + self.model.clock - self.max_storage))
        pv_ice_mat_subset_stored_years = self.model.pvice_mat_factor[
            (self.model.pvice_mat_factor['year'] >= past_storage) &
            (self.model.pvice_mat_factor['year'] <= 2020 + self.model.clock)]
        original_df = self.data_out_pca.copy()
        original_df = original_df[
            (original_df['year'] >= past_storage) &
            (original_df['year'] <= 2020 + self.model.clock)]
        waste_in_w = original_df['Yearly_Sum_Power_atEOL'].mean()
        waste_in_m2 = original_df['Yearly_Sum_Area_atEOL'].mean()
        waste_w_to_m2_factor = waste_in_m2 / waste_in_w

        self.weighted_average_mass_watt = pv_ice_mat_subset_stored_years[
            'total_massperm2'].mean() * waste_w_to_m2_factor

        len_product_as_function = len(product_as_function)
        pvice_mat_factor_copy = self.model.pvice_mat_factor[
            self.model.pvice_mat_factor['year'] <= 2020 + self.model.clock]
        conversion_factors = \
            pvice_mat_factor_copy['total_massperm2'].to_list()
        conversion_factors = conversion_factors[-len_product_as_function:]
        data_out_pca_copy = self.data_out_pca[
            self.data_out_pca['year'] <= 2020 + self.model.clock]
        waste_in_w_list = \
            data_out_pca_copy['Yearly_Sum_Power_atEOL'].to_list()
        waste_in_m2_list = \
            data_out_pca_copy['Yearly_Sum_Area_atEOL'].to_list()
        waste_in_w_list = waste_in_w_list[-len_product_as_function:]
        waste_in_m2_list = waste_in_m2_list[-len_product_as_function:]
        waste_w_m2_list = [x / y if y != 0 else 0 for x, y in
                           zip(waste_in_m2_list, waste_in_w_list)]

        product_as_mass = [x * y * z for x, y, z in zip(
            product_as_function, conversion_factors, waste_w_m2_list)]
        mass_eol = sum(product_as_mass)
        return mass_eol

    def storage_management(self, limited_paths):
        """
        Decision to handle waste in one of the end of life pathway (except
        storage) after products have been stored.
        """
        if self.purchase_choice == "new":
            self.product_years_storage.append(self.EoL_pathway)
        elif self.purchase_choice == "used":
            self.product_years_storage.append("hoard")
        count = 0
        for i in range(len(self.product_years_storage)):
            if self.product_years_storage[i] == "hoard":
                count += 1
            elif count <= self.max_storage:
                count = 0
        if count > self.max_storage:
            self.product_years_storage = []
            self.product_storage_to_other = self.number_product_hoarded
            self.number_product_hoarded = 0
            self.number_used_prod_hoarded = 0
            self.number_new_prod_hoarded = 0
            limited_paths["hoard"] = False

    def update_perceived_behavioral_control(self):
        """
        Costs from each EoL pathway and purchase choice and related perceived
        behavioral control are updated according to processes from other agents
        or own initiated costs.
        """
        for agent in self.model.schedule.agents:
            if agent.unique_id == self.recycling_facility_id:
                self.perceived_behavioral_control[2] = \
                    agent.recycling_cost + self.pca_recyc_transp_cost
            elif agent.unique_id == self.refurbisher_id:
                self.perceived_behavioral_control[0] = \
                    agent.repairing_cost
                self.perceived_behavioral_control[1] = -1 * \
                    agent.scd_hand_price * (1 - agent.refurbisher_margin)
                self.pbc_reuse[1] = agent.scd_hand_price
        self.pbc_reuse[0] = self.model.fsthand_mkt_pric
        self.perceived_behavioral_control[3] = \
            self.landfill_cost + self.pca_landfill_transp_cost
        self.perceived_behavioral_control[4] = self.hoarding_cost

    def product_mass_output_metrics(self):
        """
        Account for new and used products' volumes in mass unit.
        """
        last_capacity_new = [0] * len(self.new_products_hard_copy)
        last_capacity_new[-1] = self.new_products_hard_copy[-1]
        last_capacity_used = [0] * len(self.used_products_hard_copy)
        last_capacity_used[-1] = self.used_products_hard_copy[-1]
        self.new_products_mass += \
            self.mass_per_function_model(last_capacity_new)
        self.used_products_mass += \
            self.mass_per_function_model(last_capacity_used)

    def step(self):
        """
        Evolution of agent at each step
        """
        self.product_mass_output_metrics()
        self.product_storage_to_other = 0
        self.product_storage_to_other_ref = 0
        self.update_transport_costs()
        # Update product growth from a list:
        if self.model.clock > self.model.growth_threshold:
            self.product_growth = self.product_growth_list[1]
        self.update_product_stock()
        self.yearly_prod_n_waste()
        self.update_perceived_behavioral_control()
        self.copy_perceived_behavioral_control = \
            self.perceived_behavioral_control.copy()
        self.volume_used_products_purchased()
        self.update_product_eol("new")
        self.product_storage_to_other_ref = self.product_storage_to_other
        self.update_product_eol("used")
