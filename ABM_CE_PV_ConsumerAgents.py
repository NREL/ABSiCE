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
from utils import TIMESTEP, transform_timeseries_timestep, GeneratorSize, ConsumerAgentResolution, get_number_of_days_in_timestep, MISSING_VALUE_COST
import os
from ABM_CE_PV_RecyclerAgents import Recyclers



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
        super().__init__(model)
        self.unique_id = unique_id
        self.breed = "residential"
        self.consumers_distribution = consumers_distribution
        self.trust_levels = []
        self.number_product_EoL = 0
        self.number_used_product_EoL = 0
        self.tot_prod_EoL = 0
        self.tot_prod_EoL_m2 = 0
        self.number_product_repaired = 0
        self.number_product_sold = 0
        self.number_product_recycled = 0
        self.number_product_landfilled = 0
        self.number_product_hoarded = 0
        self.number_product_hoarded_hazardous = 0
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
        self.product_years_storage_hazardous = []
        self.max_storage = np.random.triangular(max_storage[0], max_storage[2],
                                                max_storage[1]) # this is in years
        self.max_storage_hazardous_days = self.max_storage * get_number_of_days_in_timestep(self.model.timestep)  # Default value in days, will be set later based on generator size
        self.max_storage_universal_waste_days = 365  # 1 year for universal waste
        self.max_storage_hazardous_kg = None  # Will be set later based on generator size
        self.number_product_new = 0
        self.number_product_used = 0
        self.number_product_certified = 0
        self.EoL_pathway = self.initial_choice(self.model.init_eol_rate)
        self.used_EoL_pathway = self.EoL_pathway
        self.purchase_choice = self.initial_choice(
            self.model.init_purchase_choice)
        self.generator_size = GeneratorSize.VERY_SMALL  # Default size
        self.hazardous = False  # Default value, will be set later
        self.universal_waste = False  # Default value, will be set later
        self.utility_scale_pv_contribution_factor = 1.0
        self.capacity_contribution_factor = 1.0
        self.installation_year = self.initialize_installation_year()
        self.tclp_test_result = 0

        # reporting variables
        self.waste_kg_current_step = {}

        # ! This increases model resolution nothing to do here for now  
        self.set_pca_state()
        self.set_landfill_transport_distance_and_costs()
        self.set_recycling_transport_distance_and_costs()
        self.set_hazardous_landfill_transport_distance_and_costs()
        self.set_universal_waste_landfill_transport_distance_and_costs()
        self.set_universal_waste_recycling_transport_distance_and_costs()
        self.landfill_name = self.get_landfill_name()
        self.landfill_cost = self.get_initial_landfill_cost(self.landfill_name)
        if self.model.sa_landfill_costs[0]:
            self.landfill_cost = self.model.sa_landfill_costs[1]
        else:
            self.landfill_cost = self.landfill_cost / 1E3 * \
                self.model.dynamic_product_average_wght  # $/W
        # self.init_landfill_cost = self.landfill_cost
        self.set_contribution_factors()

        # ! prepare pvice waste outputs
        self.data_out_pca = pd.read_csv(
            "dataOut_95-by-35.Adv_" + self.pca + "_.csv")
        self.data_out_pca['Yearly_Sum_Power_atEOL'] /= self.agents_per_pca
        self.data_out_pca['Yearly_Sum_Area_atEOL'] /= self.agents_per_pca
        # ! modified the initial number of products & prepare pvice outputs
        self.data_out_pca = transform_timeseries_timestep(
            self.data_out_pca, self.model.timestep)
        self.data_in_pca = pd.read_csv(
            "datain_95-by-35.Adv_" + self.pca + "_.csv")
        self.data_in_pca['new_Installed_Capacity_[MW]'] /= self.agents_per_pca
        self.data_in_pca['new_Installed_Capacity_[MW]'] *= 1E6
        self.data_in_pca = transform_timeseries_timestep(
            self.data_in_pca, self.model.timestep)
        subset_df_cap = self.data_in_pca.copy()
        subset_df_cap = subset_df_cap[
            subset_df_cap['year'] < (self.model.current_date.year)]
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
        # todo: see if this can be stored in the model.
        self.initialize_regulator_id()
            
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

    def initialize_installation_year(self) -> None:
        """
        Initialize installation year based on the consumer_agent_resolution
        """
        if self.model.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            self.installation_year = self.model.current_date.year
        elif self.model.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            self.installation_year = self.model.agent_site_map[self.unique_id][4]
        else:
            raise ValueError("Invalid consumer agent resolution.")
        
    def initialize_regulator_id(self) -> None:
        """
        Initialize regulator id based on the consumer_agent_resolution
        """
        self.regulator_id = None
        for id, state in self.model.regulator_state_map.items():
            if state == self.state:
                self.regulator_id = id
                break
        if self.regulator_id is None:
            # If we get here, no matching regulator was found
            print(f"Warning: No regulator found for state '{self.state}' (agent {self.unique_id})")

    def update_installation_year(self) -> None:
       # When reaching installation end-of-life, agent install new PV panels
       if self.installation_year is None:
           self.initialize_installation_year()
       if self.installation_year + self.model.product_lifetime == self.model.current_date.year:
           self.installation_year = self.model.current_date.year

    def update_transport_costs(self):
        """
        Update transportation costs according to the (evolving) mass of waste.
        # ! remove weight so NOT according to the (evolving) mass of waste.
        """
        #if self.pca == 'p31':
            #print(self.unique_id, self.pca_recyc_transp_dist, 
            #      self.model.transportation_cost, 
            #      self.model.dynamic_product_average_wght)
        # Update transportation costs based on waste type
        if self.universal_waste:
            self.recyc_transp_cost = self.universal_waste_recyc_transp_dist * \
                self.model.get_transportation_cost() / 1E3
            self.landfill_transp_cost = self.universal_waste_landfill_transp_dist * \
                self.model.get_transportation_cost() / 1E3
        else:
            self.recyc_transp_cost = self.recyc_transp_dist * \
                self.model.get_transportation_cost(self.hazardous) / 1E3
                # ! remove weight * \ self.model.dynamic_product_average_wght
            self.landfill_transp_cost = self.landfill_transp_dist * \
                self.model.get_transportation_cost() / 1E3

        self.hazardous_landfill_transp_cost = \
            self.hazardous_landfill_transp_dist * \
            self.model.get_transportation_cost(True) / 1E3
            # ! remove weight * \ self.model.dynamic_product_average_wght
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

    def get_additional_capacity(self) -> float:
        """
        Get additional capacity installed by the agent in the current time
        step.
        :return: Additional capacity installed (in W).
        """
        subset_df_cap = self.data_in_pca.copy()
        subset_df_cap = subset_df_cap[
            subset_df_cap['date'] == self.model.current_date]
        # Multiply by the capacity contribution factor to account for the
        # contribution of the agent based on its resolution (PCA or site-level).
        additional_capacity = subset_df_cap[
            'new_Installed_Capacity_[MW]'].iloc[0] * self.capacity_contribution_factor
        return additional_capacity

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

        additional_capacity = self.get_additional_capacity()
        self.model.pca_install[self.pca] += additional_capacity
        self.model.pca_install_test += additional_capacity
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
                yearly_waste_file['date'] <= self.model.current_date]
            yearly_waste_m2 = sum(
                yearly_waste_m2['Yearly_Sum_Area_atEOL'].tolist())
            self.number_product_EoL_m2 = yearly_waste_m2 * (
                1 - self.used_new_ratio)
            self.number_used_product_EoL_m2 = yearly_waste_m2 * \
                self.used_new_ratio
            self.model.pca_tot_waste_w[self.pca] += yearly_waste
            self.model.pca_tot_waste_m2[self.pca] += yearly_waste_m2
        else:
            yearly_waste = yearly_waste_file[
                yearly_waste_file['date'] == self.model.current_date]
            self.number_product_EoL = yearly_waste[
                'Yearly_Sum_Power_atEOL'].iloc[0] * (1 - self.used_new_ratio)
            self.number_used_product_EoL = yearly_waste[
                'Yearly_Sum_Power_atEOL'].iloc[0] * self.used_new_ratio
            yearly_waste_m2 = yearly_waste_file[
                yearly_waste_file['date'] == self.model.current_date]
            self.number_product_EoL_m2 = yearly_waste_m2[
                'Yearly_Sum_Area_atEOL'].iloc[0] * (1 - self.used_new_ratio)
            self.number_used_product_EoL_m2 = yearly_waste[
                'Yearly_Sum_Area_atEOL'].iloc[0] * self.used_new_ratio
            self.model.pca_tot_waste_w[self.pca] += yearly_waste[
                'Yearly_Sum_Power_atEOL'].iloc[0]
            self.model.pca_tot_waste_m2[self.pca] += yearly_waste_m2[
                'Yearly_Sum_Area_atEOL'].iloc[0]

        self.tot_prod_EoL = (self.number_product_EoL + self.number_used_product_EoL) * \
            self.capacity_contribution_factor * self.utility_scale_pv_contribution_factor
        self.tot_prod_EoL_m2 = (self.number_product_EoL_m2 + self.number_used_product_EoL_m2) * \
            self.capacity_contribution_factor * self.utility_scale_pv_contribution_factor

        subset_df_remaining_cap = self.data_out_pca.copy()
        subset_df_remaining_cap = subset_df_remaining_cap[
            subset_df_remaining_cap['date'] <= self.model.current_date]
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
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos,
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
        for agent in self.model.agents:
            if self.model.num_consumers + self.model.num_prod_n_recyc <= \
                    agent.unique_id < self.model.num_consumers + \
                    self.model.num_prod_n_recyc + self.model.num_refurbishers:
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
                    for agent in self.model.agents:
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
        if self.model.seeding["Seeding"] and self.model.clock// self.model.timestep.value >= \
                self.model.seeding["Year"]:
            for consumer in range(self.model.seeding["number_seed"]):
                if self.unique_id == \
                        self.model.list_consumer_id_seed[consumer]:
                    second_hand_p = 0
                    repair_c = 0
                    for agent in self.model.agents:
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
        if self.model.seeding_recyc["Seeding"] and self.model.clock // self.model.timestep.value >= \
                self.model.seeding_recyc["Year"]:
            for consumer in range(self.model.seeding_recyc["number_seed"]):
                if self.unique_id == \
                        self.model.list_consumer_id_seed[consumer]:
                    self.perceived_behavioral_control[2] *= \
                        self.model.seeding_recyc["discount"]
        if product_type == "new":
            # If hazardous waste regulation is enabled, the agent
            # will have to manage hazardous waste according to the
            # regulatory policies governed by the regulator.
            if self.model.hazardous_waste_regulation_enabled:
                self.hazardous_waste_management()
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
        past_storage = max(0, (self.model.current_date.year - self.max_storage))
        pv_ice_mat_subset_stored_years = self.model.pvice_mat_factor[
            (self.model.pvice_mat_factor['year'] >= past_storage) &
            (self.model.pvice_mat_factor['date'] <= self.model.current_date)]
        avg_weight_factor_stored_pv = pv_ice_mat_subset_stored_years[
            'total_massperm2'].mean()

        original_df = self.data_out_pca.copy()
        original_df = original_df[
            (original_df['year'] >= past_storage) &
            (original_df['date'] <= self.model.current_date)]
        waste_in_w = original_df['Yearly_Sum_Power_atEOL'].mean()
        waste_in_m2 = original_df['Yearly_Sum_Area_atEOL'].mean()
        waste_w_to_m2_factor = waste_in_m2 / waste_in_w
        # if self.unique_id == 0:
        #    print(waste_w_to_m2_factor)

        new_eol_vol = self.number_product_EoL_m2 * self.model.weight_factor \
            + avg_weight_factor_stored_pv * storage * waste_w_to_m2_factor
        used_eol_vol = self.number_used_product_EoL_m2 * \
            self.model.weight_factor

        if eol_pathway == "repair":
            self.number_product_repaired += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[0]
            if product_type == "new":
                self.number_new_prod_repaired += new_eol_vol
            else:
                self.number_used_prod_repaired += used_eol_vol
            self.model.pca_outputs[self.pca][eol_pathway] += (new_eol_vol +
                                                              used_eol_vol)
            self.waste_kg_current_step[eol_pathway] = (new_eol_vol +
                                                         used_eol_vol)
        elif eol_pathway == "sell":
            self.number_product_sold += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[1]
            if product_type == "new":
                self.number_new_prod_sold += new_eol_vol
            else:
                self.number_used_prod_sold += used_eol_vol
            self.model.pca_outputs[self.pca][eol_pathway] += (new_eol_vol +
                                                              used_eol_vol)
            self.waste_kg_current_step[eol_pathway] = (new_eol_vol +
                                                         used_eol_vol)
        elif eol_pathway == "recycle":
            self.number_product_recycled += managed_waste
            if not self.model.epr_business_model:
                self.consumer_costs += managed_waste * \
                                       self.perceived_behavioral_control[2]
            if product_type == "new":
                self.number_new_prod_recycled += new_eol_vol
            else:
                self.number_used_prod_recycled += used_eol_vol
            self.model.pca_outputs[self.pca][eol_pathway] += (new_eol_vol +
                                                              used_eol_vol)
            self.waste_kg_current_step[eol_pathway] = (new_eol_vol +
                                                         used_eol_vol)
        elif eol_pathway == "landfill":
            self.number_product_landfilled += managed_waste
            self.consumer_costs += managed_waste * \
                self.perceived_behavioral_control[3]
            if product_type == "new":
                self.number_new_prod_landfilled += new_eol_vol
            else:
                self.number_used_prod_landfilled += used_eol_vol
            self.model.pca_outputs[self.pca][eol_pathway] += (new_eol_vol +
                                                              used_eol_vol)
            self.waste_kg_current_step[eol_pathway] = (new_eol_vol +
                                                         used_eol_vol)
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
                self.number_used_prod_hoarded += used_eol_vol
            self.model.pca_outputs[self.pca][eol_pathway] += (new_eol_vol +
                                                              used_eol_vol)
            self.waste_kg_current_step[eol_pathway] = (new_eol_vol +
                                                         used_eol_vol)
            if self.hazardous:
                self.number_product_hoarded_hazardous += managed_waste
        if self.unique_id == 0:
            test = 0
            for value in self.model.pca_outputs[self.pca].values():
                test += value

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
        past_storage = max(0, (self.model.current_date.year - self.max_storage))
        pv_ice_mat_subset_stored_years = self.model.pvice_mat_factor[
            (self.model.pvice_mat_factor['year'] >= past_storage) &
            (self.model.pvice_mat_factor['date'] <= self.model.current_date)]
        original_df = self.data_out_pca.copy()
        original_df = original_df[
            (original_df['year'] >= past_storage) &
            (original_df['date'] <= self.model.current_date)]
        waste_in_w = original_df['Yearly_Sum_Power_atEOL'].mean()
        waste_in_m2 = original_df['Yearly_Sum_Area_atEOL'].mean()
        waste_w_to_m2_factor = waste_in_m2 / waste_in_w

        self.weighted_average_mass_watt = pv_ice_mat_subset_stored_years[
            'total_massperm2'].mean() * waste_w_to_m2_factor

        len_product_as_function = len(product_as_function)
        pvice_mat_factor_copy = self.model.pvice_mat_factor[
            self.model.pvice_mat_factor['date'] <= self.model.current_date]
        conversion_factors = \
            pvice_mat_factor_copy['total_massperm2'].to_list()
        conversion_factors = conversion_factors[-len_product_as_function:]
        data_out_pca_copy = self.data_out_pca[
            self.data_out_pca['date'] <= self.model.current_date]
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
        max_storage_period = self.max_storage * self.model.timestep.value
        for i in range(len(self.product_years_storage)):
            if self.product_years_storage[i] == "hoard":
                count += 1
            elif count <= max_storage_period:
                count = 0
        if count > max_storage_period:
            self.product_years_storage = []
            self.product_storage_to_other = self.number_product_hoarded
            self.number_product_hoarded = 0
            self.number_used_prod_hoarded = 0
            self.number_new_prod_hoarded = 0
            limited_paths["hoard"] = False

        self.update_product_storage_hazardous() 
        if self.is_hazardous_waste_storage_limit_exceeded() or \
           self.is_universal_waste_storage_limit_exceeded():
            self.number_product_hoarded_hazardous = 0
            self.product_years_storage_hazardous = []
            limited_paths["hoard"] = False

    def update_product_storage_hazardous(self): 
        """
        Update the storage of hazardous products based on the purchase choice.
        """
        if self.hazardous or self.universal_waste:
            self.product_years_storage_hazardous.append(self.EoL_pathway)
        else:
            self.product_years_storage_hazardous.append("na")
    
    def is_hazardous_waste_storage_limit_exceeded(self):
        """
        Check if the storage limit for hazardous waste is exceeded based on
        the number of years the product has been stored and the maximum
        storage period defined in the model.
        returns True if the limit is exceeded, False otherwise.
        """
        if self.hazardous:
            count = 0
            max_storage_period = self.max_storage_hazardous_days / get_number_of_days_in_timestep(self.model.timestep)
            for eol in self.product_years_storage_hazardous:
                if eol == "hoard":
                    count += 1
                elif count <= max_storage_period:
                    count = 0
            if count > max_storage_period:
                return True
            elif self.max_storage_hazardous_kg is not None:
                # Check if the total mass of stored products per month exceeds
                # the maximum storage capacity in kg.
                total_mass_stored = [0] * len(self.product_years_storage_hazardous)
                total_mass_stored[-1] = self.number_product_hoarded_hazardous
                total_mass_stored = self.mass_per_function_model(total_mass_stored)
                
                total_mass_stored_month = total_mass_stored / self.number_of_months if self.number_of_months > 0 else total_mass_stored
                if total_mass_stored_month > self.max_storage_hazardous_kg:
                    return True
        return False
    
    def is_universal_waste_storage_limit_exceeded(self):
        """
        Check if the storage limit for universal waste is exceeded based on
        the number of days the product has been stored and the maximum
        storage period defined in the model.
        returns True if the limit is exceeded, False otherwise.
        """
        if self.universal_waste:
            count = 0
            max_storage_period = self.max_storage_universal_waste_days / get_number_of_days_in_timestep(self.model.timestep)
            for eol in self.product_years_storage_hazardous:
                if eol == "hoard":
                    count += 1
                elif count <= max_storage_period:
                    count = 0
            if count > max_storage_period:
                return True
        return False
            

    def get_hoarding_cost(self):
        """
        Get the cost of hoarding based on if the waste is hazardous
        or not. If the waste is hazardous, the cost is higher.
        """
        if self.hazardous:
            return self.hoarding_cost + \
                   self.model.hazardous_waste_management_cost['hoard'] / 1E3 * self.model.dynamic_product_average_wght
        else:
            return self.hoarding_cost
        
    def _get_rtn_data(self, df: pd.DataFrame) -> float:

        matching_row = df.loc[
                (df['case_id'] == self.agent_identifier) &
                (df['date'] <= self.model.current_date)
            ]
        
        if not matching_row.empty:
            # take the row with the latest date that is less than or equal to the current date
            latest_date = matching_row['date'].max()
            latest_rows = matching_row.loc[matching_row['date'] == latest_date]
            return latest_rows.iloc[0]  # return the first row if there are multiple with the same latest date
        else:
            earliest_rows = df.loc[
                df['case_id'] == self.agent_identifier].sort_values(by='date')
            if earliest_rows.empty:
                # ! If there are no rows for the case_id, return a row with NaN cost and print a warning message.
                print(f"Warning: No RTN data available for {self.agent_identifier}.")
                return pd.DataFrame({"case_id": [self.agent_identifier], "date": [self.model.current_date], "Cost": [np.nan]}).iloc[0]
            print(
                f"Warning: No RTN data available for {self.agent_identifier} on or before {self.model.current_date}. "
                f"Using earliest available data from {earliest_rows['date'].iloc[0]}."
            )
            return earliest_rows.iloc[0]
        
    def _get_rtn_landfill_cost(self) -> float:
        landfill_cost_rows = self._get_rtn_data(self.model.landfill_cost_df)
        total_landfill_cost = landfill_cost_rows['Cost']
        if pd.isna(total_landfill_cost):
            # ! If the landfill cost data for the agent in the current year is NaN, use the MISSING_VALUE_COST as a fallback and print a warning message.
            print(f"Warning: Landfill cost data for {self.agent_identifier} in {self.model.current_date.year} is NaN in RTN model. Using {MISSING_VALUE_COST} as landfill cost.")
            return MISSING_VALUE_COST
        else:
            return total_landfill_cost
        
    # make this method accessible from constructor
    def get_initial_landfill_cost(self, landfill_name: str):
        """
        Get the initial cost of landfill based on the source of landfill
        cost data. If the source is 'rtn', then get the cost from the rtn
        model, otherwise get the cost from the regular landfill file.
        """
        if self.model.rtn:
            return self._get_rtn_landfill_cost()
        else:
            landfill_name_column = self.model.landfill_data_params['landfill_name_column']
            landfill_volume_column = self.model.landfill_data_params['landfill_volume_column']
            landfill_cost = self.model.landfill_cost_df.loc[
            self.model.landfill_cost_df[landfill_name_column] == landfill_name,
            landfill_volume_column].values[0]
            return landfill_cost
        
    def get_landfill_cost(self):
        """
        get the cost of landfill based on if the waste is hazardous
        or universal waste. Priority: hazardous > universal_waste > regular.
        """
        if self.hazardous:
            return self.hazardous_landfill_cost + \
                   self.model.hazardous_waste_management_cost['landfill'] / 1E3 * self.model.dynamic_product_average_wght
        elif self.universal_waste:
            return self.universal_waste_landfill_cost
        else:
            if self.model.rtn:
                return self._get_rtn_landfill_cost() / 1E3 * self.model.dynamic_product_average_wght
            return self.landfill_cost
        
    def set_pca_state(self):
        """
        Set the PCA and state for the consumer agent based on the
        consumer agent resolution.
        """
        if self.model.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            self.pca = self.model.agent_pca_map[self.unique_id][0]
            self.state = self.model.agent_pca_map[self.unique_id][1]
            self.agents_per_pca = self.model.agent_pca_map[self.unique_id][2]
        elif self.model.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            self.pca = self.model.agent_site_map[self.unique_id][2]
            self.state = self.model.agent_site_map[self.unique_id][3]
            # for site-level resolution, setting agents_per_pca to 1
            # to avoid any further division when calculating waste volume
            # per agent without making changes elsewhere in the code
            self.agents_per_pca = 1
        else:
            raise ValueError("Invalid consumer agent resolution.")
        
    def get_pca_landfill_transp_cost(self):
        """
        Get the transportation cost for landfill based on if the waste is
        hazardous or universal waste. Priority: hazardous > universal_waste > regular.
        """
        if self.hazardous:
            return self.hazardous_landfill_transp_cost
        elif self.universal_waste:
            return self.universal_waste_landfill_transp_cost
        else:
            return self.landfill_transp_cost
        
    @property
    def number_of_months(self):
        """
        Calculate the number of months since the start of the simulation.
        """
        return (self.model.current_date.year - 2020) * 12 + \
               self.model.current_date.month - 1


    def update_perceived_behavioral_control(self):
        """
        Costs from each EoL pathway and purchase choice and related perceived
        behavioral control are updated according to processes from other agents
        or own initiated costs.
        """
        for agent in self.model.agents:
            if agent.unique_id == self.recycling_facility_id:
                # Use universal waste recycling costs if applicable
                if self.universal_waste:
                    recyc_cost = self.get_recycling_cost(agent.recycling_cost) + self.universal_waste_recyc_transp_cost * 0.0077
                else:
                    recyc_cost = self.get_recycling_cost(agent.recycling_cost) + self.recyc_transp_cost * 0.0077
                self.perceived_behavioral_control[2] = recyc_cost  # ! Multiply
                # ! by average mass per watt instead of dynamic
            elif agent.unique_id == self.refurbisher_id:
                self.perceived_behavioral_control[0] = \
                    agent.repairing_cost
                self.perceived_behavioral_control[1] = -1 * \
                    agent.scd_hand_price * (1 - agent.refurbisher_margin)
                self.pbc_reuse[1] = agent.scd_hand_price
        self.pbc_reuse[0] = self.model.fsthand_mkt_pric
        self.perceived_behavioral_control[3] = (
            self.get_landfill_cost() +
            self.get_pca_landfill_transp_cost() * 0.0077) # ! Multiply
                # ! by average mass per watt instead of dynamic
        self.perceived_behavioral_control[4] = self.get_hoarding_cost()

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
        
    def get_generator_size_from_waste(self, waste_kg: float, thresholds: dict) -> GeneratorSize:
        """
        Get the generator size based on the waste amount and thresholds.
        :param waste_kg: The amount of waste in kg.
        :param thresholds: A dictionary mapping generator sizes to their thresholds.
        :return: The generator size that can handle the given waste amount.
        :raises ValueError: If no suitable generator size is found.
        """

        new_generator_size = None
        new_generator_max_waste = None

        unlimited_generator_size = None

        for generator_size, threshold in thresholds.items():
            if threshold.waste_generation_limit_kg is not None:
                if new_generator_size is None: # First valid generator size
                    new_generator_size = generator_size
                    new_generator_max_waste = threshold.waste_generation_limit_kg
                elif new_generator_max_waste < waste_kg <= threshold.waste_generation_limit_kg: # waste fits in this generator size
                    new_generator_size = generator_size
                    new_generator_max_waste = threshold.waste_generation_limit_kg
            else:
                unlimited_generator_size = generator_size # This generator size has no storage limit
        
        # If no generator size was found that can handle the waste amount,
        # check if there is an unlimited generator size available.
        # If so, use that size.
        # If not, raise an error.
        if waste_kg > thresholds[new_generator_size].waste_generation_limit_kg:
            if unlimited_generator_size is not None:
                new_generator_size = unlimited_generator_size
                new_generator_max_waste = None
            else:
                raise ValueError("No suitable generator size found for the given waste amount.")                    
        
        return new_generator_size
            
        
    def update_generator_size(self, regulator_thresholds: dict):
        """
        Update the generator size based on the total waste generated
        in the current period and the threshold limits set by the regulator.
        It uses the total waste generated per month to determine the generator size.
        :param regulator_thresholds: A dictionary mapping generator sizes to their thresholds.
        """
        if self.hazardous:
            hazardous_waste_mass = [0] * len(self.new_products_hard_copy)
            hazardous_waste_mass[-1] = self.tot_prod_EoL
            hazardous_waste_mass_month = self.mass_per_function_model(hazardous_waste_mass) / self.number_of_months if self.number_of_months > 0 else self.mass_per_function_model(hazardous_waste_mass)
            self.generator_size = self.get_generator_size_from_waste(
            hazardous_waste_mass_month, regulator_thresholds)

    def update_universal_waste_generator_size(self, universal_waste_thresholds: dict):
        """
        Update the generator size based on the total waste generated
        in the current period and the threshold limits set by the regulator
        for universal waste.
        It uses the total waste generated in the past year to determine the generator size.
        :param universal_waste_thresholds: A dictionary mapping generator sizes to their thresholds.
        """
        if self.universal_waste:
            # if not universal_waste_thresholds:
            #     # If no thresholds are provided, default to SMALL generator size
            #     self.generator_size = GeneratorSize.SMALL
            past_year_index = len(self.new_products_hard_copy) - self.model.timestep.value
            universal_waste_mass_past_year = self.new_products_hard_copy[past_year_index:]
            universal_waste_mass = self.mass_per_function_model(universal_waste_mass_past_year).sum()
            self.generator_size = self.get_generator_size_from_waste(
                universal_waste_mass, universal_waste_thresholds)

    def update_universal_waste_limits(self, universal_waste_thresholds: dict):
        """
        Update the storage limits based on the generator size
        and the thresholds set by the regulator for universal waste.
        :param universal_waste_thresholds: A dictionary mapping generator sizes to their thresholds.
        """
        if self.universal_waste:
            if universal_waste_thresholds[self.generator_size].max_storage_days is not None:
                self.max_storage_hazardous_days = universal_waste_thresholds[self.generator_size].max_storage_days
            
    def hazardous_waste_management(self):
        """
        determine if the waste generated is hazardous
        depending on the regulatory policy in place.
        If the waste is hazardous, update the generator size
        based on the thresholds set by the regulator.
        Update the storage limits based on the generator size.
        """
        if self.regulator_id is None:
            self.initialize_regulator_id()
        agent = self.model.agent_map[self.regulator_id]
        if agent.is_exclusion_applicable(self.recycling_facility_id):
            # If the product is exempt from regulations, it is not hazardous
            # It is treated the same as non-hazardous waste
            self.hazardous = False
            self.universal_waste = False
        else:
            # If the TCLP test is applicable, check if the waste is hazardous
            # based on the TCLP test results.
            is_tclp_positive = self.model.tclp_test()
            self.tclp_test_result = int(is_tclp_positive)
            if is_tclp_positive:
                if agent.is_universal_waste_regulation_applicable():
                    # If the product is subject to universal waste regulations, it is not hazardous
                    # but treated as universal waste
                    # the storage limit is set to 1 year
                    self.hazardous = False
                    self.universal_waste = True
                    self.update_universal_waste_generator_size(
                        agent.universal_waste_thresholds)
                    self.update_universal_waste_limits(
                        agent.universal_waste_thresholds)
                    # Set the recycling facility to the closest universal waste recycler
                    self.recycling_facility_id = self.get_closest_recycler_id()
                else:
                    # If the product is hazardous, set the hazardous flag to True
                    self.hazardous = True
                    self.universal_waste = False
            # If the waste is hazardous, update the generator size based on the thresholds
            self.update_generator_size(agent.thresholds)
            self.update_hazardous_storage_limits(agent.thresholds)
            self.update_perceived_behavioral_control()

    def update_hazardous_storage_limits(self, regulator_thresholds: dict):
        """
        Update the storage limits based on the generator size
        and the thresholds set by the regulator.
        :param regulator_thresholds: A dictionary mapping generator sizes to their thresholds.
        """
        if self.hazardous:
            if regulator_thresholds[self.generator_size].max_storage_days is not None:
                self.max_storage_hazardous_days = regulator_thresholds[self.generator_size].max_storage_days
            if regulator_thresholds[self.generator_size].max_storage_kg is not None:
                self.max_storage_hazardous_kg = regulator_thresholds[self.generator_size].max_storage_kg

    def _find_closest_recycler_name(self, distance_df):
        """
        Helper function to find the name of the closest recycler.
        
        Args:
            distance_df: DataFrame containing recycler distances
            
        Returns:
            Name of the closest recycler
        """
        recyc_distances = distance_df[str(self.agent_identifier)]
        closest_recycler_idx = recyc_distances.idxmin()
        closest_recycler_name = distance_df.loc[closest_recycler_idx, 'Recycler Name']
        return closest_recycler_name

    def get_closest_recycler_id(self):
        """
        Find the recycling facility with the shortest distance
        Returns the agent ID of the closest recycler.
        """
        if self.universal_waste:
            # Use universal waste recyclers
            closest_recycler_name = self._find_closest_recycler_name(
                self.model.universal_waste_recycler_distance_df.copy())
        else:
            # Use regular recyclers
            closest_recycler_name = self._find_closest_recycler_name(
                self.model.recycler_distance_df.copy())
        
        # Use Mesa AgentSet to select the recycler agent with matching name
        recycler_agents = self.model.agents.select(
            filter_func= lambda agent: isinstance(agent, Recyclers) and agent.recycler_name == closest_recycler_name
        )
        
        if len(recycler_agents) > 0:
            return recycler_agents[0].unique_id
        else:
            raise ValueError(f"No recycler agent found with name: {closest_recycler_name}")

    def set_recycling_transport_distance_and_costs(self):
        """
        Initialize recycling transportation costs and distances.
        """

        recyc_transp_dist = self.model.recycler_distance_df.copy()
        recyc_transp_dist = recyc_transp_dist[str(self.agent_identifier)]
        recyc_transp_dist = recyc_transp_dist.to_list()
        self.recyc_transp_dist = min(recyc_transp_dist)
        self.recyc_transp_cost = self.recyc_transp_dist * \
            self.model.get_transportation_cost(self.hazardous) / 1E3
            # ! remove weight * \ self.model.dynamic_product_average_wght

    def set_universal_waste_recycling_transport_distance_and_costs(self):
        """
        Initialize universal waste recycling transportation costs and distances.
        """
        universal_waste_recyc_transp_dist = self.model.universal_waste_recycler_distance_df.copy()
        universal_waste_recyc_transp_dist = universal_waste_recyc_transp_dist[str(self.agent_identifier)]
        universal_waste_recyc_transp_dist = universal_waste_recyc_transp_dist.to_list()
        self.universal_waste_recyc_transp_dist = min(universal_waste_recyc_transp_dist)
        self.universal_waste_recyc_transp_cost = self.universal_waste_recyc_transp_dist * \
            self.model.get_transportation_cost() / 1E3

    def set_landfill_transport_distance_and_costs(self):
        """
        Initialize landfill transportation costs and distances.
        """

        # ! TODO: change landfill costs
        landfill_transp_dist = self.model.landfill_distance_df.copy()
        landfill_transp_dist = landfill_transp_dist[str(self.agent_identifier)]
        landfill_transp_dist = landfill_transp_dist.to_list()
        self.landfill_transp_dist = min(landfill_transp_dist)
        self.landfill_transp_cost = self.landfill_transp_dist * \
            self.model.get_transportation_cost() / 1E3 
            # ! remove weight * \ self.model.dynamic_product_average_wght

    def set_hazardous_landfill_transport_distance_and_costs(self):
        """
        Initialize hazardous landfill transportation costs and distances.
        """
        hazardous_landfill_transp_dist = \
            self.model.hazardous_landfill_distance_df.copy()
        hazardous_landfill_transp_dist = \
            hazardous_landfill_transp_dist[str(self.agent_identifier)].to_list()
        self.hazardous_landfill_transp_dist = \
            min(hazardous_landfill_transp_dist)
        self.hazardous_landfill_transp_cost = \
            self.hazardous_landfill_transp_dist * \
            self.model.get_transportation_cost(True) / 1E3
        self.hazardous_landfill_name = \
            self.model.hazardous_landfill_distance_df.loc[
                self.model.hazardous_landfill_distance_df[str(self.agent_identifier)] ==
                self.hazardous_landfill_transp_dist, 'Facility Name'].iloc[0]
        hazardous_landfills_data = self.model.hazardous_landfill_cost_df.copy()
        self.hazardous_landfill_cost = hazardous_landfills_data.loc[
            hazardous_landfills_data['Facility Name'] == self.hazardous_landfill_name,
            '$/ Ton'].iloc[0]  # in $/ton
        
    def set_universal_waste_landfill_transport_distance_and_costs(self):
        """
        Initialize universal waste landfill transportation costs and distances.
        """
        universal_waste_landfill_transp_dist = \
            self.model.universal_waste_landfill_distance_df.copy()
        universal_waste_landfill_transp_dist = \
            universal_waste_landfill_transp_dist[str(self.agent_identifier)].to_list()
        self.universal_waste_landfill_transp_dist = \
            min(universal_waste_landfill_transp_dist)
        self.universal_waste_landfill_transp_cost = \
            self.universal_waste_landfill_transp_dist * \
            self.model.get_transportation_cost() / 1E3
        self.universal_waste_landfill_name = \
            self.model.universal_waste_landfill_distance_df.loc[
                self.model.universal_waste_landfill_distance_df[str(self.agent_identifier)] ==
                self.universal_waste_landfill_transp_dist, 'Facility Name'].iloc[0]
        universal_waste_landfills_data = self.model.universal_waste_landfill_cost_df.copy()
        self.universal_waste_landfill_cost = universal_waste_landfills_data.loc[
            universal_waste_landfills_data['Facility Name'] == self.universal_waste_landfill_name,
            '$/ Ton'].iloc[0]  # in $/ton
        
    def set_contribution_factors(self):
        """
        Set contribution factors based on consumer agent resolution. These
        factors are used to adjust the impact of utility-scale PV and capacity
        based on the agent's resolution level.
        """
        if self.model.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            self.utility_scale_pv_contribution_factor = 1
            self.capacity_contribution_factor = 1
        elif self.model.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            row = self.model.reeds_data.loc[
                (self.model.reeds_data['r'] == self.pca) & (self.model.reeds_data['t'] <= self.model.current_date.year)
            ]
            if row.empty:
                # print(f"Warning: No REEDS data found for PCA {self.pca} in year {self.model.current_date.year}. Setting contribution factors to 1")
                self.utility_scale_pv_contribution_factor = 1
            else:
                self.utility_scale_pv_contribution_factor = row['utility_scale_pv_contribution_factor'].values[-1]
            agents_in_pca = self.model.uspvdb.loc[
                self.model.uspvdb['PCA'] == self.pca]
            total_capacity_in_pca = agents_in_pca['p_cap_ac'].sum()
            agent_capacity = self.model.uspvdb.loc[
                self.model.uspvdb['case_id'] == self.agent_identifier, 'p_cap_ac'].values[0]
            self.capacity_contribution_factor = agent_capacity / total_capacity_in_pca
            
        
    def get_landfill_name(self) -> str:
        """
        Get the name of the landfill based on the transportation distance from the landfill dataframe.
        :return: The name of the landfill.
        """

        if self.model.rtn:
            landfill_cost_row = self._get_rtn_data(self.model.landfill_cost_df)
            landfill_name = landfill_cost_row['Landfill Name']
            return landfill_name
        else:
            return self.model.landfill_distance_df.loc[
                self.model.landfill_distance_df[str(self.agent_identifier)] ==
                self.landfill_transp_dist, self.model.landfill_data_params['landfill_name_column']].iloc[0]
        
    def _get_rtn_recycling_cost(self) -> float:
        recycling_cost_row = self._get_rtn_data(self.model.recycling_costs_df)
        # sum all rows Cost values if multiple rows are returned for the same case_id and date
        total_recycling_cost = recycling_cost_row['Cost']
        if pd.isna(total_recycling_cost):
            # If the recycling cost is NaN, print a warning and return a default value for the cost.
            print(f"Warning: Recycling cost for {self.agent_identifier} in {self.model.current_date.year} is NaN. Using {MISSING_VALUE_COST} as cost.")
            return MISSING_VALUE_COST
        else:
            return total_recycling_cost
        
    def get_recycling_cost(self, agent_recycling_cost: float) -> float:
        """
        Get the recycling cost based on the source of recycling cost data.
        If the source is 'rtn', then get the cost from the rtn model, otherwise
        get the cost from the associated recycling facility agent.
        """
        if self.model.rtn:
            return self._get_rtn_recycling_cost()
        else:
            return agent_recycling_cost
        
    @property
    def agent_identifier(self) -> str:
        """
        Get the agent identifier based on the consumer agent resolution. If
        the resolution is PCA, return the PCA. If the resolution is SITE, return
        the site case ID.
        :return: The agent identifier.
        """
        
        if self.model.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            return self.pca
        elif self.model.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            return self.model.agent_site_map[self.unique_id][0]
        else:
            raise ValueError("Invalid consumer agent resolution.")
        
    def _get_agent_lat_lon(self) -> tuple:
        """
        Get the latitude and longitude of the agent based on its identifier.
        :return: A tuple containing the latitude and longitude of the agent.
        """
        if self.model.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            lat = self.model.pca_data.loc[
                self.model.pca_data['PCA'] == self.agent_identifier, 'Lat'].values[0]
            lon = self.model.pca_data.loc[
                self.model.pca_data['PCA'] == self.agent_identifier, 'Long'].values[0]
        elif self.model.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            lat = self.model.uspvdb.loc[
                self.model.uspvdb['case_id'] == self.agent_identifier, 'Latitude'].values[0]
            lon = self.model.uspvdb.loc[
                self.model.uspvdb['case_id'] == self.agent_identifier, 'Longitude'].values[0]
        else:
            raise ValueError("Invalid consumer agent resolution.")
        return lat, lon

    def step(self):
        """
        Evolution of agent at each step
        """
        self.product_mass_output_metrics()
        self.product_storage_to_other = 0
        self.product_storage_to_other_ref = 0
        # reset waste generated in the current step for each EoL pathway
        self.waste_kg_current_step = {}
        self.update_transport_costs()
        # Update product growth from a list:
        if self.model.clock // self.model.timestep.value > self.model.growth_threshold:
            self.product_growth = self.product_growth_list[1]
        self.update_product_stock()
        self.yearly_prod_n_waste()
        self.update_perceived_behavioral_control()
        self.copy_perceived_behavioral_control = \
            self.perceived_behavioral_control.copy()
        self.volume_used_products_purchased()
        self.update_product_eol("new")
        self.product_storage_to_other_ref = self.product_storage_to_other
        # self.update_product_eol("used")
        self.update_installation_year()


def report_output_consumer(agent: Consumers, field: str) -> any:
    """
    Report specific output field for the agent.
    :param field: The field to report.
    :return: The value of the specified field.
    """
    if field == "name":
        if agent.model.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            return f"{agent.pca}_{agent.unique_id}"
        elif agent.model.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            return agent.model.agent_site_map[agent.unique_id][1]
    elif field in ["latitude", "longitude"]:
        lat, lon = agent._get_agent_lat_lon()
        return lat if field == "latitude" else lon
    elif field == "repair_kg":
        return agent.waste_kg_current_step.get("repair", 0)
    elif field == "sell_kg":
        return agent.waste_kg_current_step.get("sell", 0)
    elif field == "recycle_kg":
        return agent.waste_kg_current_step.get("recycle", 0)
    elif field == "landfill_kg":
        return agent.waste_kg_current_step.get("landfill", 0)
    elif field == "hoard_kg":
        return agent.waste_kg_current_step.get("hoard", 0)
    elif field == "total_waste_W":
        return agent.tot_prod_EoL
    elif field == "total_waste_m2":
        return agent.tot_prod_EoL_m2
    elif field == "total_installed_capacity_W":
        return agent.get_additional_capacity()
    elif field == "tclp_test_result":
        return agent.tclp_test_result
    else:
        raise ValueError(f"Field '{field}' not recognized for reporting.")