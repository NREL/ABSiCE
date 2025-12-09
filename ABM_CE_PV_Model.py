# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 09:33 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Model - Circular Economy Agent-based Model (CE ABM)
This module contains the model class that creates and activates agents. The
module also defines inputs (default values can be changed by user) and collect
outputs.
"""

# ! Goals of the NSF development project:
# 1) Revamp the PV ABM code (format, efficiency)
# 2) Improve the accuracy of waste predictions
# 3) Improve the model's resolution (at least state)
# 4) Improve cost modeling, including transportation & logistics
# 5) Improve end markets' resolution
# 6) Start thinking and implementing environmental justice capabilities(e.g.,
#    estimation of job created by recycling activities etc.)
# ! TODO list:
# 1) Revamp the PV ABM code (format, efficiency):
#   i) Clean code: solve basic typo errors etc. - high priority
#     a) Find out how to keep Flake8 from showing an error when importing all
#        with "*"
#   ii) Replace basic functions with functions from CEWAM and the TPB_ABM -
#       high priority:
#     a) Check regularly that the model still works and provides same results
#        as before
#     b) Change the Python environment to use the latest version of Mesa (clone
#        the environment from the TPB_ABM)
#   iii) Refactor variable and file names to make them general (as a general
#        CE ABM framework rather than a CE PV ABM) - low priority, if times
#        allows
# 2) Improve the accuracy of waste predictions:
#   i) Use PV ICE as a pip install library or the data inputs from PV ICE -
#      high priority:
#     a) Justification: that would speed up extending the number of materials
#        and accuracy of PV panels vintages in the ABM
#     b) If needed, discuss with Silvana and the PV ICE team
#   ii) Use PV ICE baseline scenario and validate that waste generation
#       is the same than in PV ICE publications - high priority
# 3) Improve the model's resolution (at least state):
#   i) Use PV ICE to improve the resolution of the capacity and waste
#      projections by states - high priority
#   ii) Modify the agent as an entity - moderate priority:
#     a) Number - find a compromise between a high number of agents and low
#        computational requirements (keep below 40-50 seconds per state)
#     b) Creation and destruction of agents: use the functions from CEWAM
#     c) Refine the agent types: utility, commercial or residential PV
# 4) Improve cost modeling, including transportation & logistics:
#   i) Add the TCLP costs and other costs associated with assessing the module
#      viability/performance (pre-transportation costs) - moderate priority:
#   ii) Improve transportation modeling - high priority::
#     a) Any better source than ATRI for the cost/mile?
#     b) Use mock-up facility locations to develop the transportation model
#        based on the OpenRoute service API (from the API, use one time live
#        calls (one per origin-destination) or call o the origin-destination
#        matrix - get the distance and other information if easy and
#        potentially relevant)
#     c) Replace the mock-ups by real values from Texas A&M once they have
#        their model
#   iii) Any other additions? For instance, could the regulator agents from
#        CEWAM be added, are they relevant? What would be their behavioral
#        rules? What about landfill agents? - low priority, if times allows
# 5) Improve end markets' resolution:
#   i) Find more accurate price data for different recovered materials -
#      moderate priority
#   ii) Expand the number of materials and their end markets; for instance have
#       the aluminum recycler and the automotive market as a low grade silicon
#       application (i.e., Silumin) or another type of recycler and the
#       electronic or PV markets as a high grade silicon application (of course
#       it would depend on the quality of the silicon obtained with a
#       particular recycling process) - high priority
#   iii) Add market constraints (e.g., limited demand for a certain material
#        which would require finding other markets if supply from PV is too
#        high) - moderate priority
#   iv) Anything else we can think of or gather from the working groups and
#       stakeholders? - low priority, if times allows
# 6) Start thinking and implementing environmental justice capabilities(e.g.,
#    estimation of job created by recycling activities etc.) - low priority,
#    if times allows

from mesa import Model
from ABM_CE_PV_ConsumerAgents import Consumers, report_output_consumer
from ABM_CE_PV_RecyclerAgents import Recyclers
from ABM_CE_PV_RefurbisherAgents import Refurbishers
from ABM_CE_PV_ProducerAgents import Producers
from ABM_CE_PV_RegulatorAgents import Regulators
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
from math import e, gamma
import pandas as pd
import random
import PV_ICE
import os
import csv
import pandas as pd
from geopy.geocoders import Nominatim
import time
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from utils import TIMESTEP, ConsumerAgentResolution, PCA_MISSING_VALUE, transform_timeseries_timestep, transform_pca_timeseries_timestep
from datetime import datetime



class ABM_CE_PV(Model):
    def __init__(self,
                 seed=None,
                 calibration_n_sensitivity=1,
                 calibration_n_sensitivity_2=1,
                 calibration_n_sensitivity_3=1,
                 calibration_n_sensitivity_4=1,
                 calibration_n_sensitivity_5=1,
                 timestep=TIMESTEP.ANNUAL,
                 consumer_agent_resolution=ConsumerAgentResolution.SITE,
                 model_states=None,
                 num_consumers=1000,
                 consumers_node_degree=10,
                 consumers_network_type="small-world",
                 rewiring_prob=0.1,
                 num_recyclers=16,
                 num_producers=60,
                 prod_n_recyc_node_degree=5,
                 prod_n_recyc_network_type="small-world",
                 num_refurbishers=15,
                 consumers_distribution={"residential": 1,
                                         "commercial": 0., "utility": 0.},
                 init_eol_rate={"repair": 0.005, "sell": 0.01,
                                "recycle": 0.1, "landfill": 0.4425,
                                "hoard": 0.4425},
                 init_purchase_choice={"new": 0.9995, "used": 0.0005,
                                       "certified": 0},
                 total_number_product=[38, 38, 38, 38, 38, 38, 38, 139, 251,
                                       378, 739, 1670, 2935, 4146, 5432, 6525,
                                       3609, 4207, 4905, 5719],
                 product_distribution={"residential": 1,
                                       "commercial": 0., "utility": 0.},
                 product_growth=[0.166, 0.045],
                 growth_threshold=10,
                 failure_rate_alpha=[2.4928, 5.3759, 3.93495],
                 hoarding_cost=[0, 0.001, 0.0005],
                 landfill_cost=[
                     0.0089, 0.0074, 0.0071, 0.0069, 0.0056, 0.0043,
                     0.0067, 0.0110, 0.0085, 0.0082, 0.0079, 0.0074, 0.0069,
                     0.0068, 0.0068, 0.0052, 0.0052, 0.0051, 0.0074, 0.0062,
                     0.0049, 0.0049, 0.0047, 0.0032, 0.0049, 0.0065, 0.0064,
                     0.0062, 0.0052, 0.0048, 0.0048, 0.0044, 0.0042, 0.0039,
                     0.0039, 0.0045, 0.0055, 0.0050, 0.0049, 0.0044, 0.0044,
                     0.0039, 0.0033, 0.0030, 0.0041, 0.0050, 0.0040, 0.0040,
                     0.0038, 0.0033],
                 hazardous_waste_management_cost={"repair": 0.0, "sell": 0.0,
                                                    "recycle": 0.0, "landfill": 0.0,
                                                    "hoard": 0.0}, # $/ton
                 theory_of_planned_behavior={
                     "residential": True, "commercial": True, "utility": True},
                 w_sn_eol=0.23,
                 w_pbc_eol=0.44,
                 w_a_eol=0.59,
                 w_sn_reuse=0.497,
                 w_pbc_reuse=0.382,
                 w_a_reuse=0.464,
                 product_lifetime=30,
                 all_EoL_pathways={"repair": True, "sell": True,
                                   "recycle": True, "landfill": True,
                                   "hoard": True},
                 max_storage=[1, 8, 4],
                 att_distrib_param_eol= [0.595, 0.1], # [0.805, 0.09],
                 att_distrib_param_reuse=[0.01, 0.185], # [0.223, 0.262],
                 original_recycling_cost=[0.128-1E-6, 0.128+1E-6, 0.128],
                 recycling_learning_shape_factor=-0.01, # -0.3,
                 repairability=0.55,
                 original_repairing_cost=[0.1, 0.45, 0.23],
                 repairing_learning_shape_factor=-0.31,
                 scndhand_mkt_pric_rate=[0.4, 0.2],
                 fsthand_mkt_pric=0.45,
                 fsthand_mkt_pric_reg_param=[1, 0.04],
                 refurbisher_margin=[0.4, 0.6, 0.5],
                 purchase_choices={"new": True, "used": True,
                                   "certified": False},
                 product_mass_fractions={"Product": 1, "Aluminum": 0.08,
                                         "Glass": 0.76, "Copper": 0.01,
                                         "Insulated cable": 0.012,
                                         "Silicon": 0.036, "Silver": 0.00032},
                 established_scd_mkt={"Product": True, "Aluminum": True,
                                      "Glass": True, "Copper": True,
                                      "Insulated cable": True,
                                      "Silicon": False, "Silver": False},
                 scd_mat_prices={"Product": [np.nan, np.nan, np.nan],
                                 "Aluminum": [0.66, 1.98, 1.32],
                                 "Glass": [0.01, 0.06, 0.035],
                                 "Copper": [3.77, 6.75, 5.75],
                                 "Insulated cable": [3.22, 3.44, 3.33],
                                 "Silicon": [2.20, 3.18, 2.69],
                                 "Silver": [453, 653, 582]},
                 virgin_mat_prices={"Product": [np.nan, np.nan, np.nan],
                                    "Aluminum": [1.76, 2.51, 2.14],
                                    "Glass": [0.04, 0.07, 0.055],
                                    "Copper": [4.19, 7.50, 6.39],
                                    "Insulated cable": [3.22, 3.44, 3.33],
                                    "Silicon": [2.20, 3.18, 2.69],
                                    "Silver": [453, 653, 582]},
                 material_waste_ratio={"Product": 0., "Aluminum": 0.,
                                       "Glass": 0., "Copper": 0.,
                                       "Insulated cable": 0., "Silicon": 0.4,
                                       "Silver": 0.},
                 recovery_fractions={"Product": np.nan, "Aluminum": 0.92,
                                     "Glass": 0.85, "Copper": 0.72,
                                     "Insulated cable": 1, "Silicon": 0,
                                     "Silver": 0},
                 product_average_wght=0.1,
                 mass_to_function_reg_coeff=0.03,
                 recycling_states=[
                     'Texas', 'Arizona', 'Oregon', 'Oklahoma',
                     'Wisconsin', 'Ohio', 'Kentucky', 'South Carolina'],
                 # transportation_cost=0.0314,
                 transportation_cost=0.095,
                 hazardous_transportation_cost=0.395, # $/ton-km
                 used_product_substitution_rate=[0.6, 1, 0.8],
                 imperfect_substitution=0,
                 epr_business_model=False,
                 recycling_process={"frelp": False, "asu": False,
                                    "hybrid": False},
                 dynamic_lifetime_model={"Dynamic lifetime": False,
                                         "d_lifetime_intercept": 15.9,
                                         "d_lifetime_reg_coeff": 0.87,
                                         "Seed": False, "Year": 5,
                                         "avg_lifetime": 50},
                 extended_tpb={"Extended tpb": False,
                               "w_convenience": 0.28, "w_knowledge": -0.51,
                               "knowledge_distrib": [0.5, 0.49]},
                 seeding={"Seeding": False,
                          "Year": 10, "number_seed": 50},
                 seeding_recyc={"Seeding": False,
                                "Year": 10, "number_seed": 50,
                                "discount": 0.35},
                # parameters for TCLP model. The default values are taken from
                # Li, F., Tatapudi, S. R., Shaw, S. L., Libby, C., Bicer, B., & TamizhMani, G. (2025). 
                # Photovoltaic module leach testing: Database development and statistical analysis. 
                # Journal of Environmental Management, 377, 124666.
                 tclp_params = {
                        "fresh_mean": 2.18,   # observed mean (mg/L Pb) for fresh modules
                        "fresh_std": 0.86,    # observed std (mg/L Pb) for fresh modules
                        "aged_mean": 3.20,    # observed mean (mg/L Pb) for field aged modules
                        "aged_std": 1.33,     # observed std (mg/L Pb) for field aged modules
                        "k": 0.30,            # rate of degradation
                        "a50": 15,  # midpoint age
                        "hazard_cutoff": 5.0, # mg/L Pb threshold for hazard classification as per EPA
                        # Optional lower bound for std to avoid collapse
                        "min_std": 0.05,
                        "distribution": "weibull"  # distribution type: "normal" or "weibull"
                 },
                 pv_ice=False,
                 pca=False,
                 pca_scenario=False,
                 geopy=False,
                 calculate_distances=False,
                 rtn = False,
                 solar_cycle =False,
                 landfill_data_params = {
                        "landfill_volume_column": "$/metric ton",
                        "landfill_name_column": "Facility Name"},
                 hazardous_waste_regulation_enabled=False,
                 landfill_solar_waste_acceptance_ratio=0.4,
                 last_step=31,
                 sa_landfill_costs=(False, 0.0037),
                 file_name={'Landfill data': "Landfills_data_2023.csv",
                            'PCA-landfill distances':
                                "pca_landfills_distances.csv",
                            'Hazardous landfill data': "Landfills_data_SA.csv",
                            'Hazardous PCA-landfill distances':
                                "pca_landfills_distances_SA.csv",
                            'Site-landfill distances':
                                "site_landfills_distances.csv",}):

        """Initiate model.

        Args:
            seed (int, optional): number used to initialize the random
                generator. Defaults to None.
            calibration_n_sensitivity (int, optional): enable varying
                different type of input parameters . Defaults to 1.
            calibration_n_sensitivity_2 (int, optional): enable varying
                different type of input parameters . Defaults to 1.
            calibration_n_sensitivity_3 (int, optional): enable varying
                different type of input parameters . Defaults to 1.
            calibration_n_sensitivity_4 (int, optional): enable varying
                different type of input parameters . Defaults to 1.
            calibration_n_sensitivity_5 (int, optional): enable varying
                different type of input parameters . Defaults to 1.
            timestep (TIMESTEP, optional): time step of the model. Defaults to
                TIMESTEP.ANNUAL. Scaling is applied linearly to the model data
                to match the time step.
            consumer_agent_resolution (ConsumerAgentResolution, optional): Determines the resolution of consumer agents in the model.
                Defaults to ConsumerAgentResolution.PCA.
            model_states (list, optional): list of US states to model. Defaults to None (which means all states).
            num_consumers (int, optional): number of consumers.
                Defaults to 1000.
            consumers_node_degree (int, optional): average node degree in the
                network. Defaults to 10.
            consumers_network_type (str, optional): network type.
                Defaults to "small-world".
            rewiring_prob (float, optional): probability of rewiring an edge in
                the network. Defaults to 0.1.
            num_recyclers (int, optional): number of recyclers. Defaults to 16.
            num_producers (int, optional): number of producers. Defaults to 60.
            prod_n_recyc_node_degree (int, optional): average node degree in
                the network. Defaults to 5.
            prod_n_recyc_network_type (str, optional): network type.
                Defaults to "small-world".
            num_refurbishers (int, optional): number of refurbishers.
                Defaults to 15.
            consumers_distribution (dict, optional): type of consumers.
                Defaults to {"residential": 1, "commercial": 0.,
                "utility": 0.}.
            init_eol_rate (dict, optional): initial distribution of end-of-life
                (eol) pathways adoption in the consumer population. Defaults to
                {"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill":
                0.4425, "hoard": 0.4425}.
            init_purchase_choice (dict, optional): initial distribution of
                purchase option adoption in the consumer population. Defaults
                to {"new": 0.9995, "used": 0.0005, "certified": 0}.
            total_number_product (list, optional): Total number of products
                expressed in relevant units (e.g., kg or kW) for the first
                initial years. Defaults to [38, 38, 38, 38, 38, 38, 38, 139,
                251, 378, 739, 1670, 2935, 4146, 5432, 6525, 3609, 4207, 4905,
                5719].
            product_distribution (dict, optional): product shares among
            consumer types. Defaults to {"residential": 1, "commercial": 0.,
                "utility": 0.}.
            product_growth (list, optional): compounded annual growth rate
                (CAGR). Defaults to [0.166, 0.045].
            growth_threshold (int, optional): threshold separating period of
                different CAGR. Defaults to 10.
            failure_rate_alpha (list, optional): alpha parameter of the Weibull
                function that models waste generation. Defaults to [2.4928,
                5.3759, 3.93495].
            hoarding_cost (list, optional): storage costs. Defaults to
                [0, 0.001, 0.0005].
            landfill_cost (list, optional): landfill costs. Defaults to
                [ 0.0089, 0.0074, 0.0071, 0.0069, 0.0056, 0.0043, 0.0067,
                0.0110, 0.0085, 0.0082, 0.0079, 0.0074, 0.0069, 0.0068, 0.0068,
                0.0052,
                0.0052, 0.0051, 0.0074, 0.0062, 0.0049, 0.0049, 0.0047, 0.0032,
                0.0049, 0.0065, 0.0064, 0.0062, 0.0052, 0.0048, 0.0048, 0.0044,
                0.0042, 0.0039, 0.0039, 0.0045, 0.0055, 0.0050, 0.0049, 0.0044,
                0.0044, 0.0039, 0.0033, 0.0030, 0.0041, 0.0050, 0.0040, 0.0040,
                0.0038, 0.0033].
            theory_of_planned_behavior (dict, optional): define how the
                different population types make choices. Defaults to
                { "residential": True, "commercial": True, "utility": True}.
            w_sn_eol (float, optional): weight of the subjective norms variable
                in the decision-making model (i.e., the theory of planned
                behavior (TPB)) for the eol options. Defaults to 0.27.
            w_pbc_eol (float, optional): weight of the perceived behavioral
                control variable in the TPB for the eol options. Defaults to
                0.44.
            w_a_eol (float, optional): weight of the attitude  variable in the
                TPB for the eol options. Defaults to 0.39.
            w_sn_reuse (float, optional): weight of the subjective norms
                variable in the TPB for the purchase options. Defaults to
                0.497.
            w_pbc_reuse (float, optional): weight of the perceived behavioral
                control variable in the TPB for the purchase options. Defaults
                to 0.382.
            w_a_reuse (float, optional): weight of the attitude variable in the
                TPB for the purchase options. Defaults to 0.464.
            product_lifetime (int, optional): product lifetime, beta parameter
                of the Weibull function that models waste generation. Defaults
                to 30.
            all_EoL_pathways (dict, optional): eol options that are available.
                Defaults to {"repair": True, "sell": True, "recycle": True,
                "landfill": True, "hoard": True}.
            max_storage (list, optional): parameters defining the potential
                storage time for the product. Defaults to [1, 8, 4].
            att_distrib_param_eol (list, optional): parameters used to
                distribute attitude values regarding eol options within the
                population. Defaults to [0.544, 0.1].
            att_distrib_param_reuse (list, optional): parameters used to
                distribute attitude values regarding purchase options within
                the population. Defaults to [0.223, 0.262].
            original_recycling_cost (list, optional): parameters used to
                distribute recycling costs at the start of the simulation.
                Defaults to [0.106, 0.128, 0.117].
            recycling_learning_shape_factor (float, optional): learning effect
                parameter defining how recycling costs decrease as recycling
                volumes increase. Defaults to -0.39.
            repairability (float, optional): share of products that can be
                repaired. Defaults to 0.55.
            original_repairing_cost (list, optional): parameters used to
                distribute repair costs at the start of the simulation.
                Defaults to [0.1, 0.45, 0.23].
            repairing_learning_shape_factor (float, optional): learning effect
                parameter defining how repair costs decrease as repair volumes
                increase. Defaults to -0.31.
            scndhand_mkt_pric_rate (list, optional): parameters used to
                distribute the ratio between used product price to new product
                price. Defaults to [0.4, 0.2].
            fsthand_mkt_pric (float, optional): new product price. Defaults to
                0.45.
            fsthand_mkt_pric_reg_param (list, optional):  parameters used to
                model new product price decrease. Defaults to [1, 0.04].
            refurbisher_margin (list, optional):  parameters used to
                distribute refurbisher's margins. Defaults to [0.4, 0.6, 0.5].
            purchase_choices (dict, optional): purchase options that are
                available. Defaults to {"new": True, "used": True, "certified":
                False}.
            product_mass_fractions (dict, optional): product material
                composition. Defaults to {"Product": 1, "Aluminum": 0.08,
                "Glass": 0.76, "Copper": 0.01, "Insulated cable": 0.012,
                "Silicon": 0.036, "Silver": 0.00032}.
            established_scd_mkt (dict, optional): defines if materials can be
                recycled in an established end market or not. Defaults to
                {"Product": True, "Aluminum": True, "Glass": True, "Copper":
                True, "Insulated cable": True, "Silicon": False, "Silver":
                False}.
            scd_mat_prices (dict, optional): parameters used to
                distribute secondary (scrap) material prices. Defaults to
                {"Product": [np.nan, np.nan, np.nan], "Aluminum":
                [0.66, 1.98, 1.32], "Glass": [0.01, 0.06, 0.035], "Copper":
                [3.77, 6.75, 5.75], "Insulated cable": [3.22, 3.44, 3.33],
                "Silicon": [2.20, 3.18, 2.69], "Silver": [453, 653, 582]}.
            virgin_mat_prices (dict, optional): parameters used to
                distribute primary (raw) material prices. Defaults to
                {"Product": [np.nan, np.nan, np.nan], "Aluminum":
                [1.76, 2.51, 2.14], "Glass": [0.04, 0.07, 0.055], "Copper":
                [4.19, 7.50, 6.39], "Insulated cable": [3.22, 3.44, 3.33],
                "Silicon": [2.20, 3.18, 2.69], "Silver": [453, 653, 582]}.
            material_waste_ratio (dict, optional): industrial material waste
                ratios. Defaults to {"Product": 0., "Aluminum": 0., "Glass":
                0., "Copper": 0., "Insulated cable": 0., "Silicon": 0.4,
                "Silver": 0.}.
            recovery_fractions (dict, optional): material recovery fractions
                from the recycling process. Defaults to {"Product": np.nan,
                "Aluminum": 0.92, "Glass": 0.85, "Copper": 0.72,
                "Insulated cable": 1, "Silicon": 0, "Silver": 0}.
            product_average_wght (float, optional): average weight of the
                product. Defaults to 0.1.
            mass_to_function_reg_coeff (float, optional): parameter used to
                model the product weight decrease. Defaults to 0.03.
            recycling_states (list, optional): states with recycling
                facilities. Defaults to [ 'Texas', 'Arizona', 'Oregon',
                'Oklahoma', 'Wisconsin', 'Ohio', 'Kentucky', 'South Carolina'].
            transportation_cost (float, optional): cost of product
                transportation. Defaults to 0.0314.
            used_product_substitution_rate (list, optional): substitution rate
                of used product to new product. Defaults to [0.6, 1, 0.8].
            imperfect_substitution (int, optional): additional substitution
                rate due to unintended consequences such as the rebound effect.
                Defaults to 0.
            epr_business_model (bool, optional): parameter to activate an
                extended producer responsibility (EPR) scenario. Defaults to
                False.
            recycling_process (dict, optional): parameter to activate different
                recycling process scenarios. Defaults to {"frelp": False,
                "asu": False, "hybrid": False}.
            dynamic_lifetime_model (dict, optional): parameter to activate
                product lifetime scenarios. Defaults to {"Dynamic lifetime":
                False, "d_lifetime_intercept": 15.9, "d_lifetime_reg_coeff":
                0.87, "Seed": False, "Year": 5, "avg_lifetime": 50}.
            extended_tpb (dict, optional): optional parameters for the
                decision-making model. Defaults to {"Extended tpb": False,
                "w_convenience": 0.28, "w_knowledge": -0.51,
                "knowledge_distrib": [0.5, 0.49]}.
            seeding (dict, optional): seeding scenario for used products.
                Defaults to {"Seeding": False, "Year": 10, "number_seed": 50}.
            seeding_recyc (dict, optional): seeding scenario for recycling
                products. Defaults to {"Seeding": False, "Year": 10,
                "number_seed": 50, "discount": 0.35}.
            hazardous_waste_regulation_enabled: bool - Whether hazardous waste regulations are enabled.
            landfill_solar_waste_acceptance_ratio: float - The ratio of landfills that accept solar waste. Defaults to 0.4.
            as per research from Taylor Curtis.
            rtn: bool - Whether to use the TAMU transportation, recycling, and landfill cost data.
            solar_cycle: bool - Whether to use the solar cycle landfill cost data.
            landfill_data_params: dict - Parameters for landfill data.
        """
        # Set up variables
        # att_distrib_param_eol[0] = calibration_n_sensitivity
        # att_distrib_param_reuse[0] = calibration_n_sensitivity_2
        # original_recycling_cost = [x * calibration_n_sensitivity_3 for x in
        #                          original_recycling_cost]
        # landfill_cost = [x * calibration_n_sensitivity_4 for x in
        #                landfill_cost]
        # att_distrib_param_eol[1] = att_distrib_param_eol[1] * \
        #   calibration_n_sensitivity_4
        # w_sn_eol = w_sn_eol * calibration_n_sensitivity_5

        super().__init__(seed=seed)
        # np.random.seed(self.seed)
        # random.seed(self.seed)
        self.seed = seed
        self.timestep = timestep
        self.rtn = rtn
        self.solar_cycle = solar_cycle
        self.landfill_data_params = landfill_data_params
        self.model_states = model_states

        #Set path for data saving
        testfolder = str(Path().resolve() / 'PV_ICE' / 'TEMP' / 'PCA')

        if not os.path.exists(testfolder):
            os.makedirs(testfolder)
        # print ("Your simulation will be stored in %s" % testfolder)


        SupportingMaterialFolder = str(Path().resolve()/ 'PV_ICE' / 'baselines' / 'SupportingMaterial')
        BaselinesFolder = str(Path().resolve()/ 'PV_ICE' / 'baselines')

        reedsFile = os.path.join(SupportingMaterialFolder, 'December Core Scenarios ReEDS Outputs Solar Futures v3a.xlsx')
        # print ("Input file is stored in %s" % reedsFile)

        rawdf = pd.read_excel(reedsFile,
                        sheet_name="new installs PV")

        rawdf.drop(columns=['Tech'], inplace=True)
        rawdf.set_index(['Scenario','Year','PCA', 'State'], inplace=True)

        scenarios = list(rawdf.index.get_level_values('Scenario').unique())
        PCAs = list(rawdf.index.get_level_values('PCA').unique())
        STATEs = list(rawdf.index.get_level_values('State').unique())

        self.file_names = file_name

        # Consumer agent resolution determines what each consumer agent represents
        self.consumer_agent_resolution = consumer_agent_resolution
        # Load the USPVDB and ReEDS data if using site-level consumer agents
        # this is needed to map sites to PCAs and get the utility-scale PV contribution factors
        # If using PCA-level consumer agents, need to load the pca_longlat file instead
        if self.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            self.uspvdb = pd.read_excel(os.path.join(
                os.path.dirname(__file__), 'USPVDB', 'uspvdb_v3_0_20250430_with_pca.xlsx'))
            self.uspvdb = self.uspvdb[self.uspvdb['PCA'] != PCA_MISSING_VALUE]

            if self.model_states is not None:
                self.uspvdb = self.uspvdb[self.uspvdb['p_state'].isin(self.model_states)]

            self.reeds_data = pd.read_excel(os.path.join(os.path.dirname(__file__),
                                                    'ReEDS', 'StdScen24_annual_balancingAreas_Mid_Case_CO2e_95by2035.xlsx'))
            if self.model_states is not None:
                self.reeds_data = self.reeds_data[self.reeds_data['state'].isin(self.model_states)]
            # Pre-calculate utility-scale PV contribution factor for each PCA
            self.reeds_data['utility_scale_pv_contribution_factor'] = \
            self.reeds_data['upv_MW'] / (self.reeds_data['upv_MW'] +
                                    self.reeds_data['distpv_MW'])

        GISfile = os.path.join(SupportingMaterialFolder, 'gis_centroid_n.csv')
        GIS = pd.read_csv(GISfile)
        GIS = GIS.set_index('id')
        GIS.head()
        GIS.loc['p1'].long

        # # 1. Create ReEDS Scenarios BASELINE Files

        # import PV_ICE
        r1 = PV_ICE.Simulation(name='Simulation1', path=testfolder)
        massmodulefilepath = os.path.join(Path().resolve().parent.parent.parent/ 'PV_ICE/baselines/baseline_modules_mass_US.csv')
        energymodulefilepath = os.path.join(Path().resolve().parent.parent.parent/ 'PV_ICE/baselines/baseline_modules_energy.csv')
        r1.createScenario(name='US', massmodulefile=massmodulefilepath, energymodulefile=energymodulefilepath)
        r1.scenMod_noCircularity() # Reeds Solar Future Study had circularity paths set to 0
        baseline = r1.scenario['US'].dataIn_m
        baseline = baseline.drop(columns=['new_Installed_Capacity_[MW]'])
        baseline.set_index('year', inplace=True)
        baseline.index = pd.PeriodIndex(baseline.index, freq='Y')  # Y -- Annual
        baseline.head()


        massmodulefile = os.path.join(BaselinesFolder, 'baseline_modules_mass_US.csv')

        with open(massmodulefile, newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)  # gets the first line
            row2 = next(reader)  # gets the first line

        row11 = 'year'
        for x in row1[1:]:
            row11 = row11 + ',' + x 

        row22 = 'year'
        for x in row2[1:]:
            row22 = row22 + ',' + x 
        if pca:
            for ii in range(len(rawdf.unstack(level=1))):
                PCA = rawdf.unstack(level=1).iloc[ii].name[1]
                SCEN = rawdf.unstack(level=1).iloc[ii].name[0]
                SCEN = SCEN.replace('+', '_')
                filetitle = SCEN+'_'+PCA +'.csv'
                subtestfolder = os.path.join(testfolder, 'PCAs')
                if not os.path.exists(subtestfolder):
                    os.makedirs(subtestfolder)
                filetitle = os.path.join(subtestfolder, filetitle)
                A = rawdf.unstack(level=1).iloc[ii]
                A = A.droplevel(level=0)
                A.name = 'new_Installed_Capacity_[MW]'
                A = pd.DataFrame(A)
                A.index = pd.PeriodIndex(A.index, freq='A')
                A = pd.DataFrame(A)
                A['new_Installed_Capacity_[MW]'] = A['new_Installed_Capacity_[MW]'] * 0.85  # 85% of capacity is Silicon PV?
                A['new_Installed_Capacity_[MW]'] = A['new_Installed_Capacity_[MW]'] * 1000   # ReEDS file is in GW.
                # Add other columns
                A = pd.concat([A, baseline.reindex(A.index)], axis=1)

                header = row11 + '\n' + row22 + '\n'

                with open(filetitle, 'w', newline='') as ict:
                    # Write the header lines, including the index variable for
                    # the last one if you're letting Pandas produce that for
                    # you. (see above).
                    for line in header:
                        ict.write(line)

                    #    savedata.to_csv(ict, index=False)
                    A.to_csv(ict, header=False)

                # Create Scenarios in PV_ICE
                # Rename difficult characters from Scenarios Names
                simulationname = scenarios
                simulationname = [w.replace('+', '_') for w in simulationname]
                SFscenarios = [simulationname[0], simulationname[4], simulationname[8]]

        excel_file_path = 'TEMP/Table A-1_Global PV Recyclers_states.xlsx'

        if geopy:
            os.chdir('../../../')
            df = pd.read_excel(excel_file_path, skiprows=[0])  # Skip the first row
            df = df[df['Country'] == 'United States of America']
            df_recycler_out = pd.DataFrame(columns=["Recycler Name",
                                                    "Long", "Lat"])
            # Initialize a geocoder to fetch latitude and longitude
            # coordinates
            geolocator = Nominatim(user_agent="recycler_locator")
            for index, row in df.iterrows():
                recycler_name = row['Recycler Name']
                state = row['State']
                city = row['City']
                try:
                    # Fetch latitude and longitude for the recycler
                    location = geolocator.geocode(f"{city}, {state}",
                                                  timeout=10)
                    if location:
                        recycler_latitude = location.latitude
                        recycler_longitude = location.longitude
                    else:
                        recycler_latitude = None
                        recycler_longitude = None
                except Exception as e:
                    print(f"Error geocoding for {recycler_name}: {str(e)}")
                    recycler_latitude = None
                    recycler_longitude = None
                data_to_append = pd.DataFrame(
                    {"Recycler Name": recycler_name,
                     "Long": recycler_longitude, "Lat": recycler_latitude})
                df_recycler_out = pd.concat(
                    [df_recycler_out, data_to_append], ignore_index=True)
            df_recycler_out.to_csv("../../../TEMP/recycler_data.csv",
                                   index=False)

        # else:
        #    print("\nthis is path, ", os.getcwd())
        #    df = pd.read_csv('../../../TEMP/recycler_data.csv')

        pca_longlat = pd.DataFrame(columns=["PCA", "Long", "Lat"])

        #     #### Create the 3 Scenarios and assign Baselines

        if self.rtn:
            # If using the RTN model results from Texas A&M University
            # load the recycler data from the RTN model
            self.recycler_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "RTN", "recycler_data.csv"))
        else:
            self.recycler_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "TEMP", 
                                                    "recycler_data.csv"))

        if pca_scenario:
            i = 0
            r1 = PV_ICE.Simulation(name=SFscenarios[i], path=testfolder)
            energymodulefilepath = os.path.join(Path().resolve().parent.parent.parent/ 'PV_ICE/baselines/baseline_modules_energy.csv')
            baslinefolderpath = os.path.join(Path().resolve().parent.parent.parent/ 'PV_ICE/baselines')

            for jj in range(0, len(PCAs)):

                filetitle = SFscenarios[i]+'_'+PCAs[jj]+'.csv'
                filetitle = os.path.join(testfolder, 'PCAs', filetitle)   
                print("filetitle:", filetitle) 
                r1.createScenario(name=PCAs[jj], massmodulefile=filetitle, energymodulefile=energymodulefilepath)
                r1.scenario[PCAs[jj]].addMaterials(['glass', 'silicon', 'silver', 'copper', 'aluminium_frames'], baselinefolder=baslinefolderpath)
                output_filename = f"matdataout_{SFscenarios[i]}_{PCAs[jj]}_.csv"
                output_filename0 = f"dataOut_{SFscenarios[i]}_{PCAs[jj]}_.csv"
                output_filename_in = f"datain_{SFscenarios[i]}_{PCAs[jj]}_.csv"

                r1.trim_Years(startYear=2010, endYear=2050)

                # All -- but these where not included in the Reeds initial study as we didn't have encapsulant or backsheet
                # r1.scenario[PCAs[jj]].addMaterials(['glass', 'silicon', 'silver', 'copper', 'aluminium_frames', 'encapsulant', 'backsheet'], baselinefolder=r'..\baselines')
                r1.scenario[PCAs[jj]].latitude = GIS.loc[PCAs[jj]].lat
                r1.scenario[PCAs[jj]].longitude = GIS.loc[PCAs[jj]].long

                data_to_append = pd.DataFrame(
                    {"PCA": [PCAs[jj]], "Long": [GIS.loc[PCAs[jj]].long],
                     "Lat": [GIS.loc[PCAs[jj]].lat]})

                # Append data to pca_longlat using pd.concat
                pca_longlat = pd.concat([pca_longlat, data_to_append],
                                        ignore_index=True)

            # Calculate distances from the current PCA to all recyclers
                # for index, row in df.iterrows():
                #     if geopy:
                #         recycler_name = row['Recycler Name']
                #         state = row['State']
                #         city = row['City']

                #         try:
                #             # Fetch latitude and longitude for the recycler
                #             location = geolocator.geocode(f"{city}, {state}", timeout=10)
                #             if location:
                #                 recycler_latitude = location.latitude
                #                 recycler_longitude = location.longitude
                #             else:
                #                 recycler_latitude = None
                #                 recycler_longitude = None
                #         except Exception as e:
                #             print(f"Error geocoding for {recycler_name}: {str(e)}")
                #             recycler_latitude = None
                #             recycler_longitude = None
                #     else:
                #         recycler_name = row['Recycler Name']
                #         recycler_latitude = row['Latitude']
                #         recycler_longitude = row['Longitude']

                #     # Calculate the distance from the PCA to the recycler
                #     if recycler_latitude is not None and \
                #             recycler_longitude is not None:
                #         pca_latitude = GIS.loc[PCAs[jj]].lat
                #         pca_longitude = GIS.loc[PCAs[jj]].long
                #         distance = haversine(pca_latitude, pca_longitude, recycler_latitude, recycler_longitude)

                #         # Append the distance to the DataFrame
                #         data_to_append = pd.DataFrame({"PCA": PCAs[jj], "Recycler": recycler_name, "Distance (km)": distance}, ignore_index=True)
                # distance_df = pd.concat([distance_df, data_to_append], ignore_index=True)

            #     # # Calculate distances from the current PCA to all recyclers
                # for index, row in df.iterrows():
                #     if geopy:
                #         recycler_name = row['Recycler Name']
                #         state = row['State']
                #         city = row['City']

                #         try:
                #             # Fetch latitude and longitude for the recycler
                #             location = geolocator.geocode(f"{city}, {state}", timeout=10)
                #             if location:
                #                 recycler_latitude = location.latitude
                #                 recycler_longitude = location.longitude
                #             else:
                #                 recycler_latitude = None
                #                 recycler_longitude = None
                #         except Exception as e:
                #             print(f"Error geocoding for {recycler_name}: {str(e)}")
                #             recycler_latitude = None
                #             recycler_longitude = None
                #     else:
                #         recycler_name = row['Recycler Name']
                #         recycler_latitude = row['Latitude']
                #         recycler_longitude = row['Longitude']

                #     # Calculate the distance from the PCA to the recycler
                #     if recycler_latitude is not None and \
                #             recycler_longitude is not None:
                #         pca_latitude = GIS.loc[PCAs[jj]].lat
                #         pca_longitude = GIS.loc[PCAs[jj]].long
                #         distance = haversine(pca_latitude, pca_longitude, recycler_latitude, recycler_longitude)

                #         # Append the distance to the DataFrame
                #         distance_df = distance_df.append({"PCA": PCAs[jj], "Recycler": recycler_name, "Distance (km)": distance}, ignore_index=True)

                # distance_df.to_csv("../../../TEMP/pca_recycler_distances.csv", index=False)

                pca_longlat.to_csv("../../../TEMP/pca_longlat.csv",
                                   index=False)

                self.df_in = r1.scenario[PCAs[jj]].dataIn_m
                self.df_in.to_csv(output_filename_in, index=False)
                self.year_column = self.df_in['year']

                r1.calculateMassFlow()

                self.df0 = r1.scenario[PCAs[jj]].dataOut_m
                self.df0 = self.df0.join(self.year_column)
                self.df0.to_csv(output_filename0, index=False)

                self.df = r1.scenario[PCAs[jj]].material['silicon'].matdataOut_m
                self.df = self.df.join(self.year_column)
                self.df.to_csv(output_filename, index=False)

        # Load PCA data if needed for distance calculations and for consumer agent reporting
        if self.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            self.pca_data = pd.read_csv("../../../TEMP/pca_longlat.csv")  # Replace with your PCA data file

        if pv_ice:

            testfolder = str(Path().resolve().parent.parent)
            r1 = PV_ICE.Simulation(name='Simulation1', path=testfolder)
            r1.createScenario(name='standard', massmodulefile=r'./baselines/baseline_modules_mass_US.csv')
            r1.scenario['standard'].addMaterial('glass', massmatfile=r'./baselines/baseline_material_mass_glass.csv' )
            r1.scenario['standard'].addMaterial('silicon', massmatfile=r'./baselines/baseline_material_mass_silicon.csv' )

            self.df0 = r1.scenario['standard'].dataIn_m
            self.df0.to_csv("df1_dataout.csv", index=False)

            self.df0 = r1.calculateMassFlow()
            print("\n df0", self.df0)
            self.df1 = r1.scenario['standard'].dataOut_m
            print("Keys", self.df1.keys())
            print("\nFirst df", self.df1.head())
            self.df1.to_csv("df1_dataout1.csv", index=False)

            self.df2 = r1.scenario['standard'].material['silicon'].matdataOut_m
            self.df2.to_csv("df2_matdataout.csv", index=False)

        if calculate_distances:
            # Function to calculate the Haversine distance between two points
            # given their latitude and longitude
            def haversine(lat1, lon1, lat2, lon2):
                # Convert latitude and longitude from degrees to radians
                lat1, lon1, lat2, lon2 = map(radians,
                                             [lat1, lon1, lat2, lon2])

                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * \
                    sin(dlon / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                return 6371 * c  # Radius of the Earth in kilometers

            def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
                # Convert latitude and longitude from degrees to radians
                lat1, lon1, lat2, lon2 = map(np.radians,
                                             [lat1, lon1, lat2, lon2])

                # Haversine formula
                # Convert inputs to 2D arrays for broadcasting
                # Let M and N be the lengths of lat1/lon1 and lat2/lon2 respectively
                # Before broadcasting:
                # lat1, lon1: shape (M,)
                # lat2, lon2: shape (N,)
                # After broadcasting: 
                # lat2 and lon2 become shape (N, 1)
                # Resulting distance matrix will have shape (N, M)
                # This computes distances from each point in (lat2, lon2) to all points in (lat1, lon1)
                dlon = lon2[:, np.newaxis] - lon1 
                dlat = lat2[:, np.newaxis] - lat1 
                a = np.sin(dlat / 2)**2 + np.cos(lat1[np.newaxis, :]) * np.cos(lat2[:, np.newaxis]) * \
                    np.sin(dlon / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                return 6371 * c  # Radius of the Earth in kilometers
            
            if self.solar_cycle:
                # If using the solar cycle landfill cost data
                landfills_data = pd.read_csv(os.path.join(
                    os.path.dirname(__file__),
                    "SolarCycle", "wbj_solar_cycle_combined.csv"),
                    index_col=0)
                
            else:
                landfills_data = pd.read_csv("../../../TEMP/" +
                                                self.file_names['Landfill data'])
                
            if self.consumer_agent_resolution == ConsumerAgentResolution.PCA:

                # Create an empty DataFrame to store distances
                distance_df = pd.DataFrame(columns=self.pca_data['PCA'],
                                        index=self.recycler_data['Recycler Name'])
                distance_df2 = pd.DataFrame(columns=self.pca_data['PCA'],
                                            index=landfills_data[self.landfill_data_params['landfill_name_column']])

                # Calculate distances between each PCA and each recycler
                for pca_index, pca_row in self.pca_data.iterrows():
                    pca_lat = pca_row['Lat']
                    pca_lon = pca_row['Long']

                    for recycler_index, recycler_row in self.recycler_data.iterrows():
                        recycler_lat = recycler_row['Latitude']
                        recycler_lon = recycler_row['Longitude']
                        distance = haversine(float(pca_lat), float(pca_lon),
                                            float(recycler_lat),
                                            float(recycler_lon))

                        # Fill in the distance in the DataFrame
                        distance_df.at[recycler_row['Recycler Name'],
                                    pca_row['PCA']] = distance

                    for landfills_index, landfills_row in \
                            landfills_data.iterrows():
                        landfills_lat = landfills_row['Latitude']
                        landfills_lon = landfills_row['Longitude']

                        distance2 = haversine(pca_lat, pca_lon,
                                            landfills_lat, landfills_lon)

                        # Fill in the distance in the DataFrame
                        distance_df2.at[landfills_row[self.landfill_data_params['landfill_name_column']],
                                        pca_row['PCA']] = distance2

                # Save the distances to a CSV file
                if self.rtn:
                    # If using the RTN model results from Texas A&M University
                    # save the distances to the RTN model folder
                    distance_df.to_csv(os.path.join(os.path.dirname(__file__), "RTN", "pca_recycler_distances.csv"))
                    distance_df2.to_csv(os.path.join(os.path.dirname(__file__), "RTN",
                                                    self.file_names['PCA-landfill distances']))                   
                else:
                    distance_df.to_csv("../../../TEMP/pca_recycler_distances.csv")
                    if self.solar_cycle:
                        # If using the solar cycle landfill cost data
                        # save the landfill distances to the solar cycle folder
                        distance_df2.to_csv(os.path.join(
                        os.path.dirname(__file__), "SolarCycle",
                        self.file_names['PCA-landfill distances']))
                    else:
                        distance_df2.to_csv("../../../TEMP/" +
                                        self.file_names['PCA-landfill distances'])
                
            if self.consumer_agent_resolution == ConsumerAgentResolution.SITE:

                # Calculate distances between each site and each recycler and landfill
                
                site_lats = self.uspvdb['Latitude'].to_numpy().astype(float)
                site_lons = self.uspvdb['Longitude'].to_numpy().astype(float)
                recycler_lats = self.recycler_data['Latitude'].to_numpy().astype(float)
                recycler_lons = self.recycler_data['Longitude'].to_numpy().astype(float)
                landfill_lats = landfills_data['Latitude'].to_numpy().astype(float)
                landfill_lons = landfills_data['Longitude'].to_numpy().astype(float)

                distance_matrix_recycler = haversine_vectorized(
                    site_lats, site_lons, recycler_lats, recycler_lons)
                distance_matrix_landfill = haversine_vectorized(
                    site_lats, site_lons, landfill_lats, landfill_lons)
                
                landfill_distance_df = pd.DataFrame(distance_matrix_landfill,
                                                        index=landfills_data[self.landfill_data_params['landfill_name_column']],
                                                        columns=self.uspvdb['case_id'])
                                                    
                site_recycler_distance_df = pd.DataFrame(distance_matrix_recycler,
                                                        index=self.recycler_data['Recycler Name'],
                                                        columns=self.uspvdb['case_id'])
                if self.rtn:
                    # If using the RTN model results from Texas A&M University
                    # save the distances to the RTN model folder
                    site_recycler_distance_df.to_csv(os.path.join(
                        os.path.dirname(__file__), "RTN",
                        "site_recycler_distances.csv"))
                    landfill_distance_df.to_csv(os.path.join(
                        os.path.dirname(__file__), "RTN",
                        "site_landfill_distances.csv"))
                else:
                    site_recycler_distance_df.to_csv("../../../TEMP/site_recycler_distances.csv")
                    if self.solar_cycle:
                        # If using the solar cycle landfill cost data
                        # save the landfill distances to the solar cycle folder
                        site_recycler_distance_df.to_csv("../../../TEMP/site_recycler_distances.csv")
                        landfill_distance_df.to_csv(os.path.join(
                            os.path.dirname(__file__), "SolarCycle",
                            "site_landfill_distances.csv"))
                    else:
                        landfill_distance_df.to_csv("../../../TEMP/site_landfill_distances.csv")

                hazardous_landfills_data = pd.read_csv("../../../TEMP/Landfills_data_SA.csv")
                hazardous_landfill_lats = hazardous_landfills_data['Latitude'].to_numpy().astype(float)
                hazardous_landfill_lons = hazardous_landfills_data['Longitude'].to_numpy().astype(float)
                hazardous_landfill_distance_matrix = haversine_vectorized(
                    site_lats, site_lons, hazardous_landfill_lats, hazardous_landfill_lons)
                hazardous_landfill_distance_df = pd.DataFrame(hazardous_landfill_distance_matrix,
                                                        index=hazardous_landfills_data['Facility Name'],
                                                        columns=self.uspvdb['case_id'])
                hazardous_landfill_distance_df.to_csv(
                    os.path.join(
                        os.path.dirname(__file__), "TEMP", 'hazardous_site_landfill_distances.csv'))

        self.correct_mat_factor = pd.read_csv(
            '../../../TEMP/correct_mat_factor.csv')    
        self.hazardous_landfill_cost_df = pd.read_csv(
            '../../../TEMP/' + self.file_names['Hazardous landfill data'])

        self.correct_mat_factor = transform_timeseries_timestep(
            self.correct_mat_factor, self.timestep, scale=False)
        
        self.data = pd.read_excel(reedsFile)  # this is the pca file

        # Filter to only include specified model states
        if self.model_states is not None:
            print("Model will run for the following states:", self.model_states)
            self.data = self.data[self.data['State'].isin(self.model_states)]

        if self.rtn:
            # If using the RTN model results from Texas A&M University
            # load the recycling and landfill data from the RTN model
            self.recycler_distance_df = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "RTN", "pca_recycler_distances.csv"))
            self.recycling_costs_df = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "RTN", "RecyclingCostsbyYearPCA.csv"))
            self.landfill_distance_df = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "RTN",
                             self.file_names['PCA-landfill distances']))
            self.landfill_cost_df = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "RTN",
                             self.file_names['Landfill data']))
        else:
            if consumer_agent_resolution == ConsumerAgentResolution.PCA:

                self.recycler_distance_df = pd.read_csv(
                '../../../TEMP/pca_recycler_distances.csv')
                if self.solar_cycle:
                    # If using the solar cycle landfill cost data
                    self.landfill_distance_df = pd.read_csv(os.path.join(
                        os.path.dirname(__file__), "SolarCycle",
                        self.file_names['PCA-landfill distances']))
                else:
                    self.landfill_distance_df = pd.read_csv(
                        '../../../TEMP/' + self.file_names['PCA-landfill distances'])
                self.hazardous_landfill_distance_df = pd.read_csv(
            '../../../TEMP/' + self.file_names['Hazardous PCA-landfill distances'])
                
            elif consumer_agent_resolution == ConsumerAgentResolution.SITE:

                self.recycler_distance_df = pd.read_csv(
                '../../../TEMP/site_recycler_distances.csv')
                if self.solar_cycle:
                    # If using the solar cycle landfill cost data
                    self.landfill_distance_df = pd.read_csv(os.path.join(
                        os.path.dirname(__file__), "SolarCycle",
                        "site_landfill_distances.csv"))
                else:
                    self.landfill_distance_df = pd.read_csv(
                        '../../../TEMP/site_landfill_distances.csv')
                self.hazardous_landfill_distance_df = pd.read_csv(
            '../../../TEMP/hazardous_site_landfill_distances.csv')
                
            self.recycling_costs_df = pd.DataFrame()
            
            if self.solar_cycle:
                # If using the solar cycle landfill cost data
                self.landfill_cost_df = pd.read_csv(os.path.join(
                    os.path.dirname(__file__),
                    "SolarCycle", "wbj_solar_cycle_combined.csv"),
                    index_col=0)
            else:
                self.landfill_cost_df = pd.read_csv(
                '../../../TEMP/' + self.file_names['Landfill data'])

        self.pv_ice_yearly_waste = 0

        self.num_consumers = self.get_num_consumers(num_consumers)
        self.consumers_node_degree = consumers_node_degree
        self.consumers_network_type = consumers_network_type
        if self.consumer_agent_resolution is ConsumerAgentResolution.PCA:
            self.agent_pca_map = self.create_agent_pca_map(self.num_consumers)
        elif self.consumer_agent_resolution is ConsumerAgentResolution.SITE:
            self.agent_site_map = self.create_agent_site_map()
        self.num_recyclers = len(self.recycler_distance_df[
            'Recycler Name'].unique())
        self.recycler_names = self.recycler_distance_df[
            'Recycler Name'].to_list()
        self.num_producers = num_producers
        self.num_prod_n_recyc = self.num_recyclers + num_producers
        self.prod_n_recyc_node_degree = prod_n_recyc_node_degree
        self.prod_n_recyc_network_type = prod_n_recyc_network_type
        self.num_refurbishers = num_refurbishers
        self.init_eol_rate = init_eol_rate
        self.init_purchase_choice = init_purchase_choice
        self.clock = 0
        self.last_step = last_step
        self.sa_landfill_costs = sa_landfill_costs
        self.hazardous_waste_management_cost = hazardous_waste_management_cost
        self.hazardous_waste_regulation_enabled = hazardous_waste_regulation_enabled
        # prune the list of landfills to those accepting solar waste using the acceptance ratio
        # passed during initialization
        self.landfill_solar_waste_acceptance_ratio = landfill_solar_waste_acceptance_ratio
        if self.landfill_solar_waste_acceptance_ratio < 1.0:
            self.filter_landfills_accepting_solar_waste()

        # ! Initialize model with PV_ICE historical installed cap
        # self.total_number_product = total_number_product
        # subset_df_init_cap = self.df0[self.df0['year'] < 2020]

        # self.pca = self.create_agents(num_consumers)[self.unique_id][0]
        all_pca_df_in = pd.DataFrame()
        all_pca_df_out = pd.DataFrame()
        valid_pcas = PCAs
        # Filter to only include specified model states
        if self.model_states is not None:
            valid_pcas = self.data[self.data['State'].isin(self.model_states)]['PCA'].unique().tolist()
        for pca in valid_pcas:
            subset_df_init_cap = pd.read_csv(
                "datain_95-by-35.Adv_" + pca + "_.csv")
            subset_df_init_cap['pca'] = pca
            all_pca_df_in = pd.concat([all_pca_df_in, subset_df_init_cap])
            subset_df_init_cap_out = pd.read_csv(
                "dataOut_95-by-35.Adv_" + pca + "_.csv")
            subset_df_init_cap_out['pca'] = pca
            all_pca_df_out = pd.concat([all_pca_df_out,
                                        subset_df_init_cap_out])
        all_pca_df_in.to_csv('all_pca_datain_95-by-35.Adv.csv')
        all_pca_df_out.to_csv('all_pca_dataOut_95-by-35.Adv.csv')

        all_pca_df_in = all_pca_df_in.groupby('year', as_index=False).sum()
        subset_df_init_cap = all_pca_df_in[all_pca_df_in['year'] < 2020]
        subset_df_init_cap = transform_timeseries_timestep(subset_df_init_cap, self.timestep)
        subset_df_init_cap = subset_df_init_cap[
            'new_Installed_Capacity_[MW]'].tolist()
        self.total_number_product = subset_df_init_cap

        self.copy_total_number_product = self.total_number_product.copy()
        self.mass_to_function_reg_coeff = mass_to_function_reg_coeff

        # Create an empty DataFrame to store the material factors
        df_mat_factor = pd.DataFrame()

        # List of valid materials
        valid_materials = ["aluminium_frames.csv", "backsheet.csv", "copper.csv", "encapsulant.csv", "glass.csv", "silicon.csv", "silver.csv"]

        # Loop through files in the specified folder
        baseline_folder = "../../baselines"
        output_folder = "../../material_factor_output_folder"


        for filename in os.listdir(baseline_folder):
            if filename.startswith("baseline_material_mass_") and filename.endswith(".csv") and not filename.endswith("_cdte.csv") and not filename.endswith("cadmium.csv") and not filename.endswith("tellurium.csv"):
                material_name = filename.split("_")[3:]
                material_name = '_'.join(material_name)
                if material_name in valid_materials:
                    file_path = os.path.join(baseline_folder, filename)
                    data = pd.read_csv(file_path, skiprows=[1])
                    # Add the material mass per m^2 column to the existing DataFrame
                    df_mat_factor["year"] = data["year"]  
                    try:
                        df_mat_factor[material_name] = data["mat_massperm2"].astype(float) / 1000
                    except ValueError:
                        # Handle non-numeric values in the column (e.g., strings)
                        print(f"Skipping non-numeric values in {material_name} column")
                        continue

        df_mat_factor["total_massperm2"] = df_mat_factor.drop('year', axis=1).sum(axis=1)

        os.makedirs(output_folder, exist_ok=True)
        output_filename = os.path.join(output_folder, "mat_factor.csv")
        df_mat_factor.to_csv(output_filename, index=False)
        df_mat_factor = transform_timeseries_timestep(df_mat_factor, self.timestep, scale=False)
        self.pvice_mat_factor = df_mat_factor
        self.weight_factor = 0
        self.max_storage = max_storage
        self.avg_weight_factor_stored_pv = 0

        self.iteration = 0
        self.running = True
        self.color_map = []
        self.theory_of_planned_behavior = theory_of_planned_behavior
        self.all_EoL_pathways = all_EoL_pathways
        self.purchase_options = purchase_choices
        self.avg_failure_rate = failure_rate_alpha

        self.pca_outputs = {}
        self.pca_install_test = 0
        self.pca_install = {}
        self.pca_tot_waste_w = {}
        self.pca_tot_waste_m2 = {}
        for pca in valid_pcas:
            pathway_dict = {}
            for pathway in self.all_EoL_pathways.keys():
                pathway_dict[pathway] = 0
            self.pca_outputs[pca] = pathway_dict
            self.pca_install[pca] = 0
            self.pca_tot_waste_w[pca] = 0
            self.pca_tot_waste_m2[pca] = 0
        self.refurbisher_outputs_watt = {}
        self.refurbisher_outputs_kg = {}
        for pathway in self.all_EoL_pathways.keys():
            self.refurbisher_outputs_watt[pathway] = 0
            self.refurbisher_outputs_kg[pathway] = 0

        # ! Changed from total_number_product to the class value (which is
        # ! PV_ICE based)
        self.original_num_prod = self.total_number_product
        self.avg_lifetime = product_lifetime
        self.fsthand_mkt_pric = fsthand_mkt_pric
        self.fsthand_mkt_pric_reg_param = fsthand_mkt_pric_reg_param
        self.repairability = repairability
        self.total_waste = 0
        self.total_yearly_new_products = 0
        self.sold_repaired_waste = 0
        self.past_sold_repaired_waste = 0
        self.repairable_volume_recyclers = 0
        self.consumer_used_product = 0
        self.recycler_repairable_waste = 0
        self.yearly_repaired_waste = 0
        self.imperfect_substitution = imperfect_substitution
        perceived_behavioral_control = [np.nan] * len(all_EoL_pathways)
        self.willingness = np.asmatrix(np.zeros((self.num_prod_n_recyc,
                                                self.num_prod_n_recyc)))
        self.product_mass_fractions = product_mass_fractions
        self.material_waste_ratio = material_waste_ratio
        self.established_scd_mkt = established_scd_mkt
        self.recovery_fractions = recovery_fractions

        pvice_mat_factor_copy = self.pvice_mat_factor[
            self.pvice_mat_factor['year'] == 2020]
        conversion_factor = \
            pvice_mat_factor_copy['total_massperm2'].iloc[0]
        all_data_out_pca = pd.read_csv(
            "all_pca_dataOut_95-by-35.Adv.csv", index_col=0)
        columns_to_expand = ['Yearly_Sum_Power_atEOL', 'Yearly_Sum_Area_atEOL']
        all_data_out_pca = transform_pca_timeseries_timestep(all_data_out_pca, self.timestep, filtered_columns=columns_to_expand)
        all_data_out_pca = all_data_out_pca.groupby(
            'year', as_index=False).mean(numeric_only=True)
        data_out_pca_copy = all_data_out_pca[
            all_data_out_pca['year'] == 2020]
        waste_in_w = \
            data_out_pca_copy['Yearly_Sum_Power_atEOL'].iloc[0]
        waste_in_m2 = \
            data_out_pca_copy['Yearly_Sum_Area_atEOL'].iloc[0]
        waste_w_m2 = waste_in_m2 / waste_in_w
        pv_ice_product_average_wght = conversion_factor * waste_w_m2

        self.product_average_wght = pv_ice_product_average_wght
        self.dynamic_product_average_wght = pv_ice_product_average_wght
        self.yearly_product_wght = pv_ice_product_average_wght

        self.transportation_cost = transportation_cost
        self.hazardous_transportation_cost = hazardous_transportation_cost
        self.epr_business_model = epr_business_model
        # Here we keep the old code regarding landfill costs. This does not
        # affect the updates made during the NSF convergence project - phase I
        self.average_landfill_cost = sum(landfill_cost) / len(landfill_cost)
        self.installer_recycled_amount = 0
        # Change eol_pathways depending on business model
        if self.epr_business_model:
            self.all_EoL_pathways["landfill"] = False
        # Dynamic lifetime model
        self.dynamic_lifetime_model = dynamic_lifetime_model
        self.extended_tpb = extended_tpb
        self.seeding = seeding
        self.seeding_recyc = seeding_recyc
        self.tclp_params = tclp_params

        self.all_gba = pd.read_excel(reedsFile)  #importing all grid balancing areas in an excel file

        self.cost_seeding = 0
        self.product_lifetime = product_lifetime
        self.d_product_lifetimes = []
        self.update_dynamic_lifetime()
        self.original_recycling_cost = original_recycling_cost
        self.recycling_process = recycling_process
        self.list_consumer_id = list(range(self.num_consumers))
        random.shuffle(self.list_consumer_id)
        self.list_consumer_id_seed = list(range(self.num_consumers))
        random.shuffle(self.list_consumer_id_seed)
        # Change recovery fractions and recycling costs depending on recycling
        # process
        self.recycling_process_change()
        self.product_growth = product_growth
        self.growth_threshold = growth_threshold
        # Initialize Regulator parameters
        unique_states = self.data.loc[:, 'State'].unique().tolist()
        self.num_regulators = len(unique_states)
        self.regulator_state_map = self.create_regulator_state_map()
        # Create a map of agents to their unique IDs
        # This is used to access agents by their unique ID
        self.agent_map = {}
        # Builds graph and defines scheduler
        self.H1 = self.init_network(self.consumers_network_type,
                                    self.num_consumers,
                                    self.consumers_node_degree, rewiring_prob)
        self.H2 = self.init_network(self.prod_n_recyc_network_type,
                                    self.num_prod_n_recyc,
                                    self.prod_n_recyc_node_degree,
                                    rewiring_prob)
        self.H3 = self.init_network("complete graph", self.num_refurbishers,
                                    "NaN", rewiring_prob)
        self.H4 = self.init_network("complete graph", self.num_regulators,
                                    "NaN", rewiring_prob)
        self.G = nx.disjoint_union(self.H1, self.H2)
        self.G = nx.disjoint_union(self.G, self.H3)
        self.G = nx.disjoint_union(self.G, self.H4)
        self.grid = NetworkGrid(self.G)
        # Compute distance for the repair, sell, recycle, landfill and storage
        # pathways. Assumptions: 1) Only certain states have recycling
        # facilities, 2) The refurbisher who performs repair and
        # landfill site are both within the state of the PV owner, 3) Sales of
        # old PV modules occur across the whole US, randomly, 4) Storage
        # occurs on site and so there is no associated transportation.
        # (See consumer module for sales of old PV modules).
        self.all_states = ['Texas', 'California', 'Montana', 'New Mexico',
                           'Arizona', 'Nevada', 'Colorado', 'Oregon',
                           'Wyoming', 'Michigan', 'Minnesota', 'Utah', 'Idaho',
                           'Kansas', 'Nebraska', 'South Dakota', 'Washington',
                           'North Dakota', 'Oklahoma', 'Missouri', 'Florida',
                           'Wisconsin', 'Georgia', 'Illinois', 'Iowa',
                           'New York', 'North Carolina', 'Arkansas', 'Alabama',
                           'Louisiana', 'Mississippi', 'Pennsylvania', 'Ohio',
                           'Virginia', 'Tennessee', 'Kentucky', 'Indiana',
                           'Maine', 'South Carolina', 'West Virginia',
                           'Maryland', 'Massachusetts', 'Vermont',
                           'New Hampshire', 'New Jersey', 'Connecticut',
                           'Delaware', 'Rhode Island']

        self.states = pd.read_csv("../../../StatesAdjacencyMatrix.csv").to_numpy()
        # Compute distances
        self.mean_distance_within_state = np.nanmean(
            np.where(self.states != 0, self.states, np.nan)) / 2
        self.states_graph = nx.from_numpy_array(self.states)
        nodes_states_dic = \
            dict(zip(list(self.states_graph.nodes),
                     list(pd.read_csv("../../../StatesAdjacencyMatrix.csv"))))
        self.states_graph = nx.relabel_nodes(self.states_graph,
                                             nodes_states_dic)
        self.recycling_states = recycling_states
        distances_to_recyclers = []
        distances_to_recyclers = self.shortest_paths(
            self.recycling_states, distances_to_recyclers)
        self.mn_mx_av_distance_to_recycler = [
            min(distances_to_recyclers), max(distances_to_recyclers),
            sum(distances_to_recyclers) / len(distances_to_recyclers)]
        # Compute transportation costs
        self.transportation_cost_rcl = [
            x * self.transportation_cost / 1E3 *
            self.dynamic_product_average_wght for x in
            self.mn_mx_av_distance_to_recycler]

        # ! TODO: change landfill transportation costs
        self.transportation_cost_rpr_ldf = self.mean_distance_within_state * \
            self.transportation_cost / 1E3 * self.dynamic_product_average_wght

        # ! We remove the use of the recycling distances calculated with the
        # ! shortest path algorithm to use the pca-recycler distances instead
        # Add transportation costs to pathways' costs
        # self.original_recycling_cost = [sum(x) for x in zip(
        #    self.original_recycling_cost, self.transportation_cost_rcl)]

        # ! we keep the assumption that repairing costs is the mean distance
        # ! within states
        original_repairing_cost = [x + self.transportation_cost_rpr_ldf for
                                   x in original_repairing_cost]
        # Create agents, G nodes labels are equal to agents' unique_ID
        for node in self.G.nodes():
            if node < self.num_consumers:
                a = Consumers(node, self, product_growth, failure_rate_alpha,
                              perceived_behavioral_control, w_sn_eol,
                              w_pbc_eol, w_a_eol, w_sn_reuse, w_pbc_reuse,
                              w_a_reuse, None, hoarding_cost,
                              used_product_substitution_rate,
                              att_distrib_param_eol, att_distrib_param_reuse,
                              max_storage, consumers_distribution,
                              product_distribution)
                # Add the agent to the node
                self.grid.place_agent(a, node)
                self.agent_map[node] = a
            elif node < self.num_recyclers + self.num_consumers:
                b = Recyclers(node, self, self.original_recycling_cost,
                              self.recycling_costs_df,
                              init_eol_rate,
                              recycling_learning_shape_factor)
                self.grid.place_agent(b, node)
                self.agent_map[node] = b
            elif node < self.num_prod_n_recyc + self.num_consumers:
                c = Producers(node, self, scd_mat_prices, virgin_mat_prices)
                self.grid.place_agent(c, node)
                self.agent_map[node] = c
            elif node < self.num_prod_n_recyc + self.num_consumers + \
                    self.num_refurbishers:
                d = Refurbishers(node, self, original_repairing_cost,
                                 init_eol_rate,
                                 repairing_learning_shape_factor,
                                 scndhand_mkt_pric_rate, refurbisher_margin,
                                 max_storage)
                self.grid.place_agent(d, node)
                self.agent_map[node] = d
            else:
                e = Regulators(node, self)
                self.grid.place_agent(e, node)
                self.agent_map[node] = e
        # Draw initial graph
        nx.draw(self.G, with_labels=True)
        # plt.show()

        # Defines reporters and set up data collector
        ABM_CE_PV_model_reporters = {
            **self.get_temporal_data(),
            "Agents repairing": lambda c: self.count_EoL("repairing"),
            "Agents selling": lambda c: self.count_EoL("selling"),
            "Agents recycling": lambda c: self.count_EoL("recycling"),
            "Agents landfilling": lambda c: self.count_EoL("landfilling"),
            "Agents storing": lambda c: self.count_EoL("hoarding"),
            "Agents buying new": lambda c: self.count_EoL("buy_new"),
            "Agents buying used": lambda c: self.count_EoL("buy_used"),
            "Agents buying certified": lambda c: self.count_EoL("certified"),
            "Waste (kg) by pca": lambda c: str(self.pca_outputs),
            "Waste (kg) refurbishers": lambda c: str(
                self.refurbisher_outputs_kg),
            "Tot waste (W) by pca": lambda c: str(self.pca_tot_waste_w),
            "Tot waste (m2) by pca": lambda c: str(self.pca_tot_waste_m2),
            "Tot install (W) by pca": lambda c: str(self.pca_install),
            "Tot install (W) by pca TEST": lambda c: str(
                self.pca_install_test)}
        
        ABM_CE_PV_agenttype_reporters = {
            Consumers: {
                **self.get_temporal_data(),
                "PCA": lambda a: getattr(a, "pca", None),
                "State": lambda a: getattr(a, "state", None),
                "Name": lambda a: report_output_consumer(a, "name"),
                "Latitude": lambda a: report_output_consumer(a, "latitude"),
                "Longitude": lambda a: report_output_consumer(a, "longitude"),
                "Waste Repair (Kg)": lambda a: report_output_consumer(a, "repair_kg"),
                "Waste Sell (Kg)": lambda a: report_output_consumer(a, "sell_kg"),
                "Waste Recycle (Kg)": lambda a: report_output_consumer(a, "recycle_kg"),
                "Waste Landfill (Kg)": lambda a: report_output_consumer(a, "landfill_kg"),
                "Waste Hoard (Kg)": lambda a: report_output_consumer(a, "hoard_kg"),
                "Total Waste (W)": lambda a: report_output_consumer(a, "total_waste_W"),
                "Total Waste (m2)": lambda a: report_output_consumer(a, "total_waste_m2"),
                "Installed Capacity (W)": lambda a: report_output_consumer(a, "total_installed_capacity_W")
            }
        }

        self.datacollector = DataCollector(
            model_reporters=ABM_CE_PV_model_reporters,
            agenttype_reporters=ABM_CE_PV_agenttype_reporters
        )

    # ## New edits
    def pv_ice_waste_calculation(self, clock, pv_ice_outputs):
        self.clock = clock
        mat_Total_EOL_Landfilled = pv_ice_outputs.at[self.clock, 'mat_Total_EOL_Landfilled']
        mat_EOL_Recycled_HQ_into_MFG = pv_ice_outputs.at[self.clock, 'mat_EOL_Recycled_HQ_into_MFG']
        mat_recycled_yield = pv_ice_outputs.at[self.clock, 'mat_recycled_yield']
        mat_recycled_all = pv_ice_outputs.at[self.clock, 'mat_recycled_all']
        mat_reMFG_2_recycle = pv_ice_outputs.at[self.clock, 'mat_reMFG_2_recycle']
        mat_reMFG = pv_ice_outputs.at[self.clock, 'mat_reMFG']
        mat_PG2_stored = pv_ice_outputs.at[self.clock, 'mat_PG2_stored']

        self.pv_ice_yearly_waste = (
            mat_Total_EOL_Landfilled +
            mat_EOL_Recycled_HQ_into_MFG +
            mat_recycled_yield +
            mat_recycled_all +
            mat_reMFG_2_recycle +
            mat_reMFG +
            mat_PG2_stored
        )
        # print("\n\ntotal:", self.pv_ice_yearly_waste )

    def create_agent_pca_map(self, num_consumers):
        pca_column = self.data['PCA']
        unique_pca = pca_column.unique()
        total_unique_pca = len(unique_pca)

        # Calculate the number of agents per unique PCA value
        agents_per_pca = num_consumers // total_unique_pca

        # Distribute agents evenly to each unique PCA value
        agents_count_per_pca = [agents_per_pca] * total_unique_pca

        # Distribute remaining agents if any
        remaining_agents = num_consumers % total_unique_pca
        for i in range(remaining_agents):
            agents_count_per_pca[i] += 1

        agents = {}
        agent_id = 0
        for i, pca_value in enumerate(unique_pca):
            state_values = self.data.loc[self.data['PCA'] == pca_value,
                                         'State']
            agents_count = agents_count_per_pca[i]

            for j in range(agents_count):
                # agent_id = self.unique_id()
                state = state_values.sample().iloc[0]
                agents[agent_id] = (pca_value, state, agents_count)
                agent_id += 1

        return agents
    
    def create_agent_site_map(self):
        """
        Create a mapping of agent ids to their respective site names, PCA, and state.
        """
        case_ids = self.uspvdb['case_id'].unique()
        agents = {}
        agent_id = 0

        for i, case_id in enumerate(case_ids):
            site_name = self.uspvdb.loc[self.uspvdb['case_id'] == case_id,
                                        'p_name'].iloc[0]
            state_value = self.uspvdb.loc[self.uspvdb['case_id'] == case_id,
                                           'p_state'].iloc[0]
            pca_value = self.uspvdb.loc[self.uspvdb['p_name'] == site_name,
                                        'PCA'].iloc[0]
            agents[agent_id] = (case_id, site_name, pca_value, state_value)
            agent_id += 1

        return agents
    
    def create_regulator_state_map(self):
        """
        Create a mapping of regulator agent ids to their respective states.
        """
        unique_states = self.data['State'].unique().tolist()
        regulator_state_map = {}
        agent_id = self.num_prod_n_recyc + self.num_consumers + \
            self.num_refurbishers
        for state in unique_states:
            regulator_state_map[agent_id] = state
            agent_id += 1
        return regulator_state_map

    def shortest_paths(self, target_states, distances_to_target):
        """
        Compute shortest paths between chosen origin states and targets with
        the Dijkstra algorithm.
        """
        for i in self.states_graph.nodes:
            shortest_paths = []
            for j in target_states:
                shortest_paths.append(
                    nx.shortest_path_length(self.states_graph, source=i,
                                            target=j, weight='weight',
                                            method='dijkstra'))
            shortest_paths_closest_target = min(shortest_paths)
            if shortest_paths_closest_target == 0:
                shortest_paths_closest_target = self.mean_distance_within_state
            distances_to_target.append(shortest_paths_closest_target)
        return distances_to_target

    def init_network(self, network, nodes, node_degree, rewiring_prob):
        """
        Set up model's industrial symbiosis (IS) and consumers networks.
        """
        if network == "small-world":
            return nx.watts_strogatz_graph(nodes, node_degree, rewiring_prob,
                                           seed=random.seed(self.seed))
        elif network == "complete graph":
            return nx.complete_graph(nodes)
        if network == "random":
            return nx.watts_strogatz_graph(nodes, node_degree, 1)
        elif network == "cycle graph":
            return nx.cycle_graph(nodes)
        elif network == "scale-free graph":
            return nx.powerlaw_cluster_graph(nodes, node_degree, 0.1)
        else:
            return nx.watts_strogatz_graph(nodes, node_degree, rewiring_prob)

    def update_dynamic_lifetime(self):
        if self.dynamic_lifetime_model["Dynamic lifetime"]:
            self.d_product_lifetimes = [
                self.dynamic_lifetime_model["d_lifetime_intercept"] +
                self.dynamic_lifetime_model["d_lifetime_reg_coeff"] *
                x for x in range(len(self.total_number_product) + self.clock
                                 + 1)]
        elif self.dynamic_lifetime_model["Seed"]:
            self.d_product_lifetimes = \
                [self.product_lifetime] * \
                (len(self.total_number_product) + self.clock + 1)
            if self.clock >= self.dynamic_lifetime_model["Year"]:
                for i in range(1, self.clock + 2 -
                               self.dynamic_lifetime_model["Year"]):
                    self.d_product_lifetimes[-i] = \
                        self.dynamic_lifetime_model["avg_lifetime"]
        else:
            self.d_product_lifetimes = \
                [self.product_lifetime] * \
                (len(self.total_number_product) + self.clock + 1)

    def waste_generation(self, avg_lifetime, failure_rate, num_product):
        """
        Generate waste, called by consumers and recyclers/refurbishers
        (to get original recycling/repairing amounts).
        """
        correction_year = len(self.total_number_product) - 1
        return [j * (1 - e**(-(((self.clock + (correction_year - z)) /
                               avg_lifetime[z])**failure_rate))).real
                for (z, j) in enumerate(num_product)]

    def recycling_process_change(self):
        """
        Compute changes to recycling parameters according to the
        techno-economic analysis of the FRELP, ASU and hybrid recycling
        processes from Heath et al. unpublished techno-economic analysis.
        """
        if self.recycling_process["frelp"]:
            self.recovery_fractions = {
                "Product": np.nan, "Aluminum": 0.994, "Glass": 0.98,
                "Copper": 0.97, "Insulated cable": 1., "Silicon": 0.97,
                "Silver": 0.94}
            self.original_recycling_cost = [0.068, 0.068, 0.068]
        elif self.recycling_process["asu"]:
            self.recovery_fractions = {
                "Product": np.nan, "Aluminum": 0.94, "Glass": 0.99,
                "Copper": 0.83, "Insulated cable": 1., "Silicon": 0.90,
                "Silver": 0.74}
            self.original_recycling_cost = [0.153, 0.153, 0.153]
        elif self.recycling_process["hybrid"]:
            self.recovery_fractions = {
                "Product": np.nan, "Aluminum": 0.994, "Glass": 0.98,
                "Copper": 0.83, "Insulated cable": 1., "Silicon": 0.97,
                "Silver": 0.74}
            self.original_recycling_cost = [0.055, 0.055, 0.055]

    def average_mass_per_function_model(self, product_as_function):
        """
        Compute the weighted average mass of the product's waste volume (in
        fu). The weights are the amount of waste for each year. The weighted
        average mass is returned each time step of the simulation.
        """
        
        # len_product_as_function = len(product_as_function)
        # pvice_mat_factor_copy = self.pvice_mat_factor[
        #     self.pvice_mat_factor['year'] <= 2020 + self.clock]
        # conversion_factors = \
        #     pvice_mat_factor_copy['total_massperm2'].to_list()
        # conversion_factors = conversion_factors[-len_product_as_function:]
        # all_data_out_pca = pd.read_csv(
        #     "all_pca_dataOut_95-by-35.Adv.csv")
        # all_data_out_pca = all_data_out_pca.groupby(
        #     'year', as_index=False).mean()
        # data_out_pca_copy = all_data_out_pca[
        #     all_data_out_pca['year'] <= 2020 + self.clock]
        # waste_in_w_list = \
        #     data_out_pca_copy['Yearly_Sum_Power_atEOL'].to_list()
        # waste_in_m2_list = \
        #     data_out_pca_copy['Yearly_Sum_Area_atEOL'].to_list()
        # waste_in_w_list = waste_in_w_list[-len_product_as_function:]
        # waste_in_m2_list = waste_in_m2_list[-len_product_as_function:]
        # waste_w_m2_list = [x / y if y != 0 else 0 for x, y in
        #                    zip(waste_in_m2_list, waste_in_w_list)]
        # product_percent = [x / sum(product_as_function) if
        #                    sum(product_as_function) != 0 else 0 for x in
        #                    product_as_function]
        # product_as_mass_percent = [x * y * z for x, y, z in zip(
        #     product_percent, conversion_factors, waste_w_m2_list)]
        # self.yearly_product_wght = conversion_factors[-1]
        # weighted_average_mass_watt = sum(product_as_mass_percent)
        # # print(weighted_average_mass_watt)

        # ! Much simpler version than commented code above:
        # ! Assuming average lifetime of 25 years so waste at t=0 (2020) is
        # ! weighting (per watt) the factor of the file first entry (iloc[0])
        # ! which is 1995 - correct_mat_factor was calculated by hand from
        # ! PV ICE outputs
        weighted_average_mass_watt = float(self.correct_mat_factor[
            'total_massperW'].iloc[self.clock])
        # print(weighted_average_mass_watt)

        return weighted_average_mass_watt

    def average_price_per_function_model(self):
        """
        Compute the price of first hand products. Price ratio is compared to
        modules of the same year.
        """
        correction_year = len(self.total_number_product) // self.timestep.value
        year = self.clock // self.timestep.value
        self.fsthand_mkt_pric = self.fsthand_mkt_pric_reg_param[0] * e**(
                -self.fsthand_mkt_pric_reg_param[1] * (year+
                                                       correction_year))

    def count_EoL(model, condition):
        """
        Count adoption in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        for agent in model.agents:
            if agent.unique_id < model.num_consumers:
                if condition == "repairing" and agent.EoL_pathway == "repair":
                    count += 1
                elif condition == "selling" and agent.EoL_pathway == "sell":
                    count += 1
                elif condition == "recycling" and \
                        agent.EoL_pathway == "recycle":
                    count += 1
                elif condition == "landfilling" and \
                        agent.EoL_pathway == "landfill":
                    count += 1
                elif condition == "hoarding" and agent.EoL_pathway == "hoard":
                    count += 1
                elif condition == "buy_new" and \
                        agent.purchase_choice == "new":
                    count += 1
                elif condition == "buy_used" and \
                        agent.purchase_choice == "used":
                    count += 1
                    model.consumer_used_product += 1
                elif condition == "buy_certified" and \
                        agent.purchase_choice == "certified":
                    count += 1
                else:
                    continue
            else:
                continue
        return count

    def report_output(model, condition):
        """
        Count waste streams in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        count2 = 0
        industrial_waste_landfill = 0
        industrial_waste_recycled = 0
        industrial_waste_landfill_mass = 0
        industrial_waste_recycled_mass = 0
        for agent in model.agents:
            if model.num_consumers + model.num_recyclers <= agent.unique_id < \
                    model.num_consumers + model.num_prod_n_recyc:
                if model.epr_business_model:
                    industrial_waste_recycled += \
                        agent.industrial_waste_generated / model.num_consumers
                    industrial_waste_recycled_mass += \
                        model.yearly_product_wght * \
                        agent.industrial_waste_generated / model.num_consumers
                else:
                    industrial_waste_landfill += \
                        agent.industrial_waste_generated / model.num_consumers
                    industrial_waste_landfill_mass += \
                        model.yearly_product_wght * \
                        agent.industrial_waste_generated / model.num_consumers
        for agent in model.agents:
            if condition == "product_stock" and agent.unique_id < \
                    model.num_consumers:
                count += sum(agent.number_product_hard_copy)
            elif condition == "product_stock_new" and agent.unique_id < \
                    model.num_consumers:
                count += sum(agent.new_products_hard_copy)
            if condition == "product_stock_used" and agent.unique_id < \
                    model.num_consumers:
                count += sum(agent.used_products_hard_copy)
            elif condition == "prod_stock_new_mass" and agent.unique_id < \
                    model.num_consumers:
                count += agent.new_products_mass
            if condition == "prod_stock_used_mass" and agent.unique_id < \
                    model.num_consumers:
                count += agent.used_products_mass
            elif condition == "product_repaired" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_repaired
            elif condition == "product_sold" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_sold
                count2 += agent.number_product_sold
                count2 += agent.number_product_repaired
            elif condition == "product_recycled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_recycled
                count += industrial_waste_recycled
            elif condition == "product_landfilled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_landfilled
                count += industrial_waste_landfill
            elif condition == "product_hoarded" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_hoarded
            elif condition == "product_new_repaired" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_new_prod_repaired
            elif condition == "product_new_sold" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_new_prod_sold
            elif condition == "product_new_recycled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_new_prod_recycled
                count += industrial_waste_recycled_mass
            elif condition == "product_new_landfilled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_new_prod_landfilled
                count += industrial_waste_landfill_mass
            elif condition == "product_new_hoarded" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_new_prod_hoarded
            elif condition == "product_used_repaired" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_used_prod_repaired
            elif condition == "product_used_sold" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_used_prod_sold
            elif condition == "product_used_recycled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_used_prod_recycled
            elif condition == "product_used_landfilled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_used_prod_landfilled
            elif condition == "product_used_hoarded" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_used_prod_hoarded
            elif condition == "consumer_costs" and agent.unique_id < \
                    model.num_consumers:
                count += agent.consumer_costs
            elif condition == "average_landfill_cost" and agent.unique_id < \
                    model.num_consumers:
                count += agent.landfill_cost / model.num_consumers
            elif condition == "average_hoarding_cost" and agent.unique_id < \
                    model.num_consumers:
                count += agent.hoarding_cost / model.num_consumers
            elif condition == "average_recycling_cost" and model.num_consumers\
                    <= agent.unique_id < model.num_consumers + \
                    model.num_recyclers:
                count += agent.recycling_cost / model.num_recyclers
            elif condition == "average_repairing_cost" and \
                model.num_consumers + model.num_prod_n_recyc <= \
                    agent.unique_id < model.num_consumers + \
                        model.num_prod_n_recyc + model.num_refurbishers:
                count += agent.repairing_cost / model.num_refurbishers
            elif condition == "average_second_hand_price" and \
                    model.num_consumers + model.num_prod_n_recyc <= \
                        agent.unique_id < model.num_consumers + \
                            model.num_prod_n_recyc + model.num_refurbishers:
                count += (-1 * agent.scd_hand_price) / model.num_refurbishers
            elif condition == "year":
                count = model.current_date.year
            elif condition == "month":
                count = model.current_date.month
            elif condition == "quarter":
                count = (model.current_date.month - 1) // 3 + 1
            elif condition == "weight":
                count = model.dynamic_product_average_wght
            elif condition == "recycled_mat_volume" and model.num_consumers + \
                    model.num_recyclers <= agent.unique_id < \
                    model.num_consumers + model.num_prod_n_recyc:
                if not np.isnan(agent.recycled_material_volume):
                    count += agent.recycled_material_volume
            elif condition == "recycled_mat_value" and model.num_consumers + \
                    model.num_recyclers <= agent.unique_id < \
                    model.num_consumers + model.num_prod_n_recyc:
                if not np.isnan(agent.recycled_material_value):
                    count += agent.recycled_material_value
            elif condition == "producer_costs" and model.num_consumers + \
                    model.num_recyclers <= agent.unique_id < \
                    model.num_consumers + model.num_prod_n_recyc:
                count += agent.producer_costs
            elif condition == "recycler_costs" and model.num_consumers <= \
                    agent.unique_id < model.num_consumers + \
                    model.num_recyclers:
                count += agent.recycler_costs
            elif condition == "refurbisher_costs" and model.num_consumers + \
                    model.num_prod_n_recyc <= agent.unique_id \
                    < model.num_consumers + model.num_prod_n_recyc + \
                    model.num_refurbishers:
                count += agent.refurbisher_costs
            elif condition == "refurbisher_costs_w_margins" and \
                model.num_consumers + model.num_prod_n_recyc \
                    <= agent.unique_id < model.num_consumers + \
                    model.num_prod_n_recyc + model.num_refurbishers:
                count += agent.refurbisher_costs_w_margins
        if condition == "product_sold":
            model.sold_repaired_waste += count2 - \
                                         model.past_sold_repaired_waste
            model.past_sold_repaired_waste = count2
        return count
    
    @property
    def current_date(self):
        """
        Returns the current date based on the model's clock and timestep
        """
        if self.timestep == TIMESTEP.ANNUAL:
            return datetime(year=2020 + self.clock, month=1, day=1)
        elif self.timestep == TIMESTEP.MONTHLY:
            return datetime(year=2020 + (self.clock // 12), month=(self.clock % 12) + 1, day=1)
        elif self.timestep == TIMESTEP.QUARTERLY:
            return datetime(year=2020 + (self.clock // 4), month=((self.clock % 4) * 3) + 1, day=1)
        else:
            raise ValueError("Unsupported timestep. Use ANNUAL, MONTHLY, or QUARTERLY.")
        
    def get_temporal_data(self):
        """
        Returns the current date in a format suitable for temporal data
        collection.
        """
        temporal_data = {
            "Year": lambda c: self.report_output("year"),
        }

        if self.timestep == TIMESTEP.MONTHLY:
            temporal_data["Month"] = lambda c: self.report_output("month")
        elif self.timestep == TIMESTEP.QUARTERLY:
            temporal_data["Quarter"] = lambda c: self.report_output("quarter")
        return temporal_data

    def pv_ice_mat_factor(self):
        """
        Defines the yearly weight (kg/m2) of pv panels. Also Compute the
        average mass of the panels for the last "stored" years.
        """
        pv_ice_mat_subset = self.pvice_mat_factor[
            self.pvice_mat_factor['date'] == self.current_date]
        self.weight_factor = pv_ice_mat_subset['total_massperm2'].iloc[0]
        past_storage = max(0, (self.current_date.year - self.max_storage[1]))
        pv_ice_mat_subset_stored_years = self.pvice_mat_factor[
            (self.pvice_mat_factor['year'] >= past_storage) & 
            (self.pvice_mat_factor['date'] < self.current_date)]
        self.avg_weight_factor_stored_pv = pv_ice_mat_subset_stored_years[
            'total_massperm2'].mean()
    def get_transportation_cost(self, hazardous: bool = False):
        """
        Returns the transportation cost based on the current date and
        the rtn flag.
        If the RTN model costs are enabled, it checks if the current year
        is within the range of the recycling costs DataFrame. If it is,
        it returns 0, since the transportation costs are accounted for
        in the recycling costs. Otherwise, it returns the transportation cost.
        If the RTN model costs are not enabled, it returns the transportation cost.
        """
        if self.rtn:
            min_year = self.recycling_costs_df['Year'].min()
            max_year = self.recycling_costs_df['Year'].max()
            if self.current_date.year >= min_year and \
                    self.current_date.year <= max_year:
                return 0
        if hazardous:
            return self.hazardous_transportation_cost
        return self.transportation_cost
    
    def filter_landfills_accepting_solar_waste(self):
        """
        Filters the landfills that accept solar waste based on the
        landfill_solar_waste_acceptance_ratio.
        """
        all_site_indices = range(len(self.landfill_distance_df))
        valid_site_indices = random.sample(all_site_indices, int(len(all_site_indices) * self.landfill_solar_waste_acceptance_ratio))
        self.landfill_distance_df = self.landfill_distance_df.iloc[valid_site_indices].reset_index(drop=True)

    def tclp_test(self, start_year: int = 2020) -> bool:
        """Age-varying (logistic) mean & std TCLP hazard classification.

        compute an age-dependent mean and standard deviation
        by smoothly interpolating between "fresh" and "aged"
        parameters using a logistic weight w(age). Then,
        sample a latent variable from a Weibull distribution
        parameterized to match the interpolated mean and std.
        Return True if the sampled latent value exceeds
        the hazard cutoff.

        This preserves gradual broadening of variance with age while avoiding
        discrete mixture sampling.

        Parameters
        ----------
        start_year : int
            Year the module was installed.

        Returns
        -------
        bool
            True if sampled latent value > hazard_cutoff.
        """

        module_age_years = self.current_date.year - start_year

        fresh_mu = self.tclp_params["fresh_mean"]
        fresh_sd = self.tclp_params["fresh_std"]
        aged_mu  = self.tclp_params["aged_mean"]
        aged_sd  = self.tclp_params["aged_std"]
        k        = self.tclp_params["k"]
        a50      = self.tclp_params["a50"]
        cutoff   = self.tclp_params["hazard_cutoff"]
        min_std  = self.tclp_params.get("min_std", 0.0)

        # Logistic weight w(age) in [0,1]
        w = 1.0 / (1.0 + e ** (-k * (module_age_years - a50)))

        # Interpolated mean & std
        mu_age = (1 - w) * fresh_mu + w * aged_mu
        sd_age = (1 - w) * fresh_sd + w * aged_sd
        sd_age = max(sd_age, min_std)

        if self.tclp_params["distribution"] == "normal":
            # Normal distribution sampling
            latent = np.random.normal(mu_age, sd_age)
            
        elif self.tclp_params["distribution"] == "weibull":
            # Weibull shape parameters estimated from mean & std
            weibull_shape = (sd_age / mu_age) ** -1.086
            weibull_scale = mu_age / gamma(1 + 1 / weibull_shape)
            latent = np.random.weibull(weibull_shape) * weibull_scale
        return latent > cutoff

    def get_num_consumers(self, target_num_consumers: int) -> int:
        """
        Calculate the number of consumer agents based on the agent resolution.
        """

        if self.consumer_agent_resolution == ConsumerAgentResolution.PCA:
            return target_num_consumers
        elif self.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            return self.uspvdb.shape[0]
        
        raise ValueError("Invalid agent resolution specified.")

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        self.total_waste = 0
        self.total_yearly_new_products = 0
        self.consumer_used_product = 0
        self.yearly_repaired_waste = 0
        self.dynamic_product_average_wght = \
            self.average_mass_per_function_model(
                self.copy_total_number_product)
        self.pv_ice_mat_factor()
        # Collect data
        self.datacollector.collect(self)
        # Refers to agent step function
        self.update_dynamic_lifetime()
        self.average_price_per_function_model()
        self.agents.do("step")
        self.clock = self.clock + 1
        if self.clock == self.last_step:
            self.datacollector.collect(self)

        # Calculate yearly waste using pv_ice_waste_calculation method
        # pass pv_output
        self.pv_ice_yearly_waste = 0
