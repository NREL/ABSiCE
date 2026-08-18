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
import yaml
from geopy.geocoders import Nominatim
import time
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from utils import TIMESTEP, ConsumerAgentResolution, PCA_MISSING_VALUE, transform_timeseries_timestep, transform_pca_timeseries_timestep, add_date_from_temporal_columns
from datetime import datetime

from absice.data.data_loader import LoadedData
from absice.schemas.simulation_config import SimulationConfig


class ABM_CE_PV(Model):
    def __init__(self, config: SimulationConfig, data: LoadedData) -> None:
        """
        Initialize the ABSiCE model.

        Parameters
        ----------
        config
            Validated simulation parameters loaded from YAML.
        data
            Preloaded datasets created by DataLoader.
        """
        super().__init__(seed=config.run.seed)

        self.config = config
        self.loaded_data = data

        # Run configuration
        seed = config.run.seed
        timestep = config.run.timestep
        last_step = config.run.last_step

        # Calibration configuration
        calibration_n_sensitivity = 1.0
        calibration_n_sensitivity_2 = 1.0
        calibration_n_sensitivity_3 = 1.0
        calibration_n_sensitivity_4 = 1.0
        calibration_n_sensitivity_5 = 1.0

        # Network configuration
        num_consumers = config.network.num_consumers
        consumers_node_degree = config.network.consumers_node_degree
        consumers_network_type = config.network.consumers_network_type
        rewiring_prob = config.network.rewiring_prob
        num_recyclers = config.network.num_recyclers
        num_producers = config.network.num_producers
        num_refurbishers = config.network.num_refurbishers
        prod_n_recyc_node_degree = config.network.prod_n_recyc_node_degree
        prod_n_recyc_network_type = config.network.prod_n_recyc_network_type

        # Consumer configuration
        consumer_agent_resolution = config.consumer.resolution
        model_states = config.consumer.model_states
        # consumers_distribution / product_distribution are read directly by
        # Consumers from self.model.config.consumer.*.

        # Product configuration
        total_number_product = config.product.total_number_product
        product_growth = config.product.product_growth
        growth_threshold = config.product.growth_threshold
        failure_rate_alpha = config.product.failure_rate_alpha.copy()
        product_lifetime = config.product.product_lifetime
        product_average_wght = config.product.product_average_wght
        mass_to_function_reg_coeff = config.product.mass_to_function_reg_coeff
        max_storage = config.product.max_storage.copy()

        # End-of-life configuration
        init_eol_rate = config.eol.init_eol_rate.copy()
        init_purchase_choice = config.eol.init_purchase_choice.copy()
        all_EoL_pathways = config.eol.all_eol_pathways.copy()
        purchase_choices = config.eol.purchase_choices.copy()

        # Theory of Planned Behavior configuration
        theory_of_planned_behavior = config.tpb.theory_of_planned_behavior
        # w_sn_eol/w_pbc_eol/w_a_eol/w_sn_reuse/w_pbc_reuse/w_a_reuse and
        # att_distrib_param_eol/att_distrib_param_reuse are read directly by
        # Consumers from self.model.config.tpb.*.
        extended_tpb = config.tpb.extended_tpb.model_dump(by_alias=True)

        # Cost configuration
        landfill_cost = config.cost.landfill_cost.copy()
        hazardous_waste_management_cost = config.cost.hazardous_waste_management_cost.copy()
        original_recycling_cost = config.cost.original_recycling_cost.copy()
        repairability = config.cost.repairability
        original_repairing_cost = config.cost.original_repairing_cost.copy()
        fsthand_mkt_pric = config.cost.fsthand_mkt_pric
        fsthand_mkt_pric_reg_param = config.cost.fsthand_mkt_pric_reg_param
        transportation_cost = config.cost.transportation_cost
        hazardous_transportation_cost = config.cost.hazardous_transportation_cost
        # hoarding_cost, used_product_substitution_rate,
        # recycling_learning_shape_factor, repairing_learning_shape_factor,
        # scndhand_mkt_pric_rate, and refurbisher_margin are read directly by
        # the relevant agents from self.model.config.cost.*.
        imperfect_substitution = config.cost.imperfect_substitution
        sa_landfill_costs = config.cost.sa_landfill_costs

        # Material configuration
        product_mass_fractions = config.material.product_mass_fractions.copy()
        established_scd_mkt = config.material.established_scd_mkt.copy()
        scd_mat_prices = config.material.scd_mat_prices.copy()
        virgin_mat_prices = config.material.virgin_mat_prices.copy()
        material_waste_ratio = config.material.material_waste_ratio.copy()
        recovery_fractions = config.material.recovery_fractions.copy()

        # Scenario configuration
        recycling_states = config.scenario.recycling_states.copy()
        epr_business_model = config.scenario.epr_business_model
        recycling_process = config.scenario.recycling_process.model_dump()
        dynamic_lifetime_model = config.scenario.dynamic_lifetime_model.model_dump(by_alias=True)
        seeding = config.scenario.seeding.model_dump(by_alias=True)
        seeding_recyc = config.scenario.seeding_recyc.model_dump(by_alias=True)
        hazardous_waste_regulation_enabled = config.scenario.hazardous_waste_regulation_enabled
        landfill_solar_waste_acceptance_ratio = config.scenario.landfill_solar_waste_acceptance_ratio

        # TCLP configuration
        tclp_params = config.tclp.model_dump()

        # Data-source configuration
        pv_ice = config.data_source.pv_ice
        pca = config.data_source.pca
        pca_scenario = config.data_source.pca_scenario
        geopy = config.data_source.geopy
        calculate_distances = config.data_source.calculate_distances
        rtn = config.data_source.rtn
        solar_cycle = config.data_source.solar_cycle

        # Temporary legacy settings
        landfill_data_params = config.legacy_data.landfill_data_params.model_dump()
        file_name = config.legacy_data.file_name.model_dump(by_alias=True)

        self.seed = seed
        # Seed every RNG the model relies on. Mesa's super().__init__(seed=...)
        # only seeds self.random; agent logic also uses the global random module
        # and numpy, so seed those too for reproducible per-run results.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.timestep = timestep
        self.rtn = rtn
        self.solar_cycle = solar_cycle
        self.landfill_data_params = landfill_data_params
        self.model_states = model_states

        #Set path for data saving
        testfolder = str(Path().resolve() / 'PV_ICE' / 'TEMP' / 'PCA')

        if not os.path.exists(testfolder):
            os.makedirs(testfolder)

        SupportingMaterialFolder = str(Path().resolve()/ 'PV_ICE' / 'baselines' / 'SupportingMaterial')
        BaselinesFolder = str(Path().resolve()/ 'PV_ICE' / 'baselines')

        rawdf = data.reeds_raw.copy()

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
            if data.uspvdb is None:
                raise ValueError(
                    "USPVDB data is required for site-level resolution."
                )
            if data.reeds_data is None:
                raise ValueError(
                    "ReEDS data is required for site-level resolution."
                )

            self.uspvdb = data.uspvdb.copy()
            self.reeds_data = data.reeds_data.copy()
            self.uspvdb = self.uspvdb[
                self.uspvdb["PCA"] != PCA_MISSING_VALUE
            ].copy()
            # PV ICE waste data is the authoritative waste source for this
            # simulation. Restrict sites to PCAs that have a waste time series
            # (e.g. drop p119, p122): sites without waste data would create
            # agents with no waste to process and fail downstream date-based
            # waste lookups on multi-step runs.
            _waste_pcas = set(data.pvice_waste_eol_df["pca"].unique())
            self.uspvdb = self.uspvdb[
                self.uspvdb["PCA"].isin(_waste_pcas)
            ].copy()

        GIS = data.gis_centroids.copy()
    
        # 1. Create ReEDS Scenarios BASELINE Files

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


        pca_longlat = pd.DataFrame(columns=["PCA", "Long", "Lat"])

        # Create the 3 Scenarios and assign Baselines

        self.recycler_data = data.recycler_data.copy()
        self.universal_waste_recyclers_data = data.uw_recycler_data.copy()

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
            self.pca_data = GIS.reset_index().copy()
            index_column = self.pca_data.columns[0]
            self.pca_data = self.pca_data[
                [index_column, "long", "lat"]
            ].rename(
                columns={
                    index_column: "PCA",
                    "long": "Long",
                    "lat": "Lat",
                }
            )

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

                # Calculate distances for Universal Waste Landfills (PCA resolution)
                universal_waste_landfills_data = pd.read_csv("../../../TEMP/" + self.file_names['Universal Waste Landfills data'])
                uw_landfill_distance_df = pd.DataFrame(columns=self.pca_data['PCA'],
                                            index=universal_waste_landfills_data['Facility Name'])
                
                for pca_index, pca_row in self.pca_data.iterrows():
                    pca_lat = pca_row['Lat']
                    pca_lon = pca_row['Long']
                    
                    for uw_landfill_index, uw_landfill_row in universal_waste_landfills_data.iterrows():
                        uw_landfill_lat = uw_landfill_row['Latitude']
                        uw_landfill_lon = uw_landfill_row['Longitude']
                        
                        distance_uw_landfill = haversine(pca_lat, pca_lon,
                                            uw_landfill_lat, uw_landfill_lon)
                        
                        uw_landfill_distance_df.at[uw_landfill_row['Facility Name'],
                                        pca_row['PCA']] = distance_uw_landfill
                
                uw_landfill_distance_df.to_csv("../../../TEMP/universal_waste_pca_landfill_distances.csv")
                
                # Calculate distances for Universal Waste Recyclers (PCA resolution)
                universal_waste_recyclers_data = pd.read_csv("../../../TEMP/" + self.file_names['Universal Waste Recyclers data'])
                uw_recycler_distance_df = pd.DataFrame(columns=self.pca_data['PCA'],
                                            index=universal_waste_recyclers_data['Recycler Name'])
                
                for pca_index, pca_row in self.pca_data.iterrows():
                    pca_lat = pca_row['Lat']
                    pca_lon = pca_row['Long']
                    
                    for uw_recycler_index, uw_recycler_row in universal_waste_recyclers_data.iterrows():
                        uw_recycler_lat = uw_recycler_row['Latitude']
                        uw_recycler_lon = uw_recycler_row['Longitude']
                        
                        distance_uw_recycler = haversine(pca_lat, pca_lon,
                                            uw_recycler_lat, uw_recycler_lon)
                        
                        uw_recycler_distance_df.at[uw_recycler_row['Recycler Name'],
                                        pca_row['PCA']] = distance_uw_recycler
                
                uw_recycler_distance_df.to_csv("../../../TEMP/universal_waste_pca_recycler_distances.csv")
                
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

                # Calculate distances for Universal Waste Landfills
                universal_waste_landfills_data = pd.read_csv("../../../TEMP/" + self.file_names['Universal Waste Landfills data'])
                uw_landfill_lats = universal_waste_landfills_data['Latitude'].to_numpy().astype(float)
                uw_landfill_lons = universal_waste_landfills_data['Longitude'].to_numpy().astype(float)
                uw_landfill_distance_matrix = haversine_vectorized(
                    site_lats, site_lons, uw_landfill_lats, uw_landfill_lons)
                uw_landfill_distance_df = pd.DataFrame(uw_landfill_distance_matrix,
                                                        index=universal_waste_landfills_data['Facility Name'],
                                                        columns=self.uspvdb['case_id'])
                uw_landfill_distance_df.to_csv(
                    os.path.join(
                        os.path.dirname(__file__), "TEMP", 'universal_waste_site_landfill_distances.csv'))

                # Calculate distances for Universal Waste Recyclers
                universal_waste_recyclers_data = pd.read_csv("../../../TEMP/" + self.file_names['Universal Waste Recyclers data'])
                uw_recycler_lats = universal_waste_recyclers_data['Latitude'].to_numpy().astype(float)
                uw_recycler_lons = universal_waste_recyclers_data['Longitude'].to_numpy().astype(float)
                uw_recycler_distance_matrix = haversine_vectorized(
                    site_lats, site_lons, uw_recycler_lats, uw_recycler_lons)
                uw_recycler_distance_df = pd.DataFrame(uw_recycler_distance_matrix,
                                                        index=universal_waste_recyclers_data['Recycler Name'],
                                                        columns=self.uspvdb['case_id'])
                uw_recycler_distance_df.to_csv(
                    os.path.join(
                        os.path.dirname(__file__), "TEMP", 'universal_waste_site_recycler_distances.csv'))

        self.correct_mat_factor = data.correct_mat_factor.copy()
        self.hazardous_landfill_cost_df = (
            data.hazardous_landfill_cost_df.copy()
        )
        self.universal_waste_landfills_data = data.uw_landfill_data.copy()
        self.universal_waste_landfill_cost_df = (
            self.universal_waste_landfills_data.copy()
        )

        self.correct_mat_factor = transform_timeseries_timestep(
            self.correct_mat_factor, self.timestep, scale=False
        )

        # ReEDS table used by existing model methods and agents.
        self.data = rawdf.reset_index().copy()
        if self.model_states is not None:
            print(
                "Model will run for the following states:",
                self.model_states,
            )
            self.data = self.data[
                self.data["State"].isin(self.model_states)
            ].copy()

        # Distance matrices are preloaded by DataLoader.
        self.recycler_distance_df = data.recycler_distance_df.copy()
        self.landfill_distance_df = data.landfill_distance_df.copy()
        self.hazardous_landfill_distance_df = (
            data.hazardous_landfill_distance_df.copy()
        )
        self.universal_waste_landfill_distance_df = (
            data.uw_landfill_distance_df.copy()
        )
        self.universal_waste_recycler_distance_df = (
            data.uw_recycler_distance_df.copy()
        )

        if self.rtn:
            # RTN cost data has not yet been added to LoadedData.
            self.recycling_costs_df = pd.read_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "RTN",
                    self.file_names["Recycling data"],
                )
            )
            self.recycling_costs_df = add_date_from_temporal_columns(
                self.recycling_costs_df, self.timestep
            )
        else:
            self.recycling_costs_df = pd.DataFrame()

        if self.solar_cycle:
            # Solar Cycle data remains a temporary legacy input.
            self.landfill_cost_df = pd.read_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "SolarCycle",
                    "wbj_solar_cycle_combined.csv",
                ),
                index_col=0,
            )
        else:
            self.landfill_cost_df = data.landfill_cost_df.copy()

        self.pv_ice_yearly_waste = 0

        self.num_consumers = self.get_num_consumers(num_consumers)
        self.consumers_node_degree = consumers_node_degree
        self.consumers_network_type = consumers_network_type
        if self.consumer_agent_resolution is ConsumerAgentResolution.PCA:
            self.agent_pca_map = self.create_agent_pca_map(self.num_consumers)
        elif self.consumer_agent_resolution is ConsumerAgentResolution.SITE:
            self.agent_site_map = self.create_agent_site_map()
        # Recycler node ids are assigned POSITIONALLY: combined-row-index j
        # across [regular ++ universal-waste] distance frames maps to node
        # num_consumers + j. Recycler names are NOT unique (duplicate names are
        # distinct physical facilities at different locations), so a name->id
        # dict would collapse them and mis-route/lose facilities. Counts use
        # ROW counts, not .unique().
        regular_recycler_names = list(
            self.recycler_distance_df["Recycler Name"].astype(str)
        )
        universal_waste_recycler_names = list(
            self.universal_waste_recycler_distance_df["Recycler Name"].astype(str)
        )

        # Facility name per node offset, in creation/row order (regular then UW).
        self.recycler_facilities: list[str] = (
            regular_recycler_names + universal_waste_recycler_names
        )
        self.num_regular_recyclers = len(regular_recycler_names)
        self.num_universal_waste_recyclers = len(universal_waste_recycler_names)
        self.num_recyclers = len(self.recycler_facilities)

        assert (
            self.num_recyclers
            == self.num_regular_recyclers + self.num_universal_waste_recyclers
        ), "recycler facility count mismatch"

        self.num_producers = num_producers
        self.num_prod_n_recyc = (
            self.num_recyclers
            + self.num_producers
        )
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
        all_pca_df_in = pd.DataFrame()
        all_pca_df_out = pd.DataFrame()
        valid_pcas = PCAs
        # Filter to only include specified model states
        if self.model_states is not None:
            valid_pcas = self.data[self.data['State'].isin(self.model_states)]['PCA'].unique().tolist()
        _pca_merged_dir = os.path.join(
            os.path.dirname(__file__), "PV_ICE", "TEMP", "PCA_merged")
        # Old PV ICE per-PCA dataOut results live in the PCA directory (only the
        # merged datain files are staged in PCA_merged).
        _pca_dataout_dir = os.path.join(
            os.path.dirname(__file__), "PV_ICE", "TEMP", "PCA")
        for pca in valid_pcas:
            # Merged datain file: Solar Futures (2010–2025) + ReEDS StdScen24
            # (2026+); used for installed capacity history (total_number_product).
            subset_df_init_cap = pd.read_csv(
                os.path.join(_pca_merged_dir, "datain_95-by-35.Adv_" + pca + "_.csv"))
            subset_df_init_cap['pca'] = pca
            all_pca_df_in = pd.concat([all_pca_df_in, subset_df_init_cap])
            # NOTE: old PV ICE results — used for product_average_wght baseline
            subset_df_init_cap_out = pd.read_csv(
                os.path.join(
                    _pca_dataout_dir,
                    f"dataOut_95-by-35.Adv_{pca}_.csv",
                )
            )
            subset_df_init_cap_out['pca'] = pca
            all_pca_df_out = pd.concat([all_pca_df_out,
                                        subset_df_init_cap_out])
        # Shared pre-2020 EoL waste baseline used by Recyclers and
        # Refurbishers (original_recycling_volume / original_repairing_volume).
        # Computed here from the in-memory, valid_pcas-filtered all_pca_df_out
        # (NOT from a re-read of the all_pca_dataOut CSV) so that model_states
        # filtering is respected.
        _baseline_yearly_waste = all_pca_df_out[all_pca_df_out['year'] <= 2020]
        self.original_eol_baseline_volume = float(
            sum(_baseline_yearly_waste['Yearly_Sum_Power_atEOL'].tolist()))
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
                    material_df = pd.read_csv(file_path, skiprows=[1])
                    # Add the material mass per m^2 column to the existing DataFrame
                    df_mat_factor["year"] = material_df["year"]
                    try:
                        df_mat_factor[material_name] = (
                            material_df["mat_massperm2"].astype(float) / 1000
                        )
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

        # Load consolidated waste EOL data (metric tons) once at model level.
        # Agents slice this by PCA in their __init__ instead of reading
        # individual per-PCA dataOut files for waste values.
        self.pvice_waste_eol_df = data.pvice_waste_eol_df.copy()
        self.pvice_waste_eol_df = transform_pca_timeseries_timestep(
            self.pvice_waste_eol_df, self.timestep, filtered_columns=['Yearly_Waste_EOL_Ton'])
        self.avg_weight_factor_stored_pv = 0

        if "date" not in self.pvice_waste_eol_df.columns:
            if self.timestep == TIMESTEP.ANNUAL:
                self.pvice_waste_eol_df["date"] = pd.to_datetime(
                    self.pvice_waste_eol_df["year"].astype(int).astype(str),
                    format="%Y",
                )

            elif self.timestep == TIMESTEP.QUARTERLY:
                self.pvice_waste_eol_df["date"] = pd.to_datetime(
                    {
                        "year": self.pvice_waste_eol_df["year"].astype(int),
                        "month": (
                            self.pvice_waste_eol_df.groupby("year").cumcount() * 3
                            + 1
                        ),
                        "day": 1,
                    }
                )

            elif self.timestep == TIMESTEP.MONTHLY:
                self.pvice_waste_eol_df["date"] = pd.to_datetime(
                    {
                        "year": self.pvice_waste_eol_df["year"].astype(int),
                        "month": (
                            self.pvice_waste_eol_df.groupby("year").cumcount()
                            + 1
                        ),
                        "day": 1,
                    }
                )
                
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
        self.pca_tot_waste_ton = {}
        self.pca_tot_waste_m2 = {}  # deprecated: waste now tracked in metric tons
        for pca in valid_pcas:
            pathway_dict = {}
            for pathway in self.all_EoL_pathways.keys():
                pathway_dict[pathway] = 0
            self.pca_outputs[pca] = pathway_dict
            self.pca_install[pca] = 0
            self.pca_tot_waste_ton[pca] = 0
            self.pca_tot_waste_m2[pca] = 0  # deprecated, always 0
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
        # NOTE: old PV ICE results — used only for product_average_wght baseline;
        # waste EOL values come from pvice_waste_eol_df (consolidated metric-ton file)
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
        self.tclp_market_share_df = data.tclp_market_share_df.copy()

        self.all_gba = rawdf.reset_index().copy()

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
        # DC is not included in the PV ICE data but is there in the USPVDB
        # so we add it here
        if self.consumer_agent_resolution == ConsumerAgentResolution.SITE:
            site_states = self.uspvdb['p_state'].unique().tolist()
            unique_states = set(unique_states).union(site_states)
            unique_states = list(unique_states)
        self.num_regulators = len(unique_states)
        self.regulator_state_map = self.create_regulator_state_map(unique_states)
        # Load the policy schedule YAML once and distribute per-state entries to agents
        self.policy_schedule_by_state = dict(data.policy_schedule_by_state)
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

        states_adjacency_df = data.states_adjacency_matrix.copy()
        self.states = states_adjacency_df.to_numpy()
        # Compute distances
        self.mean_distance_within_state = np.nanmean(
            np.where(self.states != 0, self.states, np.nan)
        ) / 2
        self.states_graph = nx.from_numpy_array(self.states)
        nodes_states_dic = dict(
            zip(self.states_graph.nodes, states_adjacency_df.columns)
        )
        self.states_graph = nx.relabel_nodes(
            self.states_graph, nodes_states_dic
        )
        self.recycling_states = recycling_states
        distances_to_recyclers = []
        distances_to_recyclers = self.shortest_paths(
            self.recycling_states, distances_to_recyclers)
        self.mn_mx_av_distance_to_recycler = [
            min(distances_to_recyclers), max(distances_to_recyclers),
            sum(distances_to_recyclers) / len(distances_to_recyclers)]
        # Compute transportation costs
        self.transportation_cost_rcl = [
            x * self.transportation_cost for x in
            self.mn_mx_av_distance_to_recycler]  # $/ton: dist [km] * cost [$/ton/km]

        # ! TODO: change landfill transportation costs
        self.transportation_cost_rpr_ldf = self.mean_distance_within_state * \
            self.transportation_cost  # $/ton: dist [km] * cost [$/ton/km]

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
                a = Consumers(node, self, perceived_behavioral_control)
                # Add the agent to the node
                self.grid.place_agent(a, node)
                self.agent_map[node] = a
            elif node < self.num_recyclers + self.num_consumers:
                b = Recyclers(node, self, self.recycling_costs_df)
                self.grid.place_agent(b, node)
                self.agent_map[node] = b
            elif node < self.num_prod_n_recyc + self.num_consumers:
                c = Producers(node, self, scd_mat_prices, virgin_mat_prices)
                self.grid.place_agent(c, node)
                self.agent_map[node] = c
            elif node < self.num_prod_n_recyc + self.num_consumers + \
                    self.num_refurbishers:
                d = Refurbishers(node, self, original_repairing_cost)
                self.grid.place_agent(d, node)
                self.agent_map[node] = d
            else:
                e = Regulators(node, self,
                               self.policy_schedule_by_state.get(
                                   self.regulator_state_map[node], {}))
                self.grid.place_agent(e, node)
                self.agent_map[node] = e
        # Draw initial graph
        nx.draw(self.G, with_labels=True)
    
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
            "Tot waste (ton) by pca": lambda c: str(self.pca_tot_waste_ton),
            "Tot waste (m2) by pca": lambda c: str(self.pca_tot_waste_m2),  # deprecated, always 0
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
                "Total Waste (ton)": lambda a: report_output_consumer(a, "total_waste_ton"),
                "Total Waste (m2)": lambda a: report_output_consumer(a, "total_waste_m2"),
                "Installed Capacity (W)": lambda a: report_output_consumer(a, "total_installed_capacity_W"),
                "TCLP Test Result": lambda a: report_output_consumer(a, "tclp_test_result"),
            }
        }

        self.datacollector = DataCollector(
            model_reporters=ABM_CE_PV_model_reporters,
            agenttype_reporters=ABM_CE_PV_agenttype_reporters
        )

    # New edits
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
        Create a mapping of agent ids to their respective site names, PCA, state, and installation year.
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
            p_year = self.uspvdb.loc[self.uspvdb['p_name'] == site_name,
                                     'p_year'].iloc[0]
            agents[agent_id] = (case_id, site_name, pca_value, state_value, p_year)
            agent_id += 1

        return agents
    
    def create_regulator_state_map(self, unique_states:list[str]) -> dict[int, str]:
        """
        Create a mapping of regulator agent ids to their respective states.
        """
        regulator_state_map = {}
        agent_id = self.num_prod_n_recyc + self.num_consumers + \
            self.num_refurbishers
        for state in unique_states:
            regulator_state_map[agent_id] = state
            agent_id += 1
        return regulator_state_map

    def _load_policy_schedule_by_state(self) -> dict[str, dict]:
        """
        Load policy_schedule.yaml once and return a mapping of state abbreviation
        to that state's policy schedule dict (keyed by policy column name).
        Parameters:
        None
        Returns:
        dict[str, dict]: Mapping of state abbreviation to its schedule, e.g.
            {'CA': {'universal_waste_regulation': {'start_year': 2025}}, ...}
        """
        path: str = os.path.join(
            os.path.dirname(__file__), "policy_regulation", "policy_schedule.yaml")
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            config: dict = yaml.safe_load(f) or {}
        raw_policies: dict = config.get('policies') or {}
        schedule_by_state: dict[str, dict] = {}
        for policy_name, entries in raw_policies.items():
            if not entries:
                continue
            for entry in entries:
                states: list[str] = entry.get('states', [])
                entry_schedule: dict = {k: v for k, v in entry.items() if k != 'states'}
                for state in states:
                    schedule_by_state.setdefault(state, {})[policy_name] = entry_schedule
        return schedule_by_state

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
                                           seed=self.seed)
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
        # ! Assuming average lifetime of 25 years so waste at t=0 (2020) is
        # ! weighting (per watt) the factor of the file first entry (iloc[0])
        # ! which is 1995 - correct_mat_factor was calculated by hand from
        # ! PV ICE outputs
        weighted_average_mass_watt = float(self.correct_mat_factor[
            'total_massperW'].iloc[self.clock])

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
        If the RTN model costs are enabled, it returns 0, since the transportation costs are accounted for. 
        Otherwise, it returns the transportation cost.
        """
        if self.rtn:
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

    def tclp_test(self, tclp_market_share_df: pd.DataFrame,
                  current_year: int, state: str = "federal") -> bool:
        """Market-share-weighted Weibull TCLP hazard classification.

        Uses annual BSF and Non-BSF market shares to compute weighted
        combined mean and standard deviation, then samples a latent TCLP
        value from a Weibull distribution parameterized from that combined
        mean/std. Returns True if the sampled latent value exceeds the
        hazard cutoff.

        Parameters
        ----------
        tclp_market_share_df : pd.DataFrame
            DataFrame containing annual BSF and Non-BSF market shares.
        current_year : int
            Current simulation year used to select market shares.
        state : str
            State of the module, used to determine hazard cutoff.

        Returns
        -------
        bool
            True if sampled latent value > hazard_cutoff.

        Notes
        -----
        Initial TCLP market share values are based on U.S. panel sales
        observed during 2005-2010. Assuming an average panel lifetime of
        30 years, these sales shares are shifted forward by 30 years to
        represent end-of-life market shares. Linear interpolation is then
        applied to estimate intermediate yearly values between anchor years.
        """

        min_year = int(tclp_market_share_df["Year"].min())
        max_year = int(tclp_market_share_df["Year"].max())
        lookup_year = max(min_year, min(current_year, max_year))

        market_share_row = tclp_market_share_df.loc[
            tclp_market_share_df["Year"] == lookup_year].iloc[0]
        bsf_share = float(market_share_row["BSF"])
        non_bsf_share = float(market_share_row["Non-BSF"])

        bsf_mean = self.tclp_params["bsf_mean"]
        bsf_std = self.tclp_params["bsf_std"]
        non_bsf_mean = self.tclp_params["non_bsf_mean"]
        non_bsf_std = self.tclp_params["non_bsf_std"]
        min_std = self.tclp_params.get("min_std", 0.0)

        combined_mean = bsf_share * bsf_mean + non_bsf_share * non_bsf_mean
        combined_std = bsf_share * bsf_std + non_bsf_share * non_bsf_std
        combined_std = max(combined_std, min_std)

        cutoff = self.tclp_params["hazard_cutoff"].get(
            state, self.tclp_params["hazard_cutoff"]["federal"])

        weibull_shape = (combined_std / combined_mean) ** -1.086
        weibull_scale = combined_mean / gamma(1 + 1 / weibull_shape)
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
        # Calculate yearly waste using pv_ice_waste_calculation method
        # pass pv_output
        self.pv_ice_yearly_waste = 0