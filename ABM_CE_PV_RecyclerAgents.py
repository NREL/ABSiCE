# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Recycler
"""

from mesa import Agent
import numpy as np


class Recyclers(Agent):
    """
    A recycler which sells recycled materials and improve its processes.

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_CE_PV_Model)
        recycling_costs_df (dataframe with recycling costs for each recycling
            facility, year, and pca category). The dataframe is used to
            update the recycling costs of recyclers. If the pca is not
            available, the recycling costs are updated using the
            original_recycling_cost.

        Config-derived values (read from the model at agent-creation time,
        not passed as constructor args):
        original_recycling_cost (a list for a triangular distribution)
            ($/metric ton) (default≈[400, 400, 400]). Read from
            self.model.original_recycling_cost (NOT self.model.config.cost.*).
        init_eol_rate["recycle"] (initial recycle EOL ratio), (default=0.1).
            From Monteiro Lunardi et al 2018 and European Commission (2015).
            (self.model.config.eol.init_eol_rate)
        recycling_learning_shape_factor, (default=-0.39). From Qiu & Suh 2019.
            (self.model.config.cost.recycling_learning_shape_factor)

    """

    def __init__(self, unique_id, model, recycling_costs_df):
        """
        Creation of new recycler agent
        """
        super().__init__(model)
        self.unique_id = unique_id
        original_recycling_cost = self.model.original_recycling_cost
        self.original_recycling_cost = np.random.triangular(
            original_recycling_cost[0], original_recycling_cost[2],
            original_recycling_cost[1])
        self.recycling_costs_df = recycling_costs_df
        self.original_fraction_recycled_waste = \
            self.model.config.eol.init_eol_rate["recycle"]
        self.recycling_learning_shape_factor = \
            self.model.config.cost.recycling_learning_shape_factor
        self.recycling_cost = self.original_recycling_cost
        self.init_recycling_cost = self.original_recycling_cost
        self.recycler_total_volume = 0
        self.recycling_volume = 0
        self.repairable_volume = 0
        self.total_repairable_volume = 0
        #  Original recycling volume is based on previous years EoL volume
        # (from 2000 to 2019). Computed once in ABM_CE_PV_Model.__init__ from
        # the valid_pcas-filtered all_pca_df_out (NOT a re-read of the
        # all_pca_dataOut CSV, to correctly respect model_states filtering).
        yearly_waste = self.model.original_eol_baseline_volume
        self.original_recycling_volume = \
            (1 - self.model.repairability) * \
            self.original_fraction_recycled_waste * yearly_waste
        self.symbiosis = False
        self.agent_i = self.unique_id - self.model.num_consumers
        self.recycler_costs = 0
        # Recycler node ids are assigned positionally: this recycler's name is
        # its offset into the model's row-ordered facility list.
        self.recycler_name = self.model.recycler_facilities[self.agent_i]
        self.hazardous = False
        self.universal_waste = False
        self.verified = False
        self.set_recycler_type()

    def set_recycler_type(self):
        # Check if the recycler is a hazardous waste recycler
        if self.model.hazardous_waste_regulation_enabled:
            hazardous_recycler_row = self.model.recycler_data[self.model.recycler_data['Recycler Name'] == self.recycler_name]
            if not hazardous_recycler_row.empty and hazardous_recycler_row['RCRA permit'].values[0]:
                self.hazardous = True
            # Check if the recycler is a universal waste recycler
            universal_waste_recycler_row = self.model.universal_waste_recyclers_data[
                self.model.universal_waste_recyclers_data['Recycler Name'] == self.recycler_name]
            if not universal_waste_recycler_row.empty and universal_waste_recycler_row['Universal Waste Permit'].values[0]:
                self.universal_waste = True
                self.hazardous = False  # If the recycler is a universal waste recycler, it cannot be a hazardous waste recycler
        
    def get_recycling_cost(self, facility_id: int = None) -> float:  # facility_id UNUSED: parameter defined but never used in function body
        """
        Get the recycling cost of the recycler.
        Either from the recycling costs dataframe or the original recycling cost.
        If the model is using the RTN, the recycling costs are obtained from the recycling_costs_df
        dataframe, which is updated with the recycling costs from the RTN model for each year, site and recycler.
        """
        if self.model.rtn:
            # Get the recycling cost from the dataframe for the current year and recycler name
            recycling_cost_row = self.recycling_costs_df[
                (self.recycling_costs_df['date'] <= self.model.current_date) &
                (self.recycling_costs_df['Recycler Name'] == self.recycler_name) &
                (self.recycling_costs_df['Site'] == facility_id)
            ]
            # If the row is not empty, return the recycling cost
            # Otherwise, return the original recycling cost
            if not recycling_cost_row.empty:
                recycling_cost_row = recycling_cost_row.sort_values(by='date', ascending=False).iloc[0]
                if np.isnan(recycling_cost_row['Cost']):
                    print(f"Warning: Recycling cost for {self.recycler_name} in {self.model.current_date.year} is NaN. Using infinity as cost.")
                    recycling_cost = np.inf
                else:
                    recycling_cost = recycling_cost_row['Cost']
                return recycling_cost 
                  
        return self.original_recycling_cost

    def update_recycled_waste(self):
        """
        Update consumers' amount of recycled waste.
        """
        if self.unique_id == self.model.num_consumers:
            for agent in self.model.agents:
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
        for agent in self.model.agents:
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
            for agent in self.model.agents:
                if agent.unique_id < self.model.num_consumers and \
                        agent.get_active_recycling_facility_id() == self.unique_id:
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
            for agent in self.model.agents:
                if agent.unique_id < self.model.num_consumers and \
                        agent.get_active_recycling_facility_id() == self.unique_id:
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
        if volume > 0 and original_volume > 0:
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
        for agent in self.model.agents:
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
            self.get_recycling_cost(),
            self.recycling_learning_shape_factor)
