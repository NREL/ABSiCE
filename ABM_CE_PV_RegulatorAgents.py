# -*- coding:utf-8 -*-
"""
Created on Thu Jul 10 2025

@author Purboday Ghosh - pghosh@nrel.gov

Agent - Regulator
"""

from mesa import Agent, Model
# import numpy as np  # UNUSED: imported but never used in file
import pandas as pd
from dataclasses import dataclass
from utils import GeneratorSize
from typing import Optional
import os
    
@dataclass
class GeneratorSizeThreshold:
    """
    A class to represent the thresholds for different generator sizes.
    Attributes:
        max_storage_kg: The maximum storage mass allowed for the generator size.
        max_storage_years: The maximum storage limit allowed for the generator size.
        waste_generation_limit_kg: The maximum waste generation limit for the generator size.
    """
    
    max_storage_kg: Optional[int]
    max_storage_years: Optional[int]
    waste_generation_limit_kg: Optional[int]  

    @classmethod
    def from_dict(cls, thresholds_dict: dict):
        """
        Create a GeneratorSizeThreshold instance from a dictionary.
        """
        max_storage_kg = thresholds_dict.get("max_storage_kg")
        max_storage_years = thresholds_dict.get("max_storage_years")
        waste_generation_limit_kg = thresholds_dict.get("waste_generation_limit_kg")
        return cls(
            max_storage_kg=max_storage_kg if pd.notna(max_storage_kg) else None,
            max_storage_years=max_storage_years if pd.notna(max_storage_years) else None,
            waste_generation_limit_kg=waste_generation_limit_kg if pd.notna(waste_generation_limit_kg) else None

        )

class Regulators(Agent):
    """
    A regulator agent that sets the regulations for PV waste management. It requires two csv files:
    - policy_by_state.csv: Contains the regulatory policies applicable to each state.
    - generator_threshold.csv: Contains the thresholds for different generator sizes.
    Attributes:
        unique_id: int - Unique identifier for the agent.
        model: Model - The model this agent belongs to.
        policy_duration: list[dict] - A list of policy durations in years applicable to the agent.
        For example, [{"policy_name": "policy1", "duration": 12}, {"policy_name": "policy2", "duration": 24}]
    """

    def __init__(self, 
                 unique_id: int, 
                 model: Model, 
                 policy_duration: list[dict] = [],             
                 ):
        """
        Creation of new regulator agent
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.internal_clock = 0  # Internal clock to track policy duration
        self.regulator_state = self.model.regulator_state_map[unique_id]
        self.regulatory_policy = pd.read_csv(os.path.join(os.path.dirname(__file__), "policy_regulation", "policy_by_state.csv"))
        self.current_regulatory_policy = self.regulatory_policy[self.regulatory_policy['state'] == self.regulator_state]
        self.policy_duration = policy_duration
        # Initialize thresholds for different generator sizes
        generator_threshold_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "policy_regulation", "generator_threshold.csv"))
        # Determine the applicable state for thresholds, defaulting to "FED" if not found
        threshold_state = self.regulator_state if self.regulator_state in generator_threshold_df['state'].values else "FED"
        thresholds = generator_threshold_df[generator_threshold_df['state'] == threshold_state].set_index('generator_size').to_dict(orient='index')
        self.thresholds = {}
        # Map generator sizes to their thresholds from the DataFrame
        for size, thresh_dict in thresholds.items():
            self.thresholds[GeneratorSize(size)] = GeneratorSizeThreshold.from_dict(thresh_dict)

    def _is_transfer_based_exclusion(self, recycler_id: int):
        """
        Check if the product is exempt from regulations based on transfer-based exclusions.
        """
        if self.current_regulatory_policy is not None and self.current_regulatory_policy["transfer_based_exclusion"].values[0] == True:
            if self.model.agent_map[recycler_id].hazardous:
                # If the recycler is hazardous waste certified, transfer-based exclusion applies
                return True
        return False  # No transport-based exclusion applies by default

    # def _is_chemical_based_exclusion(self):  # DEAD CODE: always returns False, never called
    #     """
    #     Check if the product is exempt from regulations based on chemical-based exclusions.
    #     """
    #     return False  # From Taylor's study, unclear if this applies to PV modules, but included for completeness

    def _is_verified_recycler_based_exclusion(self, recycler_id: int):
        """
        Check if the product is exempt from regulations based on verified recycler exclusions.
        """
        if self.current_regulatory_policy is not None and self.current_regulatory_policy["verified_recycler_exclusion"].values[0] == True:
            if self.model.agent_map[recycler_id].verified:
                # If the recycler is verified, exclusion applies
                return True
        return False  # No verified recycler-based exclusion applies by default

    def is_exclusion_applicable(self, recycler_id: int):
        """
        Check if the product is exempt from regulations based on the state regulations.
        """
        if self._is_transfer_based_exclusion(recycler_id) or self._is_chemical_based_exclusion() or self._is_verified_recycler_based_exclusion(recycler_id):
            return True
        return False  # No exclusions apply by default
    
    def is_universal_waste_regulation_applicable(self):
        """
        Check if the product is subject to universal waste regulations.
        """
        if self.current_regulatory_policy is not None and self.current_regulatory_policy["universal_waste_regulation"].values[0] == True:
            return True
        return False  # No universal waste regulations apply by default

    def check_and_update_regulations(self):
        """
        Check and update the regulatory policies based on the model's clock and the current policy duration.
        """

        if self.current_regulatory_policy is not None:
            # Check if policies have expired
            if len(self.policy_duration) > 0:
                # Mark expired policies as inactive
                for policy_name in self.policy_duration:
                    if policy_name in self.current_regulatory_policy.columns:
                        if self.internal_clock // self.model.timestep.value >= self.policy_duration[policy_name]:
                            self.current_regulatory_policy[policy_name] = False
                            self.internal_clock = 0

    def step(self):
        """
        The step function for the regulator agent.
        It updates the regulatory policies based on the model's state.
        """
        self.check_and_update_regulations()
        # Increment the internal clock
        self.internal_clock += 1

        