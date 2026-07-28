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
        max_storage_days: The maximum storage limit allowed for the generator size.
        waste_generation_limit_kg: The maximum waste generation limit for the generator size.
    """
    
    max_storage_kg: Optional[int]
    max_storage_days: Optional[int]
    waste_generation_limit_kg: Optional[int]  

    @classmethod
    def from_dict(cls, thresholds_dict: dict):
        """
        Create a GeneratorSizeThreshold instance from a dictionary.
        """
        max_storage_kg = thresholds_dict.get("max_storage_kg")
        max_storage_days = thresholds_dict.get("max_storage_days")
        waste_generation_limit_kg = thresholds_dict.get("waste_generation_limit_kg")
        return cls(
            max_storage_kg=max_storage_kg if pd.notna(max_storage_kg) else None,
            max_storage_days=max_storage_days if pd.notna(max_storage_days) else None,
            waste_generation_limit_kg=waste_generation_limit_kg if pd.notna(waste_generation_limit_kg) else None

        )

class Regulators(Agent):
    """
    A regulator agent that sets the regulations for PV waste management.
    It requires the following input files:
    - policy_by_state.csv: Contains the base regulatory policies applicable to each state.
    - generator_threshold.csv: Contains the thresholds for different generator sizes.
    - policy_schedule.yaml (optional): Configures dynamic activation/deactivation of policies
      by state and simulation year.
    Attributes:
        unique_id: int - Unique identifier for the agent.
        model: Model - The model this agent belongs to.
        policy_schedule_path: Optional[str] - Path to the YAML policy schedule config file.
            Defaults to policy_regulation/policy_schedule.yaml relative to this module.
        thresholds: dict - A dictionary mapping generator sizes to their thresholds (for hazardous waste).
    """

    def __init__(self,
                 unique_id: int,
                 model: Model,
                 policy_schedule: Optional[dict] = None,
                 ):
        """
        Creation of new regulator agent.
        Parameters:
        unique_id (int): Unique identifier for the agent.
        model (Model): The model this agent belongs to.
        policy_schedule (Optional[dict]): Pre-parsed policy schedule for this state's
            policies, keyed by policy column name. Each value is a dict with optional
            keys 'start_year' (int) and 'end_year' (int). Typically provided by the
            model after loading policy_schedule.yaml once. Defaults to no schedule.
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.regulator_state = self.model.regulator_state_map[unique_id]
        self.regulatory_policy = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "policy_regulation", "policy_by_state.csv"))
        # Use a copy so that in-place updates during the simulation do not
        # affect other agents sharing the same underlying DataFrame.
        self.current_regulatory_policy = (
            self.regulatory_policy[self.regulatory_policy['state'] == self.regulator_state]
            .copy()
        )
        self._policy_schedule: dict[str, dict] = policy_schedule or {}
        # Initialize thresholds for different generator sizes
        generator_threshold_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "policy_regulation", "generator_threshold.csv"))
        # Determine the applicable state for thresholds, defaulting to "FED" if not found
        threshold_state = self.regulator_state if self.regulator_state in generator_threshold_df['state'].values else "FED"
        thresholds = generator_threshold_df[generator_threshold_df['state'] == threshold_state].set_index('generator_size').to_dict(orient='index')
        self.thresholds = {}
        # Map generator sizes to their thresholds from the DataFrame
        for size, thresh_dict in thresholds.items():
            self.thresholds[GeneratorSize(size)] = GeneratorSizeThreshold.from_dict(thresh_dict)
        # Initialize universal waste thresholds
        universal_waste_threshold_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "policy_regulation", "universal_waste_generator_threshold.csv"))
        universal_waste_thresholds = universal_waste_threshold_df[universal_waste_threshold_df['state'] == threshold_state].set_index('generator_size').to_dict(orient='index')
        self.universal_waste_thresholds = {}
        for size, thresh_dict in universal_waste_thresholds.items():
            self.universal_waste_thresholds[GeneratorSize(size)] = GeneratorSizeThreshold.from_dict(thresh_dict)

    def _is_transfer_based_exclusion(self, recycler_id: int):
        """
        Check if the product is exempt from regulations based on transfer-based exclusions.
        """
        if self.current_regulatory_policy is not None and self.current_regulatory_policy["transfer_based_exclusion"].values[0] == True:
            if self.model.agent_map[recycler_id].hazardous:
                # If the recycler is hazardous waste certified, transfer-based exclusion applies
                return True
        return False  # No transport-based exclusion applies by default

    def _is_chemical_based_exclusion(self): 
        """
        Check if the product is exempt from regulations based on chemical-based exclusions.
        """
        return False  # From Taylor's study, unclear if this applies to PV modules, but included for completeness

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

    def is_epr_applicable(self) -> bool:
        """
        Check if Extended Producer Responsibility (EPR) regulation is active
        for this state. When True, the landfill pathway is disabled for all
        PV owners in the state.
        Returns:
        bool: True if EPR is currently active for this state.
        """
        if (not self.current_regulatory_policy.empty
                and self.current_regulatory_policy["epr"].values[0] == True):
            return True
        return False

    def is_recycling_bonds_applicable(self) -> bool:
        """
        Check if recycling bonds are active for this state. When True, all
        recycling costs (base cost, transportation, and waste management
        premiums) are covered by the bonds and are effectively zero for
        PV owners in the state.
        Returns:
        bool: True if recycling bonds are currently active for this state.
        """
        if (not self.current_regulatory_policy.empty
                and self.current_regulatory_policy["recycling_bonds"].values[0] == True):
            return True
        return False

    def check_and_update_regulations(self) -> None:
        """
        Check and update regulatory policy values for this state based on the
        current simulation year and the entries in policy_schedule.yaml.

        Scheduling rules (all year comparisons use the calendar year derived
        from the model clock):
        - start_year only  : policy is False before start_year, True from
                             start_year to end of simulation.
        - end_year only    : policy follows the initial CSV value until
                             end_year, then becomes False for the remainder.
        - both             : policy is True during [start_year, end_year),
                             False otherwise.
        - neither          : policy value is taken directly from
                             policy_by_state.csv with no change.
        """
        if self.current_regulatory_policy.empty or not self._policy_schedule:
            return

        current_year: int = self.model.current_date.year

        row_idx: int = self.current_regulatory_policy.index[0]
        for policy_name, schedule in self._policy_schedule.items():
            if policy_name not in self.current_regulatory_policy.columns:
                continue

            start_year: Optional[int] = schedule.get('start_year')
            end_year: Optional[int] = schedule.get('end_year')

            if start_year is not None and end_year is not None:
                # Active window: [start_year, end_year)
                new_value: bool = start_year <= current_year < end_year
            elif start_year is not None:
                # Activated at start_year, remains active to end of simulation
                new_value = current_year >= start_year
            elif end_year is not None:
                # No change needed before end_year; deactivate from end_year onwards
                if current_year < end_year:
                    continue
                new_value = False
            else:
                continue  # No schedule keys; skip

            self.current_regulatory_policy.at[row_idx, policy_name] = new_value

    def step(self) -> None:
        """
        The step function for the regulator agent.
        It updates the regulatory policies based on the current simulation year.
        """
        self.check_and_update_regulations()

        