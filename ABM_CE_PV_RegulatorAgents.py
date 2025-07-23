# -*- coding:utf-8 -*-
"""
Created on Thu Jul 10 2025

@author Purboday Ghosh - pghosh@nrel.gov

Agent - Regulator
"""

from mesa import Agent, Model
import numpy as np
import pandas as pd
from dataclasses import dataclass
from utils import GeneratorSize
from typing import Optional

@dataclass
class RegulatoryPolicy:
    """
    A class to represent the regulatory policies for PV waste management.
    Attributes:
        policy_type: The type of policy, e.g., "exclusions" or "alternative_management_standards".
        states: A list of states where the policy applies.
        duration: The duration for which the policy is applicable.

    Methods:
        is_applicable(state): Checks if the policy is applicable in the given state.
    """

    policy_type: str
    states: list
    duration: int

    def is_applicable(self, state):
        """
        Check if the policy is applicable in the given state.
        """
        return state in self.states
    
    @classmethod
    def from_dict(cls, policy_dict: dict):
        """
        Create a RegulatoryPolicy instance from a dictionary.
        """
        return cls(
            policy_type=policy_dict.get("policy_type"),
            states=policy_dict.get("states", []),
            duration=policy_dict.get("duration", 15)
        )
    
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
        return cls(
            max_storage_kg=thresholds_dict.get("max_storage_kg"),
            max_storage_years=thresholds_dict.get("max_storage_years"),
            waste_generation_limit_kg=thresholds_dict.get("waste_generation_limit_kg")

        )

class Regulators(Agent):
    """
    A regulator agent that sets the regulations for PV waste management.
    Attributes:
        unique_id: int - Unique identifier for the agent.
        model: Model - The model this agent belongs to.
        regulatory_policy: list[RegulatoryPolicy] - A list of regulatory policies applicable to the agent.
        thresholds: dict - A dictionary containing thresholds for different generator sizes.
    """

    def __init__(self, 
                 unique_id: int, 
                 model: Model, 
                 regulatory_policy: Optional[list[dict]] = None,
                 thresholds: dict = {
                     "very_small": {
                            "max_storage_kg": 1000,
                            "max_storage_years": None,
                            "waste_generation_limit_kg": 100
                        },
                        "small": {
                                "max_storage_kg": 6000,
                                "max_storage_years": 0,
                                "waste_generation_limit_kg": 1000
                        },
                        "large": {
                            "max_storage_kg": None,
                            "max_storage_years": 0,
                            "waste_generation_limit_kg": None
                        }
                     }
                     
                 ):
        """
        Creation of new regulator agent
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.regulator_state = self.model.regulator_state_map[unique_id]
        self.regulatory_policy = []
        if regulatory_policy is not None:
            # Convert the list of dictionaries to RegulatoryPolicy instances
            self.regulatory_policy = [RegulatoryPolicy.from_dict(policy) for policy in regulatory_policy]
        self.current_regulatory_policy = None
        if len(self.regulatory_policy) > 0:
            self.current_regulatory_policy = self.regulatory_policy.pop(0)  # Get the first policy if available
        self.internal_clock = 0  # Internal clock to track the duration of the current policy
        # Initialize thresholds for different generator sizes
        self.thresholds = {
            GeneratorSize.VERY_SMALL: GeneratorSizeThreshold.from_dict(thresholds.get(GeneratorSize.VERY_SMALL.value)),
            GeneratorSize.SMALL: GeneratorSizeThreshold.from_dict(thresholds.get(GeneratorSize.SMALL.value)),
            GeneratorSize.LARGE: GeneratorSizeThreshold.from_dict(thresholds.get(GeneratorSize.LARGE.value))
        }

    def _is_transport_based_exclusion(self):
        """
        Check if the product is exempt from regulations based on transport-based exclusions.
        """
        return False # Placeholder for actual logic
    
    def _is_chemical_based_exclusion(self):
        """
        Check if the product is exempt from regulations based on chemical-based exclusions.
        """
        return False # Placeholder for actual logic


    def is_exclusion_applicable(self):
        """
        Check if the product is exempt from regulations based on the state regulations.
        """
        if self.current_regulatory_policy is not None and self.current_regulatory_policy.policy_type == "exclusions":
            if self.current_regulatory_policy.is_applicable(self.regulator_state):
                if self._is_transport_based_exclusion() or self._is_chemical_based_exclusion():
                    return True
        return False  # No exclusions apply by default
    
    def is_alternative_management_standard_applicable(self):
        """
        Check if the product is subject to alternative management standards based on the state regulations.
        """
        if self.current_regulatory_policy is not None and self.current_regulatory_policy.policy_type == "alternative_management_standards":
            if self.current_regulatory_policy.is_applicable(self.regulator_state):
                return True
        return False  # No alternative management standards apply by default
    
    def check_and_update_regulations(self):
        """
        Check and update the regulatory policies based on the model's clock and the current policy duration.
        """

        if self.current_regulatory_policy is not None:
            # Check if the current policy duration has expired
            if self.internal_clock // self.model.timestep.value >= self.current_regulatory_policy.duration:
                # Reset the internal clock
                self.internal_clock = 0
                # Check if there are more policies to apply
                if len(self.regulatory_policy) > 0:
                    self.current_regulatory_policy = self.regulatory_policy.pop(0)
                else:
                    self.current_regulatory_policy = None
            

    def step(self):
        """
        The step function for the regulator agent.
        It updates the regulatory policies based on the model's state.
        """
        self.check_and_update_regulations()
        # Increment the internal clock
        self.internal_clock += 1

        