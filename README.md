# ABM CE PV

## Table of Content

- [Overview](#overview)
- [Model Description](#model-description)
- [Files](#files)
- [Installation](#installation)
- [License](#license)

## Overview
The **Agent-Based Model (ABM) Circular Economy (CE) Solar photovoltaic (PV)** designed to simulate how social factors influence **end-of-life (EOL)** management decisions. 
The model focuses on how agents (people and organization) choose between CE pathways (repair, reuse, and recycling) or linear pathways such as storage or landfilling.

ABM represents each stakeholder type with its own behavioral attributes, motivations, decisions rules, and interactions. This allows the system to exhibit realistic 
behaviors driven by social influence, peer pressure, policy, infrastructure availability, and perceived risk.

## Model Description
1. **Agents**
    -  PV owners: Residential or commercial system owners making EOL decisions
    -  Installers: Intermediaries influencing owner decisions and logistics
    -  Manufacturers: Producers of PV panels with varying levels of responsibility under policy settings
    -  Recyclers: Facilities that process and recover EOL materials

2. **End-of-Life (EOL) Management options** 
    -  Repair: Restore functionality
    -  Reuse: Re-deploy panels in secondary markets
    -  Recycling: Recover materials
    -  Landfilling: Dispose of panels as waste
    -  Storage: Temporarily hold panels when no decision is made or options are unclear

3. **Social & Behavioral Variables**
    -  **Attributes (A)** - Each agent holds attitude value between: 
        - 0 = negative attitude toward CE pathways
        - 1 = postivite attitude toward CE pathways
        - 1 - CE_Pathway = linear pathway (landfilling and storing)
    - **Theory of Planned Behavioral (TPB)** - The model incorporates the TPB, a widely used behavioral framework explaning how intention predicts actual behavior
        - **Attitude (A)** = favorability of CE vs. linear pathways
        - **Subjective Norm (SN)** = perceived social pressure or community expectations
        - **Perceived Behavioral Control (PBC)** = belief in one's ability to act (i.e access to recycling facilites)

4. **Environmental & Policy Context**
    -  Infrastructure availability
        - Recycling centers
        - Collection sites
        - Reuse/refurbishment facilities
    -  Policy settings
        - Regulatory limits
    -  Market conditions
        - Value of recovered materials
        - Cost differences among pathways

5. **Toxicity Characteristic Leaching Procedure (TCLP)**
    -  Increase perceived environmental risk
    -  Increase pressuure to avoid landfilling
    -  Influence social norms and policy constraints

## Files
**/ABSiCE/**
- **ABM_CE_PV_Model.py**
    - [Imported by *BatchRun & MultipleRun* scripts] The core agent-based model simulating a CE for PV products 
        - Sets up environment: network of consumers, producers, recyclers, and refurbisher
        - Contains model parameters: number of agents, TPB decision weights, lifecycle assumptions, costs, material flows, network structures, and policy switches
        - Controls simulation flow: agent creation, scheduling, data collection, behaviors like disposal, recycling, selling

- **ABM_CE_PV_BatchRun.py**
    - Runs multiple simulations with different parameter sets
        - Produces CSV files capturing model-level metrics like product stock, EOL, costs, and recycled material values

- **ABM_CE_PV_MultipleRun.py**
    - Similar to *BatchRun* [add]

- **ABM_CE_PV_ConsumerAgents.py**
    - Defines consumer agents who decide when to replace or dispose of PV products
        - Implements TPB to choose disposal methods (repair, reuse, recycle, landfill, and storage) and new vs used purchases
        - Handles stock updates, waste generation, costs accounting, and interactions with other agents

- **ABM_CE_PV_ProducerAgents.py**
    - Models producer agents active in manufacturing and offering PV products to consumers
        - Decision-making, and interaction with recyclers/refubishers

- **ABM_CE_PV_RecyclerAgents.py**
    - Recycler agents responsible for processing end-of-life PV products
        - Collects waste, calculates recycling costs, processes material flows back into the system

- **ABM_CE_PV_RefurbisherAgents.py**
    - Implements refurbisher agents who repair and resell used PV modules
        - Calculates repair costs

- **StatesAdjacencyMatrix.csv**
    - Contains a matrix of geographic distances between U.S. States
        -   [add]

## Installation/Setup
1. **Create a New Branch** from *rtn-integration-phase-2* branch
```bash
git checkout rtn-integration-phase-2
git pull
git checkout -b your-new-branch-name
```

2. **Install Conda** (if you have it, skip to Step 2) 
    - Option A: Anaconda (large, includes many packages)

        https://www.anaconda.com/products/distribution
    - Option B: Miniconda (lightweight, recommended)

        https://docs.anaconda.com/miniconda

After installing, open a terminal and check installation:
```bash
conda --version
```

3. **Clone the repository** on your local machine
```bash
git clone https://github.com/NatLabRockies/ABSiCE.git
cd ABSiCE
```

4. **Create the Conda environment**

The repository includes a pre‑configured environment file: 'pv_abm_env_platform_independent.yaml'
```bash
conda env create -f pv_abm_env_platform_independent.yaml
```

5. **Activate the environment**
```bash
conda activate pv_abm
```
Your terminal prompt should now begin with:
```bash
(pv_abm)    user@machine ...
```
This indicates you are working the correct software environment

6. **Verify installation**

Check Python:
```bash
python --version
```
List installed packages:
```bash
conda list
```

7. **Run the ABM**

The model file that contains the core logic is *ABM_CE_PV_Model.py*, it initilaizes and instantiates all the agents (PV owners, installers, 
recyclers, and manufacturers).

The file to run the model is *ABM_CE_PV_MultipleRun.py*, imports the model and calls its run method.
```bash
python ABM_CE_PV_MultipleRun.py
```

8. **Deactivating the environment**

When you're done
```bash
conda deactivate
```

## License
The project license is included in the repository root.