# Agent-Based Model (ABM) Circular Economy (CE) Solar Photovoltaic (PV)

## Table of Content

- [Overview](#overview)
- [Model Description](#model-description)
- [Files](#files)
- [Installation](#installation)
- [License](#license)

## Overview
The **Agent-Based Model (ABM) Circular Economy (CE) Solar Photovoltaic (PV)** designed to simulate how social factors influence **end-of-life (EOL)** management decisions. 

The model focuses on how agents (people and organization) choose between CE pathways (repair, reuse, and recycling) or linear pathways such as storage or landfilling.

ABM represents each stakeholder type with its own behavioral attributes, motivations, decisions rules, and interactions. This allows the system to exhibit realistic 
behaviors driven by social influence, peer pressure, policy, infrastructure availability, and perceived risk.

## Model Description

1. **Agents**
    - Consumers: PV owners
    - Producers: PV manufacturers
    - Recyclers: Sells recycled materials and improve its processes
    - Refurbishers: Deal with used, repaired modules
    - Regulators: Enforce policy regulations by state

2. **End-of-Life (EOL) Management options** 
    -  Repair: Restore functionality
    -  Reuse: Re-deploy panels in secondary markets
    -  Recycling: Recover materials
    -  Landfilling: Dispose of panels as waste
    -  Storage: Temporarily hold panels when no decision is made or options are unclear

3. **Theory of Planned Behavioral (TPB)** 
    - The model incorporates the TPB, a widely used behavioral framework explaning how intention predicts actual behavior
        - **Attitude (A)** = Each agent holds attitude value between: 
            - 0 = negative attitude toward CE pathways
            - 1 = postivite attitude toward CE pathways
            - 1 - CE_Pathway = linear pathway (landfilling and storing)
        - **Subjective Norm (SN)** = perceived social pressure or community expectations
        - **Perceived Behavioral Control (PBC)** = belief in one's ability to act (i.e. access to recycling facilites)

4. **Toxicity Characteristic Leaching Procedure (TCLP)**
    -  Increase perceived environmental risk
    -  Increase pressuure to avoid landfilling
    -  Influence social norms and policy constraints

5. **Model Inputs**
    - Locations of landfill and recycling centers
    - Policy settings and toggles
    - Costs for different waste-management pathways
    - Parameter values (from different tabs or settings)
    - Initial end‑of‑life (EOL) rates
    - Timestep length and number of years to simulate

6. **Model Outputs**
    - Waste amounts in kg by pathway (i.e., landfill, recycle, incineration)
    - Reported for each timestep across the simulation period

## Files
**/ABSiCE/**
- **ABM_CE_PV_Model.py**


- **ABM_CE_PV_BatchRun.py**

- **ABM_CE_PV_MultipleRun.py**

- **ABM_CE_PV_ConsumerAgents.py**

- **ABM_CE_PV_ProducerAgents.py**

- **ABM_CE_PV_RecyclerAgents.py**

- **ABM_CE_PV_RefurbisherAgents.py**

## Installation/Setup

1. **Install Conda** (if you have it, skip to Step 2) 
    - Option A: Anaconda (large, includes many packages)

        https://www.anaconda.com/products/distribution
    - Option B: Miniconda (lightweight, recommended)

        https://docs.anaconda.com/miniconda

After installing, open a terminal and check installation:
```bash
conda --version
```

2. **Forked the repository**

    - Click the **Fork** button in the top right corner of the page (https://github.com/NatLabRockies/ABSiCE.git)


3. **Cloning Your Fork** on your local machine
```bash
git clone https://github.com/your-username/NatLabRockies/ABSiCE.git
cd ABSiCE
```

4. **Create a New Branch** from *current-development* branch
```bash
git checkout current-development-branch
git pull
git checkout -b your-new-branch-name
```

5. **Create the Conda environment**

The repository includes a pre‑configured environment file: 'pv_abm_env_platform_independent.yaml'
```bash
conda env create -f pv_abm_env_platform_independent.yaml
```

6. **Activate the environment**
```bash
conda activate pv_abm
```
Your terminal prompt should now begin with:
```bash
(pv_abm)    user@machine ...
```
This indicates you are working the correct software environment

7. **Verify installation**

Check Python:
```bash
python --version
```
List installed packages:
```bash
conda list
```

8. **Run the ABM**

The model file that contains the core logic is *ABM_CE_PV_Model.py*, it initilaizes and instantiates all the agents (PV owners, installers, 
recyclers, and manufacturers).

The file to run the model is *ABM_CE_PV_MultipleRun.py*, imports the model and calls its run method.
```bash
python ABM_CE_PV_MultipleRun.py
```

9. **Deactivating the environment**

When you're done
```bash
conda deactivate
```

## License
The project license is included in the repository root.

For more details please refer to the publication:

Walzberg, J., A. Carpenter, and G. A. Heath. 2021. “Exploring PV Circularity by Modeling Socio-Technical Dynamics of Modules’ End-of-Life Management.” 2021 IEEE 48th Photovoltaic Specialists Conference (PVSC, June 20, 0041-0043,. https://doi.org/10.1109/PVSC43889.2021.9518638 )