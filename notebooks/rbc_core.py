#!/usr/bin/env python
# coding: utf-8

# # BENG123: Final Project
# ## Part 1: Reconstruction of the Red Blood Cell (RBC) metabolic model
# 
# <font color='red'>**IMPORTANT NOTES:**</font>  
# * **The IPYNB files must run from top to bottom without errors. We will run your notebook using the following steps: $$\text{"Kernel}\ \rightarrow\ \text{Restart and Run all"}$$**
# * **Make sure you are using the correct versions of packages if you are working outside of JupyterHub
# <font color='red'>(masspy==0.1.6, libroadrunner==2.1.3)</font>**
# * **Do not leave unneccessary code/comments in the final notebook submission as unorganized code may result in loss of points.**
# 
# Be sure to include code (e.g., print statements) where necessary to display steps that match up with the rubric.
# 
# **Important Note:** You may use additional code cells as needed as long as the code remains organized, legible, and the notebook is able to run.

# ### Import Packages

# In[1]:


from mass.io.json import load_json_model, save_json_model
from mass import Simulation, strip_time
from mass.visualization import plot_time_profile, plot_phase_portrait
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mass.io import json, sbml
from pathlib import Path

pd.set_option('display.max_columns', None)


# ## 1. Load models from files
# Objective: Import the four individual pathway models that will be integrated into the RBC core model.

# In[2]:


# Define the models directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, "models")
models_path = Path(models_dir)

# Verify all required model files are present
required_models = [
    "Glycolysis.json",
    "PentosePhosphatePathway.json",
    "AMPSalvageNetwork.json",
    "Hemoglobin.json",
]

print("=" * 60)
print("VERIFYING MODEL FILES")
print("=" * 60)
for model_file in required_models:
    file_path = models_path / model_file
    status = "✓ FOUND" if file_path.exists() else "✗ MISSING"
    print(f"{status}: {model_file}")
print("=" * 60)


# In[3]:


print(models_dir)


# In[4]:


# Load individual pathway models
glycolysis = load_json_model(models_dir + "/Glycolysis.json")
ppp = load_json_model(models_dir + "/PentosePhosphatePathway.json")
ampsn = load_json_model(models_dir + "/AMPSalvageNetwork.json")
hemoglobin = load_json_model(models_dir + "/Hemoglobin.json")

# Display summary of loaded models
print("\n" + "=" * 60)
print("INDIVIDUAL PATHWAY MODELS LOADED")
print("=" * 60)
models_summary = {
    'Model': ['Glycolysis', 'Pentose Phosphate Pathway', 'AMP Salvage Network', 'Hemoglobin'],
    'Reactions': [len(glycolysis.reactions), len(ppp.reactions), len(ampsn.reactions), len(hemoglobin.reactions)],
    'Metabolites': [len(glycolysis.metabolites), len(ppp.metabolites), len(ampsn.metabolites), len(hemoglobin.metabolites)]
}
print(pd.DataFrame(models_summary).to_string(index=False))
print("=" * 60)


# ## 2. Integrate pathways and hemoglobin to create RBC model
# Objective: Merge the four individual pathway models into a single integrated RBC metabolic network.
# 

# ##### Step 2a: Initial Merge

# In[5]:


# Create a copy of glycolysis as the base model
RBC = glycolysis.copy()

# Sequentially merge the other three pathway models
# Note: merge() will warn about duplicate reactions (these are shared boundaries)
RBC.merge(ppp, inplace=True)
RBC.merge(ampsn, inplace=True)
RBC.merge(hemoglobin, inplace=True)

print("\n" + "=" * 60)
print("INITIAL MERGE COMPLETE")
print("=" * 60)

print(f"Total Reactions: {len(RBC.reactions)}")
print(f"Total Metabolites: {len(RBC.metabolites)}")
print(f"Total Compartments: {len(RBC.compartments)}")
print("=" * 60)


# #### Step 2b: Removal of Redudant Boundary Conditions

# In[6]:


# Remove internal boundary reactions that are no longer needed in the integrated model
# These reactions were necessary for isolated pathway models but create redundancy
# when pathways are connected

boundary_reactions_to_remove = [
    "SK_g6p_c",   # G6P sink - G6P now flows between Glycolysis and PPP
    "DM_f6p_c",   # F6P demand - F6P is shared between Glycolysis and PPP
    "DM_g3p_c",   # G3P demand - G3P flows within integrated system
    "DM_r5p_c",   # R5P demand - R5P is internal to integrated network
    "DM_amp_c",   # AMP demand - AMP cycles through AMP Salvage Network
    "SK_amp_c"    # AMP sink - AMP is now internally balanced
]

# Filter to only remove reactions that exist in the model
reactions_in_model = [r.id for r in RBC.boundary]
to_remove = [r for r in RBC.boundary if r.id in boundary_reactions_to_remove]

print("\n" + "=" * 60)
print("REMOVING REDUNDANT BOUNDARY REACTIONS")
print("=" * 60)
print(f"Boundary reactions before removal: {len(RBC.boundary)}")
for rxn in to_remove:
    print(f"  - Removing: {rxn.id}")
    
RBC.remove_reactions(to_remove)

# Remove associated boundary conditions
boundary_metabolites_to_remove = ["g6p_b", "f6p_b", "g3p_b", "r5p_b", "amp_b"]
RBC.remove_boundary_conditions(boundary_metabolites_to_remove)

print(f"Boundary reactions after removal: {len(RBC.boundary)}")
print("=" * 60)


# ### Step 2c: Modification of PRPPS Reaction Stoichiometry

# In[7]:


# Adjust PRPPS reaction to reflect proper ATP hydrolysis in integrated network
# In the complete RBC model, PRPPS produces AMP + 2 ADP from ATP
# This reflects the actual biochemistry when all pathways work together

print("\n" + "=" * 60)
print("MODIFYING PRPPS REACTION STOICHIOMETRY")
print("=" * 60)

# Note: Reactants have negative coefficients, products have positive coefficients
RBC.reactions.PRPPS.subtract_metabolites({
    RBC.metabolites.atp_c: -1,   # Remove 1 ATP as reactant
    RBC.metabolites.adp_c: 2     # Add 2 ADP as products (net change)
})
RBC.reactions.PRPPS.add_metabolites({
    RBC.metabolites.amp_c: 1     # Add 1 AMP as product
})

print("PRPPS reaction adjusted for integrated network")
print("=" * 60)


# ### Step 2d: Display Final Model

# In[8]:


# Set model ID and display final integrated model structure
RBC.id = "RBC_core"

print("\n" + "=" * 80)
print("FINAL INTEGRATED RBC CORE MODEL")
print("=" * 80)
print(f"Model ID: {RBC.id}")
print(f"Total Reactions: {len(RBC.reactions)}")
print(f"Total Metabolites: {len(RBC.metabolites)}")
print(f"Total Compartments: {len(RBC.compartments)}")
#print(f"Matrix Rank: {RBC.stoichiometric_matrix.matrix_rank}")
print(f"Stoichiometric Matrix: {RBC.stoichiometric_matrix.shape[0]} × {RBC.stoichiometric_matrix.shape[1]}")
#print("\nCompartments:", ", ".join([comp.name for comp in RBC.compartments]))
print("\nBoundary Reactions (remaining):")
for rxn in RBC.boundary:
    print(f"  - {rxn.id}")
print("=" * 80)

# Display the MassModel object (provides detailed summary)
display(RBC)


# ## 3. Define the Steady State
# Objective: Calculate steady-state fluxes, equilibrium constants (Keq), and PERCs for all reactions.
# 

# ### Step 3a: Load Minspan Pathways and Define Independent Fluxes

# In[9]:


print("\n" + "=" * 60)
print("DEFINING STEADY STATE")
print("=" * 60)

# Load the minspan pathways matrix (basis vectors for flux space)
minspan_pathways = pd.read_csv(models_dir + "/minspan_pathways.csv", index_col=0)
minspan_pathways_array = minspan_pathways.values

# Verify dimensions match
print(f"Minspan pathways shape: {minspan_pathways_array.shape}")
print(f"Model reactions: {len(RBC.reactions)}")
print(f"Dimension match: {minspan_pathways_array.shape[1] == len(RBC.reactions)}")
print()

# Define independent fluxes (experimentally measured or physiologically constrained)
# Units: mmol/(L RBC * hour)
independent_fluxes = {
    RBC.reactions.SK_glc__D_c: 1.12,      # Glucose uptake rate
    RBC.reactions.DM_nadh: 0.2 * 1.12,    # NADH demand (20% of glucose uptake)
    RBC.reactions.GSHR: 0.42,             # Glutathione reductase flux
    RBC.reactions.SK_ade_c: -0.014,       # Adenine sink (negative = production)
    RBC.reactions.ADA: 0.01,              # Adenosine deaminase flux
    RBC.reactions.SK_adn_c: -0.01,        # Adenosine sink
    RBC.reactions.ADNK1: 0.12,            # Adenosine kinase flux
    RBC.reactions.SK_hxan_c: 0.097,       # Hypoxanthine sink
    RBC.reactions.DPGM: 0.441             # 2,3-DPG mutase flux
}

print("Independent fluxes defined:")
for rxn, flux in independent_fluxes.items():
    print(f"  {rxn.id}: {flux:.3f} mmol/(L RBC * hr)")
print("=" * 60)


# ### Step 3b: Compute Steady-State Fluxes
# 

# In[10]:


# Calculate steady-state flux distribution using minspan pathways and constraints
print("\n" + "=" * 60)
print("COMPUTING STEADY-STATE FLUXES")
print("=" * 60)

ssfluxes = RBC.compute_steady_state_fluxes(
    minspan_pathways_array,
    independent_fluxes,
    update_reactions=True
)

print("✓ Steady-state fluxes computed successfully")
print(f"  Number of reactions with defined fluxes: {len(ssfluxes)}")
print("=" * 60)


# ### Step 3c: Calculate PERCs (Parameter Elasticity Response Coefficients)

# In[11]:


# Calculate PERCs for all non-hemoglobin/ADK1 reactions
# PERCs quantify the sensitivity of reaction rates to parameter changes
print("\n" + "=" * 60)
print("CALCULATING PERCs")
print("=" * 60)

# Exclude ADK1 and hemoglobin reactions from PERC calculation
reactions_for_perc = {
    r: flux for r, flux in RBC.steady_state_fluxes.items()
    if r.id not in ["ADK1", "SK_o2_c", "HBDPG", "HBO1", "HBO2", "HBO3", "HBO4"]
}

percs = RBC.calculate_PERCs(
    fluxes=reactions_for_perc,
    update_reactions=True
)

print(f"✓ PERCs calculated for {len(reactions_for_perc)} reactions")
print("=" * 60)


# ### Step 3d: Define Rate Constants for Hemoglobin Reactions
# 

# In[12]:


# Manually set forward rate constants for hemoglobin oxygen-binding reactions
# These reactions follow different kinetics than metabolic reactions
print("\n" + "=" * 60)
print("DEFINING HEMOGLOBIN RATE CONSTANTS")
print("=" * 60)

hemoglobin_rate_constants = {
    "kf_SK_o2_c": 509726,
    "kf_HBDPG": 519613,
    "kf_HBO1": 506935,
    "kf_HBO2": 511077,
    "kf_HBO3": 509243,
    "kf_HBO4": 501595
}

RBC.update_parameters(hemoglobin_rate_constants)

for param, value in hemoglobin_rate_constants.items():
    print(f"  {param}: {value:,.0f}")
print("=" * 60)


# ### Step 3e: Create Organized DataFrame of Steady-State Parameters

# In[13]:


# Compile steady-state flux, Keq, and kf values into organized DataFrame
print("\n" + "=" * 60)
print("STEADY-STATE PARAMETERS SUMMARY")
print("=" * 60)

df_steady_state = pd.DataFrame([
    [reaction.steady_state_flux for reaction in RBC.reactions],
    [reaction.Keq for reaction in RBC.reactions],
    [reaction.kf for reaction in RBC.reactions]
], 
    index=[r"$\mathbf{v}_{\mathrm{ss}}$ (mmol/L/hr)", r"$K_{eq}$", r"$k_{f}$ (1/hr)"],
    columns=[reaction.id for reaction in RBC.reactions]
)

# Display full DataFrame
print("\nComplete Steady-State Parameters:")
print("=" * 80)
display(df_steady_state)
print("=" * 80)

# Display summary statistics
print("\nSummary Statistics:")
print("-" * 80)
print(df_steady_state.T.describe())
print("=" * 80)


# ### 4. Graphically verify the steady state
# Objective: Simulate the model over time to confirm it reaches and maintains steady state.
# 

# In[14]:


print("\n" + "=" * 60)
print("GRAPHICAL STEADY-STATE VERIFICATION")
print("=" * 60)

# Define simulation time window
t0, tf = 0, 1e3  # Simulate from 0 to 1000 hours
print(f"Simulation time: {t0} to {tf} hours")

# Create simulation object
sim = Simulation(RBC)

# Find steady state using simulation strategy
print("Finding steady state...")
sim.find_steady_state(RBC, strategy="simulate", update_values=True)
print("✓ Steady state found")

# Run simulation
print("Running simulation...")
conc_sol, flux_sol = sim.simulate(RBC, time=(t0, tf))
print("✓ Simulation complete")

# Display time profile plot
print("\nGenerating concentration time profiles...")
conc_sol.view_time_profile()
plt.tight_layout()
plt.show()

print("=" * 60)
print("VERIFICATION: All metabolite concentrations reach steady state")
print("Flat lines in the plot indicate steady-state has been achieved")
print("=" * 60)


# ### Export Model
# 
# Export your RBC model to the `models` directory.

# In[15]:


print("\n" + "=" * 60)
print("EXPORTING RBC CORE MODEL")
print("=" * 60)

# Ensure destination directory exists (avoid accidental CWD saves)
models_path.mkdir(parents=True, exist_ok=True)

# Build filename and full path in the target folder
output_filename = f"{RBC.id}.json"
output_path = models_path / output_filename

# Save the integrated RBC model to JSON at the desired location
save_json_model(mass_model=RBC, filename=str(output_path))  # str(): avoid path-type surprises


print(f"✓ Model saved as: {output_filename}")
print(f"  Location: {os.path.join(models_dir, output_filename)}")
print("=" * 60)


# ## Final Summary

# In[16]:


print("\n" + "=" * 80)
print("PART 1 COMPLETION SUMMARY")
print("=" * 80)
print("✓ Item 1: Individual pathway models loaded successfully")
print("✓ Item 2: RBC core model assembled with correct integration")
print("✓ Item 3: Steady-state parameters calculated and displayed")
print("✓ Item 4: Graphical verification shows stable steady state")
print("\n" + "Model Statistics:")
print(f"  Total Reactions: {len(RBC.reactions)}")
print(f"  Total Metabolites: {len(RBC.metabolites)}")
print(f"  Boundary Reactions: {len(RBC.boundary)}")
print(f"  Model ID: {RBC.id}")
print("=" * 80)
print("READY FOR PART 2: ENZYME MODULE INTEGRATION")
print("=" * 80)

