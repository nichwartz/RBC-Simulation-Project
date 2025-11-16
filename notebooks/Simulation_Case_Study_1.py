#!/usr/bin/env python
# coding: utf-8

# # BENG123: Final Project
# ## Part 3: Simulation Case Studies and Analysis
# 
# <font color='red'>**IMPORTANT NOTES:**</font>  
# * **The IPYNB files must run from top to bottom without errors. We will run your notebook using the following steps: $$\text{"Kernel}\ \rightarrow\ \text{Restart and Run all"}$$**
# * **Make sure you are using the correct versions of packages if you are working outside of JupyterHub
# <font color='red'>(masspy==0.1.6, libroadrunner==2.1.3)</font>**
# * **Do not leave unneccessary code/comments in the final notebook submission as unorganized code may result in loss of points.**
# 
# 
# ### Instructions
# 1. All simulations and figures that you wish to put in the final report should be done in here. Each figure generated must be displayed when the notebook is run. It is recommended to use one cell per figure generated. 
# 
# 
# 2. A template has been provided to you below to help with code organization. However, you are free to delete the template and use your own organization in this notebook. The only requirement is that this notebook is organized in a logical manner with descriptive headers to help provide clarity. One file can be submitted with all three case studies, or three files can be submitted seperately.
# 
# 
# 3. Remember to save all figures you wish to include as a .PDF or .PNG file and be sure to include them in the zip file of your final submission. All figures in the final report should have figure numbers, appropriate axis labels, limits and scale, legends and captions, and be referred in the text.
# 
# 
# 4. Save all figures in the figures folder provided. This can be done through the `figure.savefig` command. 
# 
# 
# **Recommendation:** It is recommended to utilize seperate notebooks for each case study to avoid possible errors or artificial results. **Consider duplicating this notebook for each case study performed**. Remember, each case study has a maximum of 3 figures per case study! 

# ### Import Packages

# In[1]:


from mass.io.json import load_json_model
from mass import Simulation
from mass.visualization import plot_time_profile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from copy import deepcopy

pd.set_option('display.max_columns', None)


# ### Set Up Directories

# In[2]:


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, "models")
figures_dir = os.path.join(parent_dir, "figures")

# Create figures directory if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)


# ### Load models from files
# Import the RBC model integrated with EnzymeModules from the `models` directory.

# In[3]:


# Import the RBC model integrated with EnzymeModules from the models directory
RBC_PGK = load_json_model(os.path.join(models_dir, "RBC_PGK.json"))
print(f"RBC_PGK model loaded: {len(RBC_PGK.reactions)} reactions, {len(RBC_PGK.metabolites)} metabolites")


# ### Set Up Definitions

# In[17]:


# Helper function to calculate energy charge
def calculate_energy_charge(conc_solution):
    """
    Calculate adenylate energy charge: EC = ([ATP] + 0.5[ADP]) / ([ATP] + [ADP] + [AMP])
    Energy charge indicates cellular energy status (0 = depleted, 1 = fully charged)
    Normal RBC energy charge: 0.85-0.95
    """
    # Access data directly from MassSolution
    time = conc_solution.time
    atp = conc_solution['atp_c']
    adp = conc_solution['adp_c']
    amp = conc_solution['amp_c']
    
    # Calculate energy charge at each time point
    energy_charge = (atp + 0.5 * adp) / (atp + adp + amp)
    
    return time, energy_charge

def find_pgk_param_ids(m):
    """
    Return a sorted list of parameter IDs that belong to the PGK enzyme/module.
    Looks for 'pgk' substring, case-insensitive.
    """
    param_ids = []
    # MASSpy MassModel typically has m.parameters as a DictList of Parameter
    params = getattr(m, "parameters", None)
    if params is None:
        return param_ids
    for p in params:
        pid = getattr(p, "id", None) or getattr(p, "name", None) or ""
        if re.search(r"\bpgk\b", pid, re.I):
            param_ids.append(pid)
    return sorted(set(param_ids))

def scale_model_parameters(m, ids, scale):
    """
    In-place multiply of parameter values by 'scale' for all ids provided.
    Returns a dict {id: old_value} so you can restore if needed.
    """
    changed = {}
    params = getattr(m, "parameters", None)
    if params is None:
        return changed
    for pid in ids:
        try:
            p = params.get_by_id(pid)
            old = p.value
            p.value = old * scale
            changed[pid] = old
        except Exception:
            # If get_by_id not available or id missing, try attribute walk fallback
            for p in params:
                if getattr(p, "id", "") == pid:
                    old = p.value
                    p.value = old * scale
                    changed[pid] = old
                    break
    return changed

def make_pgk_active_fraction(mm, conc_solution, enzyme_id="PGK"):
    make_active_fraction_solution(
        mass_model=mm, 
        concentration_solution=conc_solution,
        enzyme_id=enzyme_id
    )


def make_energy_charge(conc_solution, ratio_id="Energy_Charge"):
    make_ratio_solution(
        concentration_solution=conc_solution, 
        ratio_id=ratio_id,
        numerator_equation="(2 * atp_c + adp_c)",
        denominator_equation="2*(atp_c + adp_c + amp_c)",
        variables=["atp_c", "adp_c", "amp_c"]
    )


# ### Baseline Simulation

# In[5]:


# Run baseline (control) simulation
print("\nRunning baseline simulation...")
sim_control = Simulation(RBC_PGK)
t0, tf = (0, 1000)  # 1000 hours
conc_control, flux_control = sim_control.simulate(RBC_PGK, time=(t0, tf))
time_control, ec_control = calculate_energy_charge(conc_control)
print(f"Baseline energy charge: {ec_control[-1]:.4f}")


# ## Simulation Case Study

# <font color='black'>
# 
# CLINICAL BACKGROUND:- 
# This case study simulates the PGK1 I371T mutation documented in:
# Kugler W, et al. "Molecular and biochemical characterization of phosphoglycerate 
# kinase deficiency in a patient with chronic hemolytic anemia." Blood. 1998.
# DOI: 10.1182/blood.V91.2.504.504
# 
# MUTATION CHARACTERISTICS:
# - Isoleucine to Threonine substitution at position 371
# - Located near the active site, affecting catalytic efficiency
# - SEVERELY REDUCED kcat (turnover number): ~20-30% of wild-type
# - Protein stability PRESERVED: normal enzyme abundance
# - Km values relatively unchanged: substrate binding intact
# 
# CLINICAL PRESENTATION:
# - Chronic hemolytic anemia
# - Low ATP levels in erythrocytes
# - Compensatory reticulocytosis
# - Normal or slightly elevated enzyme protein levels
# - Impaired glycolytic flux through PGK
# 
# MODELING APPROACH:
# We reduce kcat (maximum velocity per enzyme molecule) by 70-75% while maintaining:
# - Normal enzyme concentration (protein abundance)
# - Normal Km values (substrate affinity)
# This specifically models the catalytic defect, NOT protein deficiency.
# 
# PERTURBATION APPLIED:
# - Reduce PGK enzyme Vmax by 75% (representing kcat reduction)
# - All other parameters remain at baseline
# - This isolates the effect of impaired catalysis vs. protein loss
# </font>

# ### Enzyme Inhibited Simulation

# In[6]:


# --- CASE 1: Enzyme Inhibition (I371T) ---
from copy import deepcopy
import numbers, re

RBC_PGK_I371T = deepcopy(RBC_PGK)

# Robust PGK finder: id, name, or stoichiometry signature
def find_pgk_reaction(model):
    for r in model.reactions:
        if re.search(r"\bpgk\b", getattr(r, "id", ""), re.I):
            return r
    for r in model.reactions:
        nm = getattr(r, "name", None)
        if nm and re.search(r"phosphoglycerate\s*kinase", nm, re.I):
            return r
    def key(m):  # id/name to lower for matching
        return (getattr(m, "id", "") or getattr(m, "name", "") or "").lower()
    synonyms = {
        "13bpg": {"13bpg","1,3-bpg","1,3-bisphosphoglycerate","1,3-dpg","13dpg"},
        "3pg": {"3pg","3-phosphoglycerate","3-p-glycerate"},
        "adp": {"adp"},
        "atp": {"atp"},
    }
    def has_any(mset, group):
        return any(any(s in x for s in synonyms[group]) for x in mset)
    for r in model.reactions:
        mets = [key(m) for m in getattr(r, "metabolites", [])]
        if mets and has_any(mets,"adp") and has_any(mets,"atp") and has_any(mets,"13bpg") and has_any(mets,"3pg"):
            return r
    return None

def try_scale_numeric_attr(obj, names, factor):
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if isinstance(val, numbers.Number):
                setattr(obj, name, val * factor)
                return ("attr", name, val, getattr(obj, name))
    return None

def try_scale_parameter_mapping(container, names, factor):
    mapping = getattr(container, "parameters", container)
    if hasattr(mapping, "get"):
        for name in names:
            if name in mapping and isinstance(mapping[name], numbers.Number):
                old = mapping[name]
                mapping[name] = old * factor
                return ("params", name, old, mapping[name])
    return None

def kinetics_type_name(r):
    k = getattr(r, "kinetics", None)
    return type(k).__name__ if k is not None else "None"

def autoscale_for_rate_law(r, factor):
    kin = getattr(r, "kinetics", None)
    ktype = kinetics_type_name(r).lower()

    # Prefer law-specific knobs
    if "michaelis" in ktype:
        for target in (kin, r):
            hit = try_scale_numeric_attr(target, ["Vmax","vmax","Vm","Vm_f","Vm_r"], factor) or \
                  try_scale_parameter_mapping(target, ["Vmax","vmax","Vm","Vm_f","Vm_r"], factor)
            if hit: return (ktype,) + hit
        for target in (kin, r):
            hit = try_scale_numeric_attr(target, ["kcat","kcat_f","kcat_r"], factor) or \
                  try_scale_parameter_mapping(target, ["kcat","kcat_f","kcat_r"], factor)
            if hit: return (ktype,) + hit
        for target in (kin, r):
            hit = try_scale_numeric_attr(target, ["E","E_total","Et","enzyme_conc"], factor) or \
                  try_scale_parameter_mapping(target, ["E","E_total","Et","enzyme_conc"], factor)
            if hit: return (ktype,) + hit
    if "mass" in ktype and "action" in ktype:
        for target in (kin, r):
            hit = try_scale_numeric_attr(target, ["kf","k_f","k_forward","k"], factor) or \
                  try_scale_parameter_mapping(target, ["kf","k_f","k_forward","k"], factor)
            if hit: return (ktype,) + hit

    # Generic fallback
    for target in (r, kin):
        if target is None: 
            continue
        hit = try_scale_numeric_attr(target, ["vmax","Vmax","Vm","kcat","k_forward","kf","k"], factor)
        if hit: return (ktype,) + hit
        hit = try_scale_parameter_mapping(target, ["vmax","Vmax","Vm","kcat","k_forward","kf","k","Vf","Vr","Vmax_f","Vmax_r"], factor)
        if hit: return (ktype,) + hit

    return (ktype, None)

# ---- Apply mutation (keep only 25% activity) ----
pgk = find_pgk_reaction(RBC_PGK_I371T)
if pgk is None:
    raise RuntimeError("PGK reaction not found. Ensure PGK id/name or stoichiometry matches.")
print(f"PGK reaction: {getattr(pgk,'id','<no id>')} | Rate law: {kinetics_type_name(pgk)}")

scaled = autoscale_for_rate_law(pgk, 0.25)
if scaled[1] is None:
    # Show introspection hints so you can map the right knob
    attrs = [a for a in dir(pgk) if not a.startswith("_")]
    print("ERROR: No kinetic parameter scaled. Inspect your model wiring.")
    print("Reaction attributes (truncated):", attrs[:40], "...")
    if hasattr(pgk, "parameters"):
        print("Reaction parameters keys:", list(pgk.parameters.keys())[:20])
    if hasattr(pgk, "kinetics") and hasattr(pgk.kinetics, "__dict__"):
        print("Kinetics attributes:", [k for k in pgk.kinetics.__dict__.keys() if not k.startswith("_")])
    raise RuntimeError("Could not identify a PGK kinetic parameter to scale.")
else:
    ktype, where, name, old, new = scaled
    loc = "reaction" if where == "attr" else "reaction.parameters"
    print(f"Scaled [{loc}] for {getattr(pgk,'id','<no id>')} (rate law: {ktype}): {name}: {old:.4e} -> {new:.4e}")

# ---- Run I371T mutant simulation ----
print("\n" + "="*80)
print("RUNNING I371T MUTANT SIMULATION")
print("="*80)

sim_I371T = Simulation(RBC_PGK_I371T)
conc_I371T, flux_I371T = sim_I371T.simulate(RBC_PGK_I371T, time=(t0, tf))
time_I371T, ec_I371T = calculate_energy_charge(conc_I371T)

print(f"I371T energy charge: {ec_I371T[-1]:.4f}")
print(f"I371T ATP: {conc_I371T['atp_c'][-1]:.4e} mM")
print(f"I371T ADP: {conc_I371T['adp_c'][-1]:.4e} mM")
print(f"\nEnergy charge reduction: {(ec_control[-1] - ec_I371T[-1]):.4f}")
print(f"ATP reduction: {((conc_control['atp_c'][-1] - conc_I371T['atp_c'][-1]) / conc_control['atp_c'][-1] * 100):.1f}%")


# In[7]:


sim_I371T = Simulation(RBC_PGK_I371T)
conc_I371T, flux_I371T = sim_I371T.simulate(RBC_PGK_I371T, time=(t0, tf))
time_I371T, ec_I371T = calculate_energy_charge(conc_I371T)

print(f"I371T energy charge: {ec_I371T[-1]:.4f}")
print(f"I371T ATP: {conc_I371T['atp_c'][-1]:.4e} mM")
print(f"I371T ADP: {conc_I371T['adp_c'][-1]:.4e} mM")
print(f"\nEnergy charge reduction: {(ec_control[-1] - ec_I371T[-1]):.4f}")
print(f"ATP reduction: {((conc_control['atp_c'][-1] - conc_I371T['atp_c'][-1]) / conc_control['atp_c'][-1] * 100):.1f}%")


# In[13]:


BASE = RBC_PGK  # assumes your base RBC+PGK model is named RBC_PGK

# Clone and reduce PGK activity ~60% (I371T-like)
RBC_PGK_I371T = deepcopy(BASE)
pgk_param_ids = find_pgk_param_ids(RBC_PGK_I371T)

print(f"[Case 1] Found {len(pgk_param_ids)} PGK-related parameters to scale.")
for k in pgk_param_ids[:10]:
    print("  ", k)
if len(pgk_param_ids) > 10:
    print("  ...")

# Global catalytic slowdown for PGK (affects kf/kr the same way here).
_ = scale_model_parameters(RBC_PGK_I371T, pgk_param_ids, scale=0.4)  # 40% of WT activity

models = [BASE, RBC_PGK_I371T]
titles = ["Normal PGK (WT)", "Mutated PGK (I371T-like)"]


# In[18]:


# (A) Baseline steady state for both models (no perturbation)
fig1, (ax1a, ax1b) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
axs_baseline = [ax1a, ax1b]

for j, model in enumerate(models):
    sim = Simulation(model)

    # Find steady state (update_values=True to store results on the model)
    conc_ss, flux_ss = sim.find_steady_state(model, strategy="simulate", update_values=True)

    # Compute Energy Charge & PGK active fraction at steady state
    make_energy_charge(conc_ss, ratio_id="Energy_Charge")
    make_pgk_active_fraction(model, conc_ss, enzyme_id="PGK")

    # Time verification around steady-state to visualize recovery (tiny nudge)
    # Here we just simulate a short window from the SS initial state.
    conc_t, flux_t = sim.simulate(model, time=(0, 300), perturbations={})
    make_energy_charge(conc_t, ratio_id="Energy_Charge")
    make_pgk_active_fraction(model, conc_t, enzyme_id="PGK")

    # Plot 1: Energy Charge vs time
    axs_baseline[j].plot(conc_t.time, conc_t["Energy_Charge"])
    axs_baseline[j].set_title(f"{titles[j]} — Energy Charge (baseline)")
    axs_baseline[j].set_xlabel("Time")
    axs_baseline[j].set_ylabel("Energy Charge")
    axs_baseline[j].grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("./figures/Case1_baseline_EnergyCharge.png", dpi=160)


# ### Figure 1
# FIGURE RATIONALE:- 
# Shows the temporal dynamics of key PGK substrates/products and energy metabolites.
# This reveals HOW the mutation progresses from initial state to chronic steady state.
# 
# EXPECTED CLINICAL CORRELATION:
# - 1,3-DPG accumulation: Substrate backup due to slow catalysis
# - 3-PG depletion: Reduced product formation
# - ATP depletion: Matches clinical finding of low RBC ATP
# - 2,3-DPG changes: Secondary metabolic compensation
# 
# Semi-log plots capture both rapid initial changes and slow equilibration.

# In[8]:


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Case Study 1: PGK1 I371T Mutation (Reduced kcat)\nMetabolite Dynamics in Chronic Hemolytic Anemia', 
             fontsize=14, fontweight='bold')

# Plot 1: 1,3-DPG (PGK substrate)
axes[0, 0].semilogy(time_control, conc_control['_13dpg_c'], 'b-', linewidth=2.5, label='Wild-type')
axes[0, 0].semilogy(time_I371T, conc_I371T['_13dpg_c'], 'r--', linewidth=2.5, label='I371T mutant')
axes[0, 0].set_xlabel('Time [hr]', fontsize=11)
axes[0, 0].set_ylabel('Concentration [mM]', fontsize=11)
axes[0, 0].set_title('1,3-DPG (Substrate)', fontsize=12)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].annotate('Expected: Accumulation\ndue to slow catalysis', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 2: 3-PG (PGK product)
axes[0, 1].semilogy(time_control, conc_control['_3pg_c'], 'b-', linewidth=2.5, label='Wild-type')
axes[0, 1].semilogy(time_I371T, conc_I371T['_3pg_c'], 'r--', linewidth=2.5, label='I371T mutant')
axes[0, 1].set_xlabel('Time [hr]', fontsize=11)
axes[0, 1].set_ylabel('Concentration [mM]', fontsize=11)
axes[0, 1].set_title('3-PG (Product)', fontsize=12)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].annotate('Expected: Depletion\ndue to reduced flux', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 3: ATP (clinical biomarker)
axes[1, 0].semilogy(time_control, conc_control['atp_c'], 'b-', linewidth=2.5, label='Wild-type')
axes[1, 0].semilogy(time_I371T, conc_I371T['atp_c'], 'r--', linewidth=2.5, label='I371T mutant')
axes[1, 0].set_xlabel('Time [hr]', fontsize=11)
axes[1, 0].set_ylabel('Concentration [mM]', fontsize=11)
axes[1, 0].set_title('ATP (Energy Status)', fontsize=12)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].annotate('Clinical Finding:\nLow ATP in patient RBCs', 
                    xy=(0.5, 0.05), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# Plot 4: 2,3-DPG (compensatory mechanism)
axes[1, 1].semilogy(time_control, conc_control['_23dpg_c'], 'b-', linewidth=2.5, label='Wild-type')
axes[1, 1].semilogy(time_I371T, conc_I371T['_23dpg_c'], 'r--', linewidth=2.5, label='I371T mutant')
axes[1, 1].set_xlabel('Time [hr]', fontsize=11)
axes[1, 1].set_ylabel('Concentration [mM]', fontsize=11)
axes[1, 1].set_title('2,3-DPG (O₂ Delivery)', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'case1_fig1_I371T_metabolites.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure 1 saved: case1_fig1_I371T_metabolites.png")


# ### Figure 2
# FIGURE RATIONALE:- 
# Energy charge directly correlates with clinical symptoms:
# - EC > 0.8: Asymptomatic
# - EC 0.65-0.8: Chronic anemia, compensated
# - EC < 0.65: Severe anemia, hemolysis
# 
# Comparing to clinical thresholds shows if mutation is compatible with life.
# """
# 
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 

# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Case Study 1: PGK1 I371T - Energy Charge and Clinical Severity', 
             fontsize=14, fontweight='bold')

# Left panel: Time course
axes[0].plot(time_control, ec_control, 'b-', linewidth=3, label='Wild-type')
axes[0].plot(time_I371T, ec_I371T, 'r--', linewidth=3, label='I371T mutant')
axes[0].axhline(y=0.8, color='green', linestyle=':', linewidth=2, label='Healthy (>0.8)')
axes[0].axhline(y=0.65, color='orange', linestyle=':', linewidth=2, label='Chronic anemia (0.65-0.8)')
axes[0].axhline(y=0.5, color='red', linestyle=':', linewidth=2, label='Severe crisis (<0.5)')
axes[0].set_xlabel('Time [hr]', fontsize=12)
axes[0].set_ylabel('Energy Charge', fontsize=12)
axes[0].set_title('Energy Charge Dynamics', fontsize=12)
axes[0].set_ylim([0, 1.05])
axes[0].legend(fontsize=9, loc='lower left')
axes[0].grid(True, alpha=0.3)

# Add shaded regions for clinical zones
axes[0].fill_between([0, tf], 0.8, 1.0, alpha=0.1, color='green', label='_nolegend_')
axes[0].fill_between([0, tf], 0.65, 0.8, alpha=0.1, color='orange', label='_nolegend_')
axes[0].fill_between([0, tf], 0, 0.5, alpha=0.1, color='red', label='_nolegend_')

# Right panel: Steady state comparison
conditions = ['Wild-type', 'I371T\n(Reduced kcat)']
final_ec = [ec_control[-1], ec_I371T[-1]]
colors = ['blue', 'red']

bars = axes[1].bar(conditions, final_ec, color=colors, alpha=0.7, edgecolor='black', linewidth=2.5)
axes[1].axhline(y=0.8, color='green', linestyle=':', linewidth=2)
axes[1].axhline(y=0.65, color='orange', linestyle=':', linewidth=2)
axes[1].axhline(y=0.5, color='red', linestyle=':', linewidth=2)
axes[1].set_ylabel('Steady-State Energy Charge', fontsize=12)
axes[1].set_title('Clinical Severity Assessment', fontsize=12)
axes[1].set_ylim([0, 1.05])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels and clinical interpretation
for bar, val in zip(bars, final_ec):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Clinical interpretation
    if val > 0.8:
        status = 'Normal'
    elif val > 0.65:
        status = 'Chronic\nAnemia'
    else:
        status = 'Severe\nDeficiency'
    axes[1].text(bar.get_x() + bar.get_width()/2., 0.1,
                status, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'case1_fig2_I371T_energy_charge.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Figure 2 saved: case1_fig2_I371T_energy_charge.png")


# ### Figure 3
# FIGURE RATIONALE:- 
# ATP/ADP ratio determines:
# - Thermodynamic driving force for ATP-dependent processes
# - Cellular work capacity
# - Ion pump efficiency (Na+/K+-ATPase critical in RBCs)
# 
# Phase portrait shows trajectory through metabolic state space.
# Clinical correlation: Low ATP/ADP = impaired ion homeostasis = hemolysis
# 

# In[10]:


fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ATP/ADP ratios
atp_adp_control = conc_control['atp_c'] / conc_control['adp_c']
atp_adp_I371T = conc_I371T['atp_c'] / conc_I371T['adp_c']

# Control trajectory
ax.plot(conc_control['atp_c'], conc_control['adp_c'], 
        'b-', linewidth=2.5, alpha=0.8, label='Wild-type')
ax.plot(conc_control['atp_c'][0], conc_control['adp_c'][0], 
        'bo', markersize=12, label='Initial state', zorder=5)
ax.plot(conc_control['atp_c'][-1], conc_control['adp_c'][-1], 
        'bs', markersize=12, label='Wild-type SS', zorder=5)

# I371T trajectory
ax.plot(conc_I371T['atp_c'], conc_I371T['adp_c'], 
        'r--', linewidth=2.5, alpha=0.8, label='I371T mutant')
ax.plot(conc_I371T['atp_c'][-1], conc_I371T['adp_c'][-1], 
        'rs', markersize=12, label='I371T SS', zorder=5)

ax.set_xlabel('ATP Concentration [mM]', fontsize=13, fontweight='bold')
ax.set_ylabel('ADP Concentration [mM]', fontsize=13, fontweight='bold')
ax.set_title('Case Study 1: PGK1 I371T Mutation\nPhase Portrait: ATP vs ADP (Metabolic State Space)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# Add ATP/ADP ratio annotations
ax.text(0.05, 0.95, f'Wild-type ATP/ADP: {atp_adp_control[-1]:.2f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.text(0.05, 0.88, f'I371T ATP/ADP: {atp_adp_I371T[-1]:.2f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
ax.text(0.05, 0.78, 'Clinical Impact:\nReduced ATP/ADP impairs\nion pump function',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'case1_fig3_I371T_phase_portrait.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Figure 3 saved: case1_fig3_I371T_phase_portrait.png")


# ### Summary

# In[11]:


# Summary output
print("\n" + "="*80)
print("CASE STUDY 1 COMPLETE: PGK1 I371T MUTATION")
print("="*80)
print(f"All figures saved to: {figures_dir}")
print("\nKEY FINDINGS:")
print(f"1. Wild-type energy charge: {ec_control[-1]:.4f} (healthy)")
print(f"2. I371T mutant energy charge: {ec_I371T[-1]:.4f}")
print(f"3. Energy charge reduction: {(ec_control[-1] - ec_I371T[-1]):.4f}")
print(f"4. ATP reduction: {((conc_control['atp_c'][-1] - conc_I371T['atp_c'][-1]) / conc_control['atp_c'][-1] * 100):.1f}%")
print(f"5. ATP/ADP ratio: WT = {atp_adp_control[-1]:.2f}, I371T = {atp_adp_I371T[-1]:.2f}")
print(f"\nCLINICAL CORRELATION:")
if ec_I371T[-1] > 0.8:
    print("   ✓ Asymptomatic or mild symptoms")
elif ec_I371T[-1] > 0.65:
    print("   ⚠ Chronic hemolytic anemia (matches clinical case)")
else:
    print("   ✗ Severe deficiency, life-threatening")
print(f"\nREFERENCE: Kugler W, et al. Blood 1998. DOI: 10.1182/blood.V91.2.504.504")

