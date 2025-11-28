# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 03:37:18 2025

@author: DenizTuna
"""

# -*- coding: utf-8 -*-
"""
Python implementation for AHP-based performance evaluation.

This script calculates:
- Ratio matrices for performance metrics (FGY, BLK, DRB)
- Eigenvector-based priorities
- Consistency indices
- Criteria weights
- Final alternative weights
- Ranking visualization

Dependencies:
    numpy
    pandas
    matplotlib
    pyanp (AHP calculations)
"""

import numpy as np
import pandas as pd
from pyanp import priority
import matplotlib.pyplot as plt


# ------------------------------
# 1. INPUT DATA
# ------------------------------

# Performance metric arrays (example: players' statistics)
FGY = np.array([.452	,.442	,.472	,.463	,.443	,.423	,.451	,.668	,.483	,.650	,.361	,.355	,.245	,.356	,.415	,.353	,.353	,.222])

BLK = np.array([4.0,2.6	,1.8	,3.8	,2.0	,1.4	,0.7	,1.1	2,.2	,2.3	,2.1,	1.6	,3.0,	2.0	,2.2	,1.4	,7.0,	0.1])

DRB = np.array([6.4,4.0,6.8,7.2,6.4,4.4,4.9,6.1,8.3,2.4,
                13.2,4.2,4.2,3.8,3.6,10.2,5.2,5.9,4.4,3.9,
                9.2,8.8,6.6,5.2,10.6,5.8,6.7,11.9,8.7,2.9])


# ------------------------------
# 2. RATIO MATRICES
# ------------------------------

FGY_ratio = np.array([FGY[i] / FGY for i in range(len(FGY))])
BLK_ratio = np.array([BLK[i] / BLK for i in range(len(BLK))])
DRB_ratio = np.array([DRB[i] / DRB for i in range(len(DRB))])

print("FGY Ratio Matrix:\n", FGY_ratio)
print("BLK Ratio Matrix:\n", BLK_ratio)
print("DRB Ratio Matrix:\n", DRB_ratio)


# ------------------------------
# 3. AHP: PRIORITIES + CONSISTENCY
# ------------------------------

FGY_pri = priority.pri_eigen(FGY_ratio)
BLK_pri = priority.pri_eigen(BLK_ratio)
DRB_pri = priority.pri_eigen(DRB_ratio)

FGY_eig = priority.pri_eigen(FGY_ratio, return_eigenval=True)
BLK_eig = priority.pri_eigen(BLK_ratio, return_eigenval=True)
DRB_eig = priority.pri_eigen(DRB_ratio, return_eigenval=True)

FGY_cons = priority.incon_std(FGY_ratio)
BLK_cons = priority.incon_std(BLK_ratio)
DRB_cons = priority.incon_std(DRB_ratio)

print("FGY Priority Vector:", FGY_pri)
print("BLK Priority Vector:", BLK_pri)
print("DRB Priority Vector:", DRB_pri)

print("FGY Consistency:", FGY_cons)
print("BLK Consistency:", BLK_cons)
print("DRB Consistency:", DRB_cons)


# Original eigenvector calculations (optional)
FGY_exp = priority.pri_expeigen(FGY_ratio)
BLK_exp = priority.pri_expeigen(BLK_ratio)
DRB_exp = priority.pri_expeigen(DRB_ratio)

# ------------------------------
# 4. CRITERIA WEIGHTS
# ------------------------------

criteria = np.array([0.906, 0.482, 0.165])
criteria_ratio = np.array([criteria[i] / criteria for i in range(len(criteria))])

criteria_pri = priority.pri_eigen(criteria_ratio)
criteria_exp = priority.pri_expeigen(criteria_ratio)

print("Criteria Weights:", criteria_pri)


# ------------------------------
# 5. FINAL WEIGHT CALCULATION
# ------------------------------

# Stack alternatives
alternative_matrix = np.vstack((FGY_exp, BLK_exp, DRB_exp))

# Weighted matrix
weighted_matrix = alternative_matrix.T * criteria_exp.T

# Final weights per alternative
final_weights = np.sum(weighted_matrix.T, axis=0)
print("Final Weights:", final_weights)

# Ranking
ranking = np.sort(final_weights)
print("Ranking:", ranking)


# ------------------------------
# 6. VISUALIZATION
# ------------------------------

labels = ['Pete Nance','Jared McCain','Reggie Jackson','Lester Quiñones',
          'Ricky Council IV','Kyle Lowry','David Roddy','Tyrese Maxey',
          'Lonnie Walker IV','Jared Butler','Eric Gordon',
          'Jalen Hood-Schifino','Justin Edwards','Paul George',
          'Quentin Grimes','Kelly Oubre Jr.','Guerschon Yabusele',
          'Caleb Martin','Phillip Wheeler','Colin Castleton',
          'Jeff Dowtin Jr.','Oshae Brissett','Chuma Okeke',
          'Joel Embiid','Andre Drummond','Marcus Bagley','KJ Martin',
          'Isaiah Mobley','Alex Reese','Adem Bona']

plt.figure(figsize=(6,6))
y = ranking
x_pos = np.arange(len(y))

plt.barh(x_pos, y, color='steelblue')
plt.title("Philadelphia Performance Graph")
plt.yticks(x_pos, labels)
plt.xlabel("Performance Weights")
plt.ylabel("Players Sorted by Weight")
plt.tight_layout()
plt.show()
