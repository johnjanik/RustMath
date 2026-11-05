"""Config + dd pin table for the ACHIRAL (3B,4C,3A) member of M24 (triangle group [3,4,3]).

Permutations (0-based, original labels):
  s0 = [4,5,18,20,16,14,23,6,11,19,13,15,2,22,1,8,0,3,12,21,17,9,10,7]   (3B: 3^8)
  s1 = [16,0,5,17,8,7,11,12,22,21,4,14,2,20,23,1,15,13,19,9,3,18,10,6]   (4C: 4^6)
Passport 3^8 | 4^6 | 3^6 1^6 (P = A^3 deg 24; P-Q = c W^4, W deg 6; Q = lam R^3 S, R deg 6,
S deg 6 with the six ELLIPTIC simple poles). Atlas: M_BASE=1 (cell fully ordinary), dim S_4 = 3,
top-echelon x = f2/(f1 + c f2), osculating law w^3.
All addresses in the a-chart coordinate X (triple zero at X = 0 by gauge).
"""
import numpy as np

SW = "/home/john/sweep_2_12_5/"
S0 = "4,5,18,20,16,14,23,6,11,19,13,15,2,22,1,8,0,3,12,21,17,9,10,7"
S1 = "16,0,5,17,8,7,11,12,22,21,4,14,2,20,23,1,15,13,19,9,3,18,10,6"
ABC = "3,4,3"

# dd-measured addresses (glue residuals 1e-16..6e-14)
W_PINS = [                                        # quadruple points (phi = 1), 4 of 6
    complex(+0.400341547059, -0.098184224177),    # base z_b
    complex(+0.444237652730, -0.304128490193),    # b2c4
    complex(-0.188097154190, +0.135378744904),    # b2c2
    complex(-0.471432223729, -0.201019679041),    # b2c6
]
R_PINS = [                                        # triple poles, 4 of 6
    complex(+0.232094603977, -0.314965262827),    # base z_c
    complex(-0.216637078115, +0.062581777007),    # c2c2
    complex(+0.009025424545, +0.582883734323),    # c2c5
    complex(+0.478881209542, -0.260250686131),    # c2c8
]
A_PINS = [                                        # triple zeros, 4 of 8
    0j,                                           # base z_a (gauge)
    complex(+0.721315062081, -0.483215147529),    # a2c8
    complex(-0.387140556605, +0.134483201485),    # a2c6
    complex(-0.197719727053, +0.109409272976),    # a2c2
]
S_PINS = [                                        # elliptic simple poles, 1 of 6
    complex(-0.459325548653, +0.049536776532),    # c2c23 (mod-3 chart, first elliptic glue)
]

RHO = {'a': "0.961159943109551552904035635039522544383276376",
       'b': "0.954694800766566687850287836332568990452182213",
       'c': "0.948931794846869339001047843668148362966505530"}
