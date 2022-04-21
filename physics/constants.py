"""A library of commonly used physical constants."""

# Universal gas constant, in units of J/mol/K.
R_UNIVERSAL = 8.3145

# The precomputed gas constant for dry air, in units of J/kg/K.
R_D = 286.69

# The gravitational acceleration constant, in units of N/kg.
G = 9.81

# The heat capacity ratio of dry air, dimensionless.
GAMMA = 1.4

# The constant pressure heat capacity of dry air, in units of J/kg/K.
CP = GAMMA * R_D / (GAMMA - 1.0)

# The constant volume heat capacity of dry air, in units of J/kg/K.
CP = CP - R_D
