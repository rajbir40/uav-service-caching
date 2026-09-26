# Configuration constants for the UAV-MEC simulator

# Environment
AREA_RADIUS = 1000.0
NUM_USERS = 20
MAX_NUM_USERS = 50
NUM_UAVS = 5
MAX_UAVS = 10
MIN_UAVS = 2

# UAV Profiles
UAV_PROFILES = [
    {"cpu_freq": 2e9, "bandwidth": 10e6, "tx_power_dbm": 27, "max_displacement": 50.0, "altitude": 120.0, "cache_capacity": 100.0},
    {"cpu_freq": 2.5e9, "bandwidth": 15e6, "tx_power_dbm": 27, "max_displacement": 50.0, "altitude": 120.0, "cache_capacity": 100.0},
]

# Communication
IOT_TX_POWER = 20.0  # dBm
NOISE_PSD = 1e-10
UAV_PROPULSION_CONSTANTS = {"c1": 10.0, "c2": 0.01}  # Constants for propulsion energy

# Task Parameters
TASK_DATA_SIZE_RANGE = (1000000, 10000000)  # bits
TASK_CPU_CYCLES_RANGE = (1000000, 10000000)  # cycles
TASK_DEADLINE_RANGE = (10.0, 100.0)  # seconds
TASK_GENERATION_PROB = 0.3

# Simulation
NUM_TIME_SLOTS = 100
SLOT_DURATION = 1.0  # seconds
UAV_ALT_MIN = 10.0
UAV_ALT_MAX = 120.0
UAV_ALT_DELTA_MAX = 20.0

# Phase 1 Constants
PHASE1_OBS_USERS = 5
PHASE1_UAV_BATTERY_J = 10000.0
PHASE1_LAT_NORM = 10.0
PHASE1_ENERGY_NORM = 1000.0
PHASE1_UAV_RX_POWER_W = 1.0
PHASE1_ACTION_DIM = 5

# Phase 3 Constants
NUM_SERVICES = 10
SERVICE_SIZE_RANGE = (20.0, 50.0)  # bits
DEFAULT_SERVICE_CATALOG = [f"service_{i}" for i in range(1, NUM_SERVICES + 1)]

# Communication Bandwidths
IOT_UAV_BANDWIDTH = 10e6  # bits/s
UAV_UAV_BANDWIDTH = 50e6  # bits/s
UAV_ANCHOR_BANDWIDTH = 100e6  # bits/s

# Constants for reachability
MIN_UAV_SEPARATION = 10.0
REACHABLE_DISTANCE = 50.0

# Energy Constants
UAV_PROPULSION_C1 = 10.0
UAV_PROPULSION_C2 = 0.01
UAV_TX_POWER = 0.1  # Watts
UAV_RX_POWER = 0.1  # Watts
COMPUTATION_ENERGY_CONSTANT = 1e-9  # Energy per CPU cycle
