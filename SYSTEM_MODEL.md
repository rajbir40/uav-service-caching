# SYSTEM_MODEL.md
# Refresh-Persistent Cooperative Service Replication for Spatially De-Coupled,
# Tail-Latency-Aware Multi-UAV MEC

> **Purpose**
>
> This document is the implementation-facing source of truth for the UAV-MEC
> system model. Claude/code agents MUST use it as the reference when modifying
> or extending the simulator.
>
> The mathematical paper may contain additional notation or presentation
> details, but implementation behavior MUST remain consistent with the rules
> below. If code requirements conflict with this document, STOP and flag the
> conflict instead of silently changing the model.

---

## 1. Research Direction

### Locked title

**Refresh-Persistent Cooperative Service Replication for Spatially De-Coupled,
Tail-Latency-Aware Multi-UAV MEC**

### Core research idea

Multiple UAVs cooperatively maintain persistent replicas of atomic services.
Replication is jointly coordinated with UAV trajectory, task association/task
migration, CPU allocation, and UAV resource constraints.

The purpose of replication is to reduce the dependence between:

```text
where a UAV is physically located
        ↓
where the required service can be obtained
        ↓
communication distance
        ↓
task latency
```

This allows UAVs to respond to dynamically changing service/task hotspots
without requiring every UAV to physically move toward every demand hotspot.

Tail latency is a major evaluation/objective focus, using metrics such as
CVaR, P95, and P99.

---

# 2. NON-NEGOTIABLE SYSTEM ASSUMPTIONS

The implementation MUST preserve these assumptions.

## 2.1 Entities

The system contains:

- `N` stationary IoT devices.
- `M` cooperating UAVs.
- Exactly one fixed anchor/registry node.
- One fixed atomic-service catalog.

## 2.2 IoT devices

IoT devices are **stationary**.

Their physical positions NEVER change.

Dynamicity comes from:

- devices becoming active/inactive;
- task arrivals;
- changing requested services;
- changing spatial demand/hotspots.

Do NOT implement IoT mobility.

## 2.3 Services

Services are:

- atomic;
- self-contained;
- independent;
- identified by a service ID.

There are NO service chains and NO inter-service dependencies.

## 2.4 Replication

Service replicas are:

- persistent;
- binary;
- distributed across UAVs;
- fleet-coordinated;
- constrained by UAV cache capacity.

A replica remains present until an explicit insertion/refresh or eviction
operation changes its state.

## 2.5 Energy

Only UAV energy is modeled.

There is:

- NO IoT battery model;
- NO IoT energy harvesting;
- NO WPT;
- NO charging station;
- NO UAV recharge mechanism.

UAV energy is finite and non-replenished.

## 2.6 UAV mobility

UAVs are mobile.

Altitude is fixed.

Only horizontal movement is a decision.

---

# 3. CORE SETS AND INDICES

Use the following conceptual sets.

```text
N = {1, ..., N}       IoT devices
M = {1, ..., M}       UAVs
S = {1, ..., S}       atomic services
T = {1, ..., T}       time slots
A = anchor node
```

Recommended implementation names:

```python
num_devices
num_uavs
num_services
num_slots

device_ids
uav_ids
service_ids
```

Avoid introducing multiple aliases for the same entity.

---

# 4. TIME MODEL

The environment is discrete-time slotted.

For:

```text
t ∈ {1, ..., T}
```

each slot has fixed duration:

```text
Δt
```

Recommended implementation:

```python
slot_duration
current_slot
```

All trajectory, demand, replication, resource, latency, and energy state
updates must be synchronized to the slot boundary.

---

# 5. PHYSICAL LOCATIONS

## 5.1 IoT device position

Each device has a fixed position:

```text
p_n ∈ R^3
```

Implementation:

```python
device_position[n]
```

This value must remain unchanged during an episode.

## 5.2 Anchor position

The anchor has a fixed position:

```text
p_A ∈ R^3
```

Implementation:

```python
anchor_position
```

## 5.3 UAV position

UAV `m` has a time-dependent position:

```text
p_m,t ∈ R^3
```

with fixed altitude.

The horizontal velocity is:

```text
v_m,t ∈ R^2
```

Trajectory:

```text
p_m,t+1 = p_m,t + v_m,t * Δt
```

with:

```text
||v_m,t|| <= v_max
```

Also enforce:

```text
p_m,t ∈ R
```

and minimum UAV separation:

```text
||p_m,t - p_m',t|| >= D_min
```

for every distinct UAV pair.

Recommended implementation:

```python
uav_position[m]
uav_velocity[m]
max_uav_speed
min_uav_separation
```

Do NOT introduce acceleration dynamics unless explicitly added later as a
separate research decision.

---

# 6. DISTANCES

The implementation must provide the following geometric distances.

## IoT → UAV

```text
d_n,m(t) = ||p_m,t - p_n||
```

Implementation:

```python
distance_device_uav(n, m)
```

## UAV → UAV

```text
d_m,m'(t) = ||p_m,t - p_m',t||
```

Implementation:

```python
distance_uav_uav(m, m_prime)
```

## UAV → Anchor

```text
d_m,A(t) = ||p_m,t - p_A||
```

Implementation:

```python
distance_uav_anchor(m)
```

These distances are the main mechanism connecting trajectory to communication
latency.

---

# 7. DYNAMIC TASK MODEL

Each device has an active/inactive state:

```text
a_n(t) ∈ {0,1}
```

where:

```text
a_n(t) = 1
```

means the device currently has an unresolved task.

Active devices:

```text
N_active(t) = {n : a_n(t) = 1}
```

Implementation:

```python
device_active[n]
active_devices
```

The exact task-arrival generation process is part of the simulator configuration.
Do not silently replace the frozen demand process with a different stochastic
model.

---

# 8. TASK REPRESENTATION

When device `n` is active:

```text
τ_n(t) = (D_n(t), C_n, ψ(n,t))
```

where:

- `D_n(t)` = task input size in bits.
- `C_n` = CPU cycles per input bit.
- `ψ(n,t)` = required atomic service ID.

Implementation:

```python
task.input_size_bits
task.cycles_per_bit
task.service_id
```

Important:

```text
service_id != task_id
```

A service is reusable infrastructure state; a task is a transient workload.

---

# 9. UAV RESOURCE STATE

Each UAV has:

## CPU

```text
W_m
```

in Hz.

Implementation:

```python
uav_cpu_capacity[m]
```

## Service cache

```text
Z_m
```

Implementation:

```python
uav_cache_capacity[m]
```

## Energy

```text
E_m(t)
```

with:

```text
0 <= E_m(t) <= E_m_max
```

Implementation:

```python
uav_energy[m]
uav_max_energy[m]
```

---

# 10. SERVICE CATALOG

The service catalog is:

```text
S = {1, ..., S}
```

Each service `s` has storage footprint:

```text
z_s
```

Implementation:

```python
service_size[s]
```

All services are atomic.

The anchor stores the entire catalog.

Therefore:

```text
anchor_has_service[s] = True
```

for every service.

---

# 11. PERSISTENT SERVICE REPLICATION

## 11.1 Replica state

For every UAV/service pair:

```text
ρ_m,s,t ∈ {0,1}
```

where:

```text
ρ_m,s,t = 1
```

means UAV `m` persistently stores service `s`.

Implementation:

```python
replica[m, s]
```

or:

```python
uav_cache[m][s]
```

Use ONE canonical representation in the simulator.

## 11.2 Cache capacity

Every UAV must satisfy:

```text
Σ_s ρ_m,s,t * z_s <= Z_m
```

Implementation invariant:

```python
sum(service_size[s] for s in cache[m]) <= cache_capacity[m]
```

This invariant MUST be checked after every replication state transition.

---

# 12. REPLICATION STATE TRANSITION

Use explicit insertion and eviction events.

```text
I_m,s,t ∈ {0,1}
E_m,s,t ∈ {0,1}
```

State transition:

```text
ρ_m,s,t = ρ_m,s,t-1 + I_m,s,t - E_m,s,t
```

with:

```text
I_m,s,t <= 1 - ρ_m,s,t-1
E_m,s,t <= ρ_m,s,t-1
```

Therefore:

- inserting an already cached service is invalid;
- evicting a non-cached service is invalid;
- a service persists if neither event occurs.

Implementation semantics:

```python
if insert:
    cache.add(service)
elif evict:
    cache.remove(service)
else:
    cache remains unchanged
```

Do not rebuild the cache randomly every slot.

---

# 13. REPLICATION VS TASK MIGRATION

This distinction is CRITICAL.

## Service replication

Replication determines:

```text
which UAVs hold which services
```

and is represented by:

```text
ρ_m,s,t
I_m,s,t
E_m,s,t
```

Replication is persistent infrastructure state.

## Task migration

Task migration determines:

```text
where the task input is executed
```

when the associated UAV does not locally hold the required service but a
reachable peer does.

Task migration is represented conceptually by:

```text
μ_n,m,m',t ∈ {0,1}
```

Task migration is NOT service replication.

Do not merge these concepts in code.

---

# 14. TASK ASSOCIATION

Every active task must be associated with exactly one UAV.

```text
α_n,m,t ∈ {0,1}
```

and:

```text
Σ_m α_n,m,t = 1
```

for every active device.

Implementation:

```python
task.associated_uav
```

or an equivalent one-hot representation.

Task dropping is NOT part of this model.

Therefore, do not silently allow:

```text
sum(alpha) = 0
```

for an active task.

The simulator assumes the operating region/fleet configuration allows every
active device to reach at least one UAV.

---

# 15. THREE SERVICE-ACCESS PATHS

For active task `n`, let:

```text
m = associated_uav(n)
s = task.service_id
```

There are exactly three conceptual paths.

## 15.1 LOCAL

Condition:

```text
replica[m, s] == 1
```

The associated UAV executes the task.

No UAV-UAV transfer.

No anchor transfer.

Execution UAV:

```text
m*
= m
```

## 15.2 PEER

Condition:

```text
replica[m, s] == 0
```

and another reachable UAV `m'` has:

```text
replica[m', s] == 1
```

The task input itself is migrated:

```text
D_n(t)
```

from `m` to `m'`.

The peer executes the task using its existing persistent service replica.

Execution UAV:

```text
m* = m'
```

Peer transfer latency:

```text
T_peer = D_n(t) / r_m,m'(t)
```

Important:

The service is NOT transferred during peer task migration.

The peer already owns the service replica.

## 15.3 ANCHOR

If:

- associated UAV has no required replica, and
- no usable peer replica is selected,

the service is acquired from the anchor.

The service transfer size is:

```text
z_s
```

and:

```text
T_anchor = z_s / r_m,A(t)
```

The associated UAV then executes the task.

Execution UAV:

```text
m* = m
```

---

# 16. ANCHOR ACQUISITION AND PERSISTENT INSERTION

An anchor transfer and persistent insertion are related but distinct concepts.

### Temporary acquisition

A UAV can obtain service `s` from the anchor for immediate task execution
without retaining it.

Then:

```text
replica[m, s] remains 0
```

after the task.

### Persistent insertion

If:

```text
I_m,s,t = 1
```

the UAV obtains/installs service `s` as a persistent replica.

The implementation must ensure the service transfer required for insertion is
not accidentally counted twice.

### IMPORTANT IMPLEMENTATION RULE

If an anchor-path task and a persistent insertion for the same `(m,s,t)` use
the same service transfer, model it as ONE physical `z_s` transfer.

Do NOT charge:

```text
z_s transfer for task acquisition
+
z_s transfer for insertion
```

unless a later explicit model decision intentionally defines two separate
transfers.

---

# 17. REACHABLE PEERS

Define a reachable-peer set:

```text
M_m(t) =
{m' != m : d_m,m'(t) <= D_reach}
```

Implementation:

```python
reachable_peers(m)
```

`D_reach` is a simulation/configuration parameter.

A peer migration is valid only when:

```text
m' in reachable_peers(m)
```

and:

```text
replica[m'][service_id] == True
```

---

# 18. COMMUNICATION MODEL

Use the common free-space + Shannon model.

For endpoints `X,Y`:

```text
g_XY(t) = g0 * d_XY(t)^(-2)
```

and:

```text
r_XY(t) =
B_XY * log2(1 + P_X * g_XY(t) / σ²)
```

Parameters:

```text
g0
B_XY
P_X
σ²
```

Three link types:

### IoT → UAV

```text
r_n,m(t)
```

using:

```text
d_n,m(t)
B_IU
P_n
```

### UAV → UAV

```text
r_m,m'(t)
```

using:

```text
d_m,m'(t)
B_UU
P_U
```

### UAV → Anchor

```text
r_m,A(t)
```

using:

```text
d_m,A(t)
B_UA
P_U
```

The UAV-UAV rate is an adaptation of the common communication template.

Do not implement P2's specific bandwidth-allocation/Lyapunov formulation unless
explicitly required later.

---

# 19. COMPUTATION MODEL

Each UAV allocates CPU capacity among tasks it executes.

For task `n` executed by UAV `m`:

```text
f_n,m,t = φ_n,m,t * W_m
```

where:

```text
0 <= φ_n,m,t <= 1
```

and:

```text
Σ_{n executed by m} φ_n,m,t <= 1
```

Therefore:

```text
Σ f_n,m,t <= W_m
```

Execution latency:

```text
T_comp_n(t) =
C_n * D_n(t) / f_n,m*(t)
```

where `m*` is the actual execution UAV.

Do not allocate CPU to the associated UAV automatically if the task was
migrated to a peer.

CPU allocation must follow the actual execution UAV.

---

# 20. END-TO-END LATENCY

For task `n`:

```text
L_n(t) =
T_up_n(t)
+
T_acq_n(t)
+
T_queue_n(t)
+
T_comp_n(t)
```

Each component must occur exactly once.

## 20.1 Uplink

Always:

```text
T_up_n(t) =
D_n(t) / r_n,m(n,t)(t)
```

The task first uploads its input to its associated UAV.

## 20.2 Acquisition/migration

### Local

```text
T_acq = 0
```

### Peer

```text
T_acq =
D_n(t) / r_m,m'(t)
```

because the task input is migrated.

### Anchor

```text
T_acq =
z_s / r_m,A(t)
```

because the required service is obtained from the anchor.

## 20.3 Queue

Queueing delay is part of end-to-end latency, but the exact queue/backlog
model is intentionally NOT hard-coded in this system-model document.

Implementation MUST keep a clean abstraction:

```python
queue_delay(task, execution_uav)
```

Do not invent a new queueing-theory model unless it is explicitly specified
by the optimization/design document.

## 20.4 Computation

```text
T_comp =
C_n * D_n / f
```

at the actual execution UAV.

---

# 21. TAIL-LATENCY DATA

At each slot:

```text
L_t =
{L_n(t) : n in N_active(t)}
```

This is the latency sample population for later:

- P95;
- P99;
- CVaR.

Because the active population changes over time, the implementation must NOT
assume a fixed number of active tasks.

Recommended metric interface:

```python
latencies = completed_task_latencies
p95 = percentile(latencies, 95)
p99 = percentile(latencies, 99)
cvar = compute_cvar(latencies, alpha)
```

The exact aggregation across slots belongs to the training/evaluation layer.

---

# 22. SPATIAL DE-COUPLING MECHANISM

This is the central research mechanism.

```text
Persistent replication state
          ↓
Service-access path
          ↓
Relevant communication distance
          ↓
Acquisition/migration latency
```

Specifically:

### Local

Relevant additional service-access distance:

```text
IoT → UAV
```

### Peer

Relevant additional service-access distance:

```text
UAV → UAV
```

### Anchor

Relevant additional service-access distance:

```text
UAV → Anchor
```

Therefore, replication changes which spatial relationship determines service
access.

This is the key reason replication and UAV trajectory must be optimized
jointly.

---

# 23. UAV ENERGY MODEL

Each UAV has a finite, non-replenished energy budget.

State:

```text
0 <= E_m(t) <= E_m_max
```

Per-slot feasibility:

```text
E_slot_m(t) <= E_m(t)
```

State transition:

```text
E_m(t+1) =
E_m(t) - E_slot_m(t)
```

Initial condition:

```text
E_m(0) = E_m_max
```

No recharge occurs.

---

# 24. ENERGY DECOMPOSITION

Per-slot UAV energy:

```text
E_slot_m =
E_prop_m
+
E_comm_m
+
E_comp_m
```

## 24.1 Propulsion

Use:

```text
E_prop_m(t) =
c1 + c2 * ||v_m,t||²
```

where `c1,c2` are UAV/aerodynamic constants.

Do NOT import the charging/recharging term from P1.

## 24.2 Communication

Communication energy is:

```text
transmit_power × transmission_time
```

Peer migration:

```text
E_peer =
P_U * D_n / r_m,m'
```

Anchor service acquisition:

```text
E_anchor =
P_U * z_s / r_m,A
```

Only UAV-initiated transfers are charged to the UAV's communication energy.

If a transfer is shared between immediate anchor acquisition and persistent
insertion, count the physical transfer once.

## 24.3 Computation

For task `n` executed by UAV `m`:

```text
E_comp =
k_m * f_n,m^ν * T_comp_n
```

Total:

```text
E_comp_m =
Σ_{n executed by m}
k_m * f_n,m^ν * T_comp_n
```

---

# 25. IMPLEMENTATION STATE

A minimal environment state should contain conceptually:

```python
state = {
    "slot": t,

    "device_positions": ...,
    "device_active": ...,

    "tasks": ...,
    "requested_services": ...,

    "uav_positions": ...,
    "uav_velocities": ...,

    "uav_energy": ...,

    "service_replicas": ...,

    "queue_state": ...,
}
```

The exact class/dictionary structure may differ, but the simulator must preserve
these semantic state variables.

---

# 26. ACTION CATEGORIES

The eventual control/action space will contain decisions related to:

1. UAV movement/trajectory.
2. Task association.
3. Peer migration/service-access choice.
4. CPU allocation.
5. Service insertion/replication.
6. Service eviction.
7. Potentially other resource controls explicitly introduced later.

Do NOT implement neural-network architecture assumptions in the environment
model.

Attention, action masking, MADRL, actor/critic structure, and reward design
belong to the training layer.

---

# 27. ACTION VALIDITY / MASKING REQUIREMENTS

The implementation should expose enough information to mask invalid actions.

Examples:

### Invalid association

An active task cannot have zero serving UAVs.

### Invalid peer migration

Cannot migrate to:

- itself;
- unreachable UAV;
- UAV without the required service.

### Invalid insertion

Cannot insert a service that already exists.

Cannot exceed cache capacity.

### Invalid eviction

Cannot evict a service that is not cached.

### Invalid movement

Cannot violate:

- maximum speed;
- operating region;
- minimum UAV separation.

### Invalid energy-consuming action

Cannot cause:

```text
E_slot_m > E_m(t)
```

Action masking may be implemented in the RL layer, but the environment must
also validate actions so invalid state transitions cannot silently occur.

---

# 28. REPLICATION COST / CHURN

Every insertion or eviction represents a replication-state change.

Define conceptual churn:

```text
C_rep =
Σ_{m,s}
(I_m,s,t + E_m,s,t) * c_m,s
```

The exact coefficient/weight belongs to the optimization/reward specification.

Do NOT silently invent a new replication-cost function in the simulator if the
training configuration already defines one.

---

# 29. WHAT MUST NOT BE IMPLEMENTED

Unless explicitly changed in a future research specification, do NOT add:

- IoT mobility.
- IoT battery constraints.
- IoT energy harvesting.
- WPT.
- Charging stations.
- UAV recharge cycles.
- Service chains.
- Inter-service dependencies.
- Automatic cache reset every slot.
- Task migration as a substitute for replication state.
- Duplicate charging of the same physical service transfer.
- Unsupported queueing models.
- P2-specific Lyapunov optimization.
- P1 charging equations.
- P4 laser/WPT mechanisms.

---

# 30. SOURCE / MODEL TRACEABILITY

The system model is grounded in the four base papers, but not every equation is
directly copied.

Conceptual mapping:

| Component | Main basis | Status |
|---|---|---|
| UAV trajectory/mobility | P1/P2/P3 | adapted |
| UAV propulsion energy | P1 | adapted |
| IoT-UAV communication | P3/P1 | adapted |
| UAV-UAV communication | P2 + P3 template | adapted |
| UAV-anchor communication | P3 | adapted |
| CPU computation | P1/P2/P3 | adapted |
| Cache capacity | P1/P3 | adapted |
| Atomic service replication | Research direction | new |
| Persistent replica state | Research direction | new |
| Fleet-wide replication | Research direction | new |
| Task migration | P2 | adapted |
| Tail latency/CVaR focus | Research direction | new |
| Spatial de-coupling mechanism | Research direction | new |
| No-WPT energy model | Research-specific constraint | deliberate exclusion |

P1/P2/P3/P4 are technical foundations, not permission to import their entire
system models.

---

# 31. IMPLEMENTATION INVARIANTS

The environment should validate these invariants after every step.

```text
1. Device positions do not change.

2. Every active task has exactly one associated UAV.

3. Every replica state is binary.

4. Cache capacity is never exceeded.

5. Insertion is only possible for a missing service.

6. Eviction is only possible for an existing service.

7. UAV positions remain inside the operating region.

8. UAV speed never exceeds v_max.

9. UAV separation never falls below D_min.

10. Peer migration only targets reachable UAVs.

11. Peer migration only targets UAVs holding the required service.

12. Computation occurs at the actual execution UAV.

13. CPU allocation never exceeds UAV capacity.

14. UAV energy never becomes negative.

15. Energy is never replenished.

16. No IoT energy state exists.

17. No service chain dependency exists.

18. The same physical communication transfer is not charged twice.

19. Queue delay is counted once.

20. Computation delay is counted once.

21. Tail-latency metrics operate on actual task latency samples.
```

---

# 32. RECOMMENDED ENVIRONMENT API

The implementation should expose functionality equivalent to:

```python
reset()

step(action)

get_active_tasks()

associate_task(task, uav)

get_required_service(task)

has_service(uav, service)

get_reachable_peers(uav)

select_execution_uav(task)

allocate_cpu(task, uav)

insert_service(uav, service)

evict_service(uav, service)

compute_distance_device_uav(device, uav)

compute_distance_uav_uav(uav_a, uav_b)

compute_distance_uav_anchor(uav)

compute_rate_device_uav(device, uav)

compute_rate_uav_uav(uav_a, uav_b)

compute_rate_uav_anchor(uav)

compute_task_latency(task)

compute_task_energy(task)

compute_uav_energy(uav)

compute_tail_latency_metrics()

validate_state()
```

These are semantic recommendations, not mandatory function names.

---

# 33. IMPLEMENTATION ORDER

When implementing or debugging, follow this dependency order:

```text
1. Static device/service/anchor configuration
            ↓
2. UAV position + movement
            ↓
3. Dynamic task generation/activity
            ↓
4. Communication distances/rates
            ↓
5. Persistent service replicas
            ↓
6. Task association
            ↓
7. Local/peer/anchor service-access selection
            ↓
8. CPU allocation/execution
            ↓
9. Latency calculation
            ↓
10. UAV energy calculation/update
            ↓
11. Tail-latency metrics
            ↓
12. RL/MADRL action and reward layer
```

Do not implement the RL layer first and then retrofit the environment
semantics.

---

# 34. CHANGE CONTROL

If a requested implementation change conflicts with this document, Claude must:

1. identify the conflicting assumption;
2. explain which equation/state transition is affected;
3. ask for or locate the updated research specification;
4. avoid silently changing the mathematical model.

This file should be updated whenever a research-level modeling decision is
officially changed.

---

# 35. FINAL IMPLEMENTATION CHECK

Before considering the simulator consistent with the paper, verify:

```text
[ ] IoT devices are stationary.
[ ] Active task population changes over time.
[ ] Services are atomic.
[ ] UAVs have persistent service replicas.
[ ] Replicas have explicit insertion/eviction state transitions.
[ ] Cache capacity is enforced.
[ ] Replication is fleet-coordinated.
[ ] Task migration is distinct from service replication.
[ ] Local/peer/anchor access paths are implemented.
[ ] Peer migration transfers task input, not the service.
[ ] Anchor acquisition transfers the service.
[ ] Shared physical transfers are counted once.
[ ] UAV-UAV communication depends on UAV geometry.
[ ] UAV-anchor communication depends on UAV-anchor geometry.
[ ] UAV trajectory changes communication distances.
[ ] CPU allocation follows actual execution UAV.
[ ] End-to-end latency has no double counting.
[ ] Tail latency uses actual task latency samples.
[ ] UAV energy is finite and non-replenished.
[ ] No WPT/recharging/device-energy model exists.
[ ] Invalid actions cannot corrupt environment state.
[ ] RL-specific mechanisms remain outside the core system model.
```

---

## Source-of-truth rule

**When implementing code, this file is the authoritative implementation
reference for the system-model semantics.**

If the paper text and this implementation document differ because of
presentation/notation, preserve the underlying semantics defined here.

If a future research decision changes the model, update this file first and
then modify the simulator accordingly.
