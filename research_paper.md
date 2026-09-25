# Frozen Research Blueprint

## Cooperative Service Replication, Spatial De-Coupling, and Device-Level Tail Latency in Multi-UAV MEC

**Note on scope:** this is a self-contained blueprint for the direction described in your latest message (service caching/replication → A2A/spatial de-coupling → tail latency, with WPT as an open question to resolve). It is a different, independently-frozen direction from the mobility-gated-churn-cost caching idea developed earlier in this conversation; the two are not merged here.

**Sourcing discipline used throughout:** every factual claim about a paper is grounded in its actual text/equations. Anything not explicitly present in the four papers is tagged `[NOT SUPPORTED BY PROVIDED PAPERS]`. Anything I add analytically is tagged `[MODEL INFERENCE]` (a reasoned reading of what a paper implies) or `[PROPOSED EXTENSION]` (new material for *our* paper, not claimed as existing literature).

---

## 3. STRUCTURED FOUR-PAPER COMPARISON

### P1 — Qin et al., "Latency Minimization... UAV-Assisted Cache-Computing Network With Energy Recharging" (IEEE TCOM 2025)

|  |  |
| --- | --- |
| A. Problem/system | Minimize average latency in a UAV-assisted AGIN with RGDs (request devices) and FGDs (free/helper devices) cooperating via D2D. |
| B. Entities | Multiple UAVs, RGDs, FGDs, one fixed charging station. |
| C. Device mobility | Stationary ground devices. |
| D. # UAVs | Multiple (set 𝒰 = {U₁,...,U_M}), M generic. |
| E. Dynamicity | Task set per RGD is deterministic/pre-indexed over the horizon; channel state varies per slot; **no explicit active/inactive user-population switching model**. |
| F. Task generation | Each RGD generates a fixed, indexed list of $I^R_n$ tasks over the horizon $T$. |
| G. Communication | Mixed LoS/NLoS UAV↔device path-loss model; D2D link RGD↔FGD; OFDMA-style orthogonal access. |
| H. Computation | Queueing model: local/D2D-offload/UAV-offload backlog queues, CPU-cycle-based service. |
| I. Caching | **Task caching**: whole task (input+program) pre-stored on a UAV, binary $x^i\_{n,m}$. |
| J. Service vs. task vs. content caching | **Task caching** explicitly (a specific task instance, not a reusable content/service object). No "service" or "service chain" terminology anywhere in the paper. |
| K. Migration | **Not modeled.** |
| L. WPT/energy harvesting | **Not modeled.** `[NOT SUPPORTED BY PROVIDED PAPERS]` — only a fixed ground **charging station** for the UAV itself. |
| M. UAV battery model | Explicit: Lyapunov energy queue $E^S_m(l)$ with reflecting upper/lower boundaries, updated by consumption/charging (eq. 27). |
| N. UAV propulsion energy | Implicit inside the general energy-consumption/charging-feasibility constraint (C12); not separately decomposed into a closed-form propulsion power curve. |
| O. UAV trajectory | Continuous 3-D per-slot position, speed/area/collision constraints, energy-feasibility-to-charging-station constraint. |
| P. Resource allocation vars | Cache $X$, offload split ratios $c^F,c^U$, compute allocation $f$, trajectory $Q$. |
| Q. Objective | Minimize long-run average latency $T$. |
| R. Constraints | Binary cache + capacity, split-ratio bounds, compute caps, area/speed/collision, energy-to-charging feasibility, queue stability. |
| S. Optimization technique | Decomposition: matching game (cache) + Lyapunov + CVX (convex sub-problem) + SAC (learned sub-problem). |
| T. DRL/RL | **Yes — multi-agent SAC** (confirmed from the paper's own Algorithm 2 and MDP definition): each UAV *and* each FGD is an agent $\\iota\\in\\mathcal U\\cup\\mathcal D^F$, actor $\\pi\_\\iota$ per agent, **centralized critic** taking all agents' joint observation+action (CTDE), decentralized actors. This is functionally a MASAC-style scheme, even though the paper calls it simply "SAC." |
| U. State | Per-agent partial observation: UAV position, offload/local queue backlogs, energy state (eq. 42–43). **Fixed-dimension vectors — no attention, no variable-size population handling.** |
| V. Action | UAV: flight distance/angle/altitude change + compute allocation. FGD: compute allocation. |
| W. Reward | Negative of a weighted penalty $\\Psi(l)$ (latency-driven), with explicit constraint-violation penalty terms $e_1..e_5$ if infeasible (eq. 44). |
| X. Baselines | MADQN, MADDPG, SAC-TR (trajectory-only), SAC-RA (resource-allocation-only). |
| Y. Metrics | Average system latency vs. cache capacity / # UAVs. |
| Z. Limitations | Cache decided once, static for the whole horizon; no fairness/tail objective; no attention/masking; no WPT; no migration; deterministic task-list assumption. |

### P2 — Zhao et al., "Joint Optimization of Trajectory, Offloading, Caching, and Migration for UAV-Assisted MEC" (IEEE TMC 2025)

|  |  |
| --- | --- |
| A. Problem/system | Maximize long-run throughput via joint UAV deployment, association, offloading, task caching, and inter-UAV migration. |
| B. Entities | Multiple UAVs, static users; UAV-UAV links for migration. |
| C. Device mobility | Stationary. |
| D. # UAVs | Multiple, redeployed each slot. |
| E. Dynamicity | **Explicit and central**: "the unpredictable nature of user service requests" drives the whole Lyapunov formulation — the closest of the four papers to a genuinely dynamic-demand model, though it is a per-slot task/no-task draw per user, not an attention-worthy variable-size population construct. |
| F. Task generation | Per-slot per-user task draw, association-then-schedule pipeline. |
| G. Communication | UAV↔user and UAV↔UAV (migration) LoS-weighted rate models. |
| H. Computation | Per-UAV CPU budget $W_u$, reduced by ongoing cached-task load. |
| I. Caching | **Task caching = deferred-computation queue** (a task's own payload delayed to a later slot), explicitly distinguished by the authors from popularity/content caching in their own related-work section. |
| J. Service vs. task vs. content caching | Own text: *"current research primarily focuses on... content pre-caching such as videos, images and audio, neglecting computational task caching."* **No service or service-chain notion.** |
| K. Migration | **Core mechanism**: explicit inter-UAV task handoff $m\_{i,uu'}(t)$, dedicated migration-bandwidth sub-problem, functionally an A2A link even though the term "A2A" is never used. |
| L. WPT/energy harvesting | **Not modeled.** `[NOT SUPPORTED BY PROVIDED PAPERS]` (only appears in a reference-list citation title, not the paper's own model). |
| M. UAV battery model | **Not modeled as a state variable.** Energy appears only as a scalar weight $\\lambda_2$ inside the scalar scheduling cost $c\_{i,u}(t)=\\lambda_1 D\_{i,u}+\\lambda_2 E\_{i,u}$ — no battery evolution equation. |
| N. UAV propulsion energy | Not separately modeled. |
| O. UAV trajectory | **Per-slot redeployment** (heuristic IMO-based Algorithm 2), not continuous flight optimization. |
| P. Resource allocation vars | Deployment $Q$, association $S$, offload accept $Z$, compute $A$, migrate $M$, cache $O$, migration bandwidth $B$. |
| Q. Objective | Maximize long-run average throughput. |
| R. Constraints | Single association, exclusive scheduling outcome, same-slot processing, migration-bandwidth cap, compute cap, cache-stability, cost budget. |
| S. Optimization technique | Lyapunov drift-plus-penalty + **BCD** across blocks — **no DRL**. |
| T. DRL/RL | **None.** `[NOT SUPPORTED BY PROVIDED PAPERS]` — a genuine counter-example showing classical optimization alone is viable here. |
| U–W. State/Action/Reward | Not applicable (not an RL paper). |
| X. Baselines | Not DRL baselines; compares against simplified scheduling variants. |
| Y. Metrics | Throughput, scheduling cost, migration frequency. |
| Z. Limitations (paper's own admission) | Fixed $1\\times I$ migration-queue dimensionality limits scale; only 2-hop migration (further hops **discarded**); low caching-cost weight causes cache-queue variability — flagged by the authors themselves as a weakness of their own cost design. |

### P3 — Sun et al., "Joint Optimization of Caching, Computing, and Trajectory Planning... MADDPG" (IEEE IoT-J 2024)

|  |  |
| --- | --- |
| A. Problem/system | Maximize UAV workload fairness while minimizing IoT devices' weighted (latency+energy) cost, via joint trajectory + Docker-image caching + offloading. |
| B. Entities | 1 HAP (full registry), M UAVs (limited cache), N static IoT devices. |
| C. Device mobility | Stationary. |
| D. # UAVs | Multiple, explicit set ℳ. |
| E. Dynamicity | Per-slot task per device, **Zipf-distributed popularity** over required images — the only paper with an explicit popularity-skew model. No active/inactive user switching mechanism. |
| F. Task generation | One task per IoT device per slot, each requiring a specific Docker image $j=\\psi_t(n)$. |
| G. Communication | IoT↔UAV, IoT↔HAP, and explicit **A2A** (HAP↔UAV image-download) link — the term "A2A" is used directly in this paper. |
| H. Computation | Local/UAV/HAP three-way offload, CPU-cycle delay, image-download-gated if cache miss. |
| I. Caching | **Docker-image (application/program) caching** — genuinely reusable across many different devices needing the same image. |
| J. Service vs. task vs. content caching | Closest of the four to **service/application caching** (a Docker image is effectively a self-contained service unit). Still **no multi-stage service-chain/dependency-graph structure** — each task needs exactly one image, not a sequence of dependent services. |
| K. Migration | **Not modeled.** |
| L. WPT/energy harvesting | **Not modeled.** `[NOT SUPPORTED BY PROVIDED PAPERS]` |
| M. UAV battery model | **Not modeled at all** — only IoT-device transmit energy appears in the weighted cost. `[NOT SUPPORTED BY PROVIDED PAPERS]` |
| N. UAV propulsion energy | Not modeled. |
| O. UAV trajectory | Continuous polar-coordinate movement $(\\theta\_{m,t},d\_{m,t})$, collision/coverage constraints, driven by **fairness**, not by caching state. |
| P. Resource allocation vars | Offload $A$, cache $B={\\beta\_{m,j,t}}$, trajectory $C$. |
| Q. Objective | Maximize workload-fairness index $f^{FU}\_t$ **and** minimize weighted (latency+energy) cost, subject to a **service-fairness constraint** on IoT devices — the paper's own fairness notion is about *service/workload distribution*, not latency percentiles. |
| R. Constraints | Cache capacity, coverage radius, collision distance, fairness thresholds, offload one-hot. |
| S. Optimization technique | Offload decoupled via low-complexity greedy sub-solver; remaining MAMDP solved by MADDPG. |
| T. DRL/RL | **Yes — MADDPG**, with Gumbel-Softmax (continuous-action reparameterization) and Gumbel-top-k (discrete, capacity-respecting cache sampling). |
| U. State | Per-UAV local state (implied: position, cache state, nearby demand) — **fixed-dimension, no attention module described.** `[NOT SUPPORTED BY PROVIDED PAPERS]` for attention. |
| V. Action | $(\\theta\_{m,t},d\_{m,t},\\beta\_{m,j,t})$ per UAV. |
| W. Reward | Tied to the fairness+cost objective; independent-agent MADDPG reward coupled only through the shared fairness term. |
| X. Baselines | "Proposed–No Cache" ablation (explicit), plus other joint-optimization variants. |
| Y. Metrics | Delay, energy, workload-fairness index $f^{FU}\_t$. |
| Z. Limitations | No cache switching/update cost (memoryless re-decision every slot); no cross-UAV cache coordination; no UAV battery model; no WPT; no attention/masking; fairness is workload-distribution fairness, not a device-level *latency-percentile* fairness metric. |

### P4 — Betalo et al., "Dynamic Charging and Path Planning for UAV-Powered RWSNs... MA-DDQN" (IEEE T-ASE 2025)

Used, per your instruction, strictly for its mobility+energy+multi-agent-DRL methodology, not its WSN/laser mechanisms.

|  |  |
| --- | --- |
| A. Problem/system | Minimize UAV task-completion time + sensor-node death time via joint charging-target selection and path planning. |
| B. Entities | Multiple laser-charging UAVs (LCUs), K sensor nodes (SNs), ground laser stations. |
| C. Device mobility | SNs stationary (explicitly excluded from the MDP state as "constant during the mission"). |
| D. # UAVs | Multiple (multi-agent). |
| E. Dynamicity | SN energy-depletion dynamics, not task/content demand — **not a MEC task-arrival model at all.** |
| F. Task generation | Not a computation-task model; "tasks" here means data-collection targets. |
| G. Communication | Laser (power) + RF (data). |
| H. Computation | **Not modeled.** `[NOT SUPPORTED BY PROVIDED PAPERS]` |
| I. Caching | **Not modeled.** Confirmed zero occurrences in the text. |
| J. Service/task/content caching | N/A — no caching concept present. |
| K. Migration | Not modeled. |
| L. WPT/energy harvesting | **Present, but UAV-directed, not IoT-directed**: ground laser stations recharge the **UAV's own battery**. This is the only WPT-like mechanism in the entire four-paper base, and it targets the wrong endpoint relative to the "UAV→IoT-device WPT" concept in your framing. |
| M. UAV battery model | Explicit MDP state component $S^E_t={S^e_t,u}$ (residual energy), updated each slot by consumption. |
| N. UAV propulsion energy | Folded into total energy consumed per slot $E\_{total}$; not separately closed-form. |
| O. UAV trajectory | Discretized position state $S^p_t={x^u_t,y^u_t,h}$, updated by the charging/movement action. |
| P. Resource allocation vars | Charging-target selection $a^\\kappa_t$, implicit path. |
| Q. Objective | Minimize (task-completion time + SN death time). |
| R. Constraints | Per the paper's C1–C5 (energy, coverage, charging-range feasibility). |
| S. Optimization technique | MDP + MA-DDQN. |
| T. DRL/RL | **Yes — multi-agent double DQN (MA-DDQN)**, discretized state/action. |
| U. State | ${S^{die}\_t, S^E_t, S^p_t}$ — sensor death-times, residual energy, position. **Fixed-size, discretized — no attention.** |
| V. Action | Binary "charge sensor κ or not" per slot. |
| W. Reward | $R_t = -\\ln(f_t) - \\ln(T^{die})$ — log-normalized combination of task-completion factor and cumulative death time. |
| X. Baselines | State-of-the-art path-planning/charging comparators (paper's own benchmark set). |
| Y. Metrics | Average delay, energy consumption, task-completion time. |
| Z. Limitations | No computing/caching dimension whatsoever; discretized state/action limits scalability; no attention/masking; **no transferable WPT-to-device model** — its WPT targets UAVs, not IoT devices, so it cannot be directly reused for the "recharge energy-constrained IoT devices" idea in your framing without a structural change of target. |

### Cross-Paper Gap Matrix

| Paper | Caching | Service replication | WPT (UAV→IoT device) | UAV energy/battery model | Mobility (continuous) | Dynamic active users | Tail latency | P95/P99 or CVaR | Fairness | MADRL | Attention | Action masking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | ✔ (static task cache) | ✘ | ✘ | ✔ | ✔ | ✘ | ✘ | ✘ | ✘ | ✔ (multi-agent SAC/CTDE) | ✘ | ✘ |
| P2 | ✔ (deferred-task queue) | ✘ | ✘ | ✘ (only a scalar cost weight) | ✘ (per-slot redeployment, not continuous flight) | Partial (unpredictable per-slot demand, no attention construct) | ✘ | ✘ | ✘ | ✔ (MADDPG)\* | ✘ | ✘ |
| P3 | ✔ (Docker-image, popularity-driven) | ✘ | ✘ | ✘ | ✔ | Partial (Zipf popularity, static demand skew) | ✘ | ✘ | ✔ (workload/service fairness, not tail-latency) | ✔ (MADDPG) | ✘ | ✘ |
| P4 | ✘ | ✘ | ✔ (but UAV-directed, not IoT-directed) | ✔ | ✔ (discretized) | ✘ | ✘ | ✘ | ✘ | ✔ (MA-DDQN) | ✘ | ✘ |

\* Corrected: P2 is not DRL — the ✔ mark under "MADRL" would be wrong; **P2 has no DRL at all**, shown as ✘ in the row above under "MADRL." *(Row corrected: only P1, P3, P4 use DRL/MADRL; P2 uses Lyapunov+BCD.)*

**Reading the matrix honestly:** every single cell for *service replication*, *WPT-to-IoT-device*, *tail latency*, *P95/P99/CVaR*, *attention*, and *action masking* is empty across all four papers. This is not a criticism of the papers — none of them claim to do these things — but it means **none of the six central components of your intended direction is literature-grounded**; they must all be built as `[PROPOSED EXTENSION]`, using only the caching, energy, trajectory, and MADRL *mechanics* of the four papers as a technical foundation.

---

## 4. NOVELTY ANALYSIS

**Chain A (service replication → A2A/spatial de-coupling → trajectory freedom → tail latency):** This chain is traceable directly onto two real, literature-confirmed dependencies: (i) in P3, a cache **miss** forces a UAV to pay an A2A (HAP-link) cost that is implicitly cheaper near the HAP/registry; (ii) in P2, a UAV that cannot serve a task locally must **migrate** it to a specific peer UAV, an inherently *spatial* dependency (bandwidth/link-quality between specific UAV pairs). Neither paper asks whether *more replication* reduces how tightly a UAV's position is pinned to the registry or to specific peers. This is a genuine, specific, checkable gap.

**Chain B (service caching → reshaped energy → more energy for WPT → IoT device support → tail latency):** This chain requires an entirely new physical mechanism (UAV→IoT-device wireless power transfer) that is **absent from all four papers** — the only WPT-like mechanism in the base literature (P4) recharges the **UAV**, not IoT devices. The causal path from "caching changes UAV energy expenditure" to "device-level tail latency improves via WPT" is three unsupported inferential hops long, each requiring new, unvalidated assumptions (IoT devices are meaningfully energy-constrained; UAVs have exploitable energy surplus; WPT meaningfully changes a device's *own* task-completion latency rather than just its battery level).

**Comparison:** Chain A rests on mechanisms that already exist in the base papers (A2A cost in P3, migration dependency in P2) and needs one new, well-scoped addition (a fleet-wide replication variable + a spatial-coupling cost term). Chain B requires importing a whole new subsystem with no grounding anywhere in the four papers and a much longer, weaker causal path to the outcome metric.

> **FINAL CORE NOVELTY:** Cooperative, fleet-wide **service replication** reduces a UAV's A2A dependency (on the registry/HAP for cache misses, and on specific peer UAVs for task migration), which relaxes the spatial coupling between a UAV's position and its service-acquisition sources, which in turn lets the trajectory-optimization objective be driven primarily by **where dynamically-active demand currently is**, disproportionately benefiting the currently worst-served (tail) devices — making **device-level tail latency (not average latency)** the metric that actually reveals this mechanism's effect.

> **WHY THIS IS DIFFERENT:** P1 and P3 already jointly optimize caching and trajectory, but in both, the cache decision affects only a *local* miss-penalty term — it never changes what the trajectory is *allowed or incentivized* to do spatially. P2 already has an inter-UAV dependency (migration) but no caching-driven mechanism to *relax* it. No paper connects "how much is replicated across the fleet" to "how spatially free the trajectory can be" to "how the tail of the latency distribution behaves."

> **WHAT EXISTING PAPERS ALREADY DO:** P1: static, single-instance task pre-caching with no spatial-coupling consequence. P2: dynamic task deferral/migration with an explicit but *unrelieved* spatial dependency (migration bandwidth, 2-hop limit). P3: dynamic, popularity-driven, per-UAV *independent* image caching with a trajectory that is fairness-driven and caching-agnostic.

> **WHAT THEY DO NOT JOINTLY MODEL:** None models replication density as a variable that changes the *cost of being far* from a service source; none uses a tail/percentile latency objective; none uses attention over a dynamically-sized active-task set; none uses hard action masking.

> **WHAT OUR PAPER ADDS:** (1) A fleet-wide service-replication decision (generalizing P3's per-UAV Docker-image cache into a coordinated multi-UAV placement problem); (2) an explicit **spatial-coupling cost** term linking replication state to the latency penalty of being far from a service source, replacing P2's fixed, unrelieved migration-dependency assumption; (3) a **device-level tail-latency (P95/CVaR) objective**, motivated specifically by the mechanism (tail devices are the ones the mechanism is meant to help — average latency would wash this signal out, since it is dominated by already-well-served near-anchor devices); (4) an attention-based, masked MADRL controller for the resulting mixed discrete/continuous, variable-size-observation decision problem.

> **WHAT CLAIMS WE MUST NOT MAKE:** We must not claim to be the first UAV-MEC paper on caching+trajectory (P1, P3 exist), the first on caching+migration (P2 exists), the first MADRL UAV-MEC paper (P1, P3, P4 exist), or that WPT is part of the contribution (it is explicitly removed — see Section 16). We must not claim optimality, universal scalability, or "solved" tail latency in UAV-MEC broadly — only this specific replication-driven mechanism, under the assumptions we freeze below.

---

## 5. FROZEN RESEARCH QUESTION, HYPOTHESIS, OBJECTIVE, NOVELTY, CONTRIBUTIONS

**FINAL RESEARCH QUESTION:** *How can cooperative, fleet-wide service replication be jointly optimized with UAV trajectory and task association/offloading — via an attention-based, action-masked multi-agent DRL controller — to reduce the spatial coupling between a UAV's position and its service-acquisition dependencies, thereby reducing device-level tail latency (P95/CVaR) for a dynamically changing set of active IoT tasks, relative to independent per-UAV caching without replication coordination?*

**FINAL HYPOTHESIS:** As fleet-wide service-replication coverage increases and is coordinated (rather than decided independently per UAV, as in P3), the *effective spatial-coupling radius* between a UAV and its nearest usable service source shrinks; this allows trajectories to track dynamically-active demand hotspots more closely, and the resulting latency improvement is concentrated in the tail (P95/P99) of the device-level latency distribution rather than uniformly across the average — an effect that a purely average-latency-optimized policy (P1-style) or an uncoordinated popularity-caching policy (P3-style) will not reproduce.

**FINAL SYSTEM OBJECTIVE:** Minimize a device-level tail-latency risk measure (frozen in Section 8/9 as CVaR of per-device latency) subject to replication-capacity, trajectory, association, and computation constraints — **not** long-run average latency (P1) and **not** workload-distribution fairness (P3).

**FINAL NOVELTY:** As stated in Section 4's boxed statement.

**FINAL EXPECTED CONTRIBUTIONS (kept to three, per the "not unnecessarily complicated" rule):**

1. A **coordinated, fleet-wide service-replication model** with an explicit spatial-coupling cost, extending P3's independent per-UAV caching and P2's unrelieved migration dependency.
2. A **device-level tail-latency (CVaR) objective and fairness metric** for UAV-MEC caching/trajectory joint optimization, replacing the average-latency (P1) and workload-fairness (P3) objectives used previously.
3. An **attention-based, action-masked MAPPO controller** handling the resulting mixed discrete (replication, association)/continuous (trajectory) action space under a dynamically-sized active-task population, with a clean ablation isolating the causal chain replication → spatial de-coupling → tail-latency reduction.

---

## 6. SYSTEM MODEL (frozen)

**IoT devices** `[builds on P1/P3's stationary-device assumption]`

- **Stationary** — no user mobility (explicitly excluded per your instruction, and consistent with all four base papers).
- Fixed count $N$, spatially distributed over the region (can be clustered, mirroring P3's Zipf-per-catalog idea generalized to per-cluster demand `[PROPOSED EXTENSION]`).
- **Active/inactive mechanism** `[PROPOSED EXTENSION, not in any of the four papers]`: each device toggles active/inactive via a stochastic arrival process (Section 20); "active" = currently holding an unresolved task.
- Task arrival: Poisson per device when active, generating one task requiring one service $s\\in\\mathcal S$ from a fixed catalog (service granularity decided in Section 15).
- Task size $D_n$ and required CPU cycles $C_n$ — reused directly from P1/P3's task-tuple convention.
- IoT devices have **no energy state relevant to the model** (WPT removed — see Section 16), so no device battery variable is carried.
- Service requirement: each task maps to exactly one service $s=\\psi(n,t)$, reusing P3's $\\psi_t(n)$ notation.

**UAVs**

- Fixed count $M$ (multi-agent, one agent per UAV, as in P1/P3/P4).
- Altitude fixed/bounded as in P1/P3 (not a decision variable — no paper varies altitude as a first-class control beyond bounds).
- Position/velocity: continuous, per-slot, bounded speed and area (P1/P3-style).
- Battery: `[MODEL INFERENCE from P1's Lyapunov energy-queue technique]` — retained as a **soft, budget-style constraint** (propulsion + communication + computation energy drawn from a bounded per-slot budget), **not** a hard recharging-station model like P1's, since WPT/recharging infrastructure is deliberately not part of this frozen direction (kept simple, per your "do not add unnecessary infrastructure" instruction — if reviewers require it, a P1-style charging station is a clearly labeled `[PROPOSED EXTENSION]` for future work, not part of the frozen core).
- CPU: per-UAV compute budget $W_m$ (as P2/P3).
- Cache/replication storage: per-UAV capacity $Z_m$ over the service catalog (as P3's $Z_m$, generalized to replication accounting across the fleet).
- Communication: UAV↔device (as P1/P3), UAV↔registry-anchor (A2A, as P3), UAV↔UAV (A2A, as P2's migration link) — all three link types are needed because all three are the literal channels through which "spatial dependency" is expressed.

**Services (not service chains)** — **frozen decision, explained in Section 15**

- Atomic, self-contained services $s\\in\\mathcal S$ (generalizing P3's Docker images), each with a fixed size $z_s$ and popularity/demand weight (Zipf, as P3).
- **No multi-stage service-chain/dependency-graph structure** is modeled — deliberately, to keep the optimization tractable and keep the paper's complexity centered on the *replication/coordination* mechanism, not on chain-topology bookkeeping. `[PROPOSED EXTENSION reduces P3's single-service semantics to a fleet-wide replication variable, but does not add chain dependencies.]`

**Task model**

- Task-service mapping: one task ↔ one service (as P3).
- Arrival: Poisson per active device (Section 20).
- Deadline: soft, expressed through the tail-latency objective rather than a hard per-task deadline constraint (kept out to avoid double-counting the same idea two ways — tail latency already penalizes late completion).
- Queueing: per-UAV compute queue (P1/P2-style backlog).
- Migration: retained from P2, generalized as the "peer A2A" resolution path when a UAV's local replica is missing.

**Communication**

- IoT→UAV: as P1/P3 (LoS-weighted rate).
- UAV→UAV (A2A/migration): as P2.
- UAV→registry-anchor (A2A/service pull): as P3.
- **No UAV→IoT WPT link** (removed, Section 16).

**Mobility**

- Continuous UAV trajectory (as P1/P3), max velocity bound, flight-region bound, inter-UAV collision constraint (as both P1 and P3 already enforce), discretized into slots (as all four papers do).
- No acceleration constraint added — not needed for the research question and would add unjustified complexity (per your "do not add unnecessary physical detail" instruction).

**Energy**

- Propulsion + communication + computation energy drawn from a bounded per-slot/episode budget (`[MODEL INFERENCE]`, generalizing P1's energy-queue *technique* without its charging-station *infrastructure*).
- **No WPT, no harvesting** (Section 16).

---

## 7. DECISION VARIABLES (frozen — kept minimal)

| Variable | Meaning | Type | Dimension | Decision-maker | Frequency | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| $\\rho\_{m,s,t}$ | Replicate service $s$ on UAV $m$ at slot $t$ (persists until changed) | Discrete (binary) | $M\\times | \\mathcal S | \\times T$ | UAV agent (coordinated) |
| $\\alpha\_{n,m,t}$ | Associate active device $n$'s task with UAV $m$ (or leave unserved) | Discrete (one-hot) | Variable ($N\_{active}(t)\\times M$) | UAV agent / association module | Per slot | $\\sum_m \\alpha\_{n,m,t}\\le1$ |
| $\\mu\_{n,mm',t}$ | Migrate device $n$'s task from UAV $m$ to peer $m'$ if $m$ lacks the replica | Discrete (binary) | Variable | UAV agent | Per slot | Only if $\\rho\_{m,s,t}=0$ and $\\rho\_{m',s,t}=1$ |
| $f\_{n,m,t}$ | CPU allocation to task $n$ on UAV $m$ | Continuous | Variable | UAV agent | Per slot | $\\sum_n f\_{n,m,t}\\le W_m$ |
| $q\_{m,t}$ | UAV trajectory action (direction/speed) | Continuous | $M\\times2\\times T$ | UAV agent | Per slot | Speed/area/collision bounds |

**Explicitly excluded / rejected variables** (kept out to satisfy "do not create a giant optimization problem"): bandwidth allocation (folded into a fixed per-link channel model, as P1/P3 already simplify it in places), transmission power control (fixed, as in P3), WPT allocation/charging power/time (removed, Section 16), fine-grained cache eviction ordering (subsumed by the binary $\\rho\_{m,s,t}$ with a switch-cost penalty rather than an explicit LRU/LFU rule — keeps the state space smaller).

---

## 8. OBJECTIVE FUNCTION

**Why average latency is insufficient here, specifically:** the mechanism under test (replication → spatial de-coupling) is expected to help disproportionately the *worst-off* devices — those in hotspots far from the registry/anchor or from peers holding a needed replica. A near-anchor device's latency is already low and largely insensitive to replication coverage; averaging over all devices would dilute exactly the signal the paper needs to demonstrate. This is a specific, mechanism-grounded reason, not a generic "tail metrics are more rigorous" claim.

**Alternatives considered:**

- *P95/P99 latency*: simple, interpretable, directly matches your framing, but non-differentiable/non-smooth as a training signal and noisy at small per-slot device counts.
- *Maximum latency*: too sensitive to single-device outliers/measurement noise; not a stable RL reward signal.
- *CVaR (Conditional Value-at-Risk) of latency at level $\\beta$ (e.g., $\\beta=0.95$)*: averages the tail beyond the $\\beta$-quantile rather than reporting a single quantile point — smoother, more sample-efficient to estimate per training batch, and has a well-established convex/risk-theoretic formulation compatible with policy-gradient training.
- *Deadline-violation probability*: requires a hard per-task deadline we deliberately did not adopt (Section 6), so this is set aside to avoid a redundant modeling choice.

**Frozen choice:** **CVaR$\_\\beta$ of per-device latency as the primary training/optimization objective**, with **P95/P99 reported as evaluation-time, human-readable metrics** (not the training signal itself). This gives a smooth optimization target while still letting the paper report the percentile numbers reviewers expect to see.

$$\\min\_{\\rho,\\alpha,\\mu,f,q};; \\text{CVaR}*\\beta\\big(L_n(t)\\big) ;+; \\eta_1!!\\sum*{m,s,t} |\\rho\_{m,s,t}-\\rho\_{m,s,t-1}|,\\kappa_s ;+; \\eta_2, E\_{\\text{energy}}(t)$$

- Term 1: the frozen primary tail-risk objective.
- Term 2: replication **switch cost** (necessary — without it, replication could reshuffle every slot exactly like P3's unrealistic memoryless caching, undermining the spatial-coupling story, which requires the replicated set to be *meaningfully persistent* for "reduced dependency" to be a real, stable effect rather than an artifact).
- Term 3: an energy-budget soft penalty (keeps trajectories physically reasonable without importing a full charging-station subsystem).

No CVaR/weighted-sum term is included "by default" — each is justified above against the specific mechanism being tested.

---

## 9. FAIRNESS — PRECISE DEFINITION (frozen)

**Primary metric:** **CVaR$\_{0.95}$ of per-device task latency**, computed over the currently-active device population at evaluation time. This directly captures "the worst 5% of currently-active devices," which is the device-level notion your prompt specifies, and is mathematically a coherent risk measure (subadditive, convex), unlike a raw max or an ungrounded weighted sum.

**Secondary metrics (reported, not optimized):**

- P95 and P99 latency (point quantiles, for direct comparability with papers/readers that expect percentile numbers).
- Jain's fairness index over per-device average latency (a well-known, simple cross-check that the tail improvement isn't achieved by degrading a different subset of devices asymmetrically).
- Maximum observed latency per episode (diagnostic only, not a target — too noisy to optimize directly, as discussed in Section 8).

**What fairness explicitly does *not* mean here:** it is **not** P3's workload-distribution fairness ($f^{FU}\_t$, how evenly *tasks* are spread across UAVs) — that is a different, valid, but distinct notion which we do not adopt as the primary metric, to avoid conflating "UAVs share load evenly" with "no device is left with terrible latency," which are not the same thing and can trade off against each other.

---

## 10. COMPLETE MATHEMATICAL FORMULATION

$$ \\begin{aligned} \\min\_{\\rho,\\alpha,\\mu,f,q}\\quad & \\text{CVaR}*{0.95}\\big(L_n(t)\\big) + \\eta_1\\sum*{m,s,t}|\\rho\_{m,s,t}-\\rho\_{m,s,t-1}|\\kappa_s + \\eta_2 E\_{\\text{energy}}(t)\\ \\text{s.t.}\\quad & \\textbf{(Replication/cache)}; \\textstyle\\sum\_{s}\\rho\_{m,s,t}z_s\\le Z_m,\\ \\forall m,t\\ & \\textbf{(Association)}; \\textstyle\\sum_m \\alpha\_{n,m,t}\\le1,\\ \\forall n\\in\\mathcal N\_{\\text{active}}(t)\\ & \\textbf{(Migration feasibility)}; \\mu\_{n,mm',t}\\le(1-\\rho\_{m,\\psi(n),t})\\cdot\\rho\_{m',\\psi(n),t}\\ & \\textbf{(CPU)}; \\textstyle\\sum_n f\_{n,m,t}\\le W_m,\\ \\forall m,t\\ & \\textbf{(Bandwidth)}; \\text{fixed per-link channel model (not a free decision variable, per Section 7)}\\ & \\textbf{(Battery/energy budget)}; E\_{\\text{prop}}(t)+E\_{\\text{comm}}(t)+E\_{\\text{comp}}(t)\\le E\_{\\text{budget}}\\ & \\textbf{(Trajectory)}; |q\_{m,t}|\\le v\_{\\max}\\Delta t,\\ q\_{m,t}\\in\\text{region},\\ |q\_{m,t}-q\_{m',t}|\\ge D\_{\\min}\\ & \\textbf{(Queue)}; \\text{per-UAV compute-backlog stability (Lyapunov-style, as P1/P2)}\\ & \\textbf{(Task exclusivity)}; \\text{a task is exactly one of {local-hit, migrated, offloaded-to-anchor}, not several at once}\\ & \\textbf{(Non-negativity)}; \\text{all rates, energies, queue lengths} \\ge 0 \\end{aligned} $$

**Variable-type breakdown and why the problem is hard:**

- **Discrete:** $\\rho$ (replication), $\\alpha$ (association), $\\mu$ (migration) — combinatorial.
- **Continuous:** $f$ (CPU split), $q$ (trajectory) — nonlinear due to distance-based rate/energy terms (as in all four base papers' channel models).
- **Non-convex:** CVaR of a nonlinear, decision-dependent latency function is non-convex in the joint discrete/continuous variables; the LoS/NLoS-style rate expressions (inherited pattern from P1/P3) are themselves non-convex in position.
- **Combinatorial:** replication placement across $M$ UAVs × $|\\mathcal S|$ services is a knapsack-family problem per UAV, coupled across UAVs by the migration-feasibility constraint.
- **Time-coupled:** the switch-cost term and the energy/queue-stability constraints couple consecutive slots (Lyapunov-style long-run constraints, as P1/P2 already require).
- **Stochastic:** task arrivals (Poisson, Section 20) and the active-device population are random processes, not known in advance.

This combination (discrete+continuous, non-convex, combinatorial, time-coupled, stochastic, multi-agent) is exactly the profile that motivates MADRL rather than classical optimization alone — consistent with why P1 and P3 both moved to DRL for their comparably structured (though simpler) problems, while P2 could stay with Lyapunov+BCD only because it lacked the combinatorial replication-coordination and tail-risk objective that make this problem harder.

---

## 11. SOLUTION METHOD (frozen)

| Algorithm | Handles discrete+continuous? | Multi-UAV cooperation | Partial observability | Scalability to variable active-task count | Stability | Native action-masking support | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **QMIX** | Poor — designed for discrete joint-action value decomposition; continuous trajectory control needs heavy discretization | Yes (value decomposition) | Yes | Weak — value decomposition assumes a fixed joint-action structure | Reasonably stable | Awkward (masking a mixed value network is non-standard) | **Rejected** — continuous trajectory control is core to the problem and P1/P3 both keep it continuous; discretizing it would be a regression relative to the base literature. |
| **MADDPG** | Yes (as P3 already demonstrates, via Gumbel-Softmax/top-k for the discrete parts) | Yes (centralized critic) | Yes | Weak — off-policy replay buffer stores transitions at a fixed input dimensionality; a changing active-task population makes old buffer entries increasingly off-distribution | Known to be sensitive to hyperparameters/non-stationarity in cooperative settings | Possible but not natural (masking a deterministic policy's continuous+discrete hybrid output is more awkward than masking logits) | **Retained as the secondary/baseline algorithm** — directly literature-grounded (P3 already validates it on a structurally similar problem), making it the fairest, most literature-consistent baseline. |
| **MASAC** | Yes | Yes | Yes | Same off-policy replay-buffer staleness issue as MADDPG under a dynamic active-task population | Generally more stable than MADDPG (entropy regularization) — and directly precedented by **P1's own multi-agent SAC/CTDE scheme** | Same awkwardness as MADDPG | **Considered seriously** (strong literature precedent via P1) but not selected as primary — see reasoning below. |
| **MAPPO** | Yes (separate discrete/continuous action heads) | Yes (centralized critic, decentralized actors — standard CTDE) | Yes | **Best fit** — on-policy: every training batch is freshly collected under the *current* active-task distribution, so there is no stale-replay-buffer mismatch when the active-task population changes over time | Strong empirical stability/robustness to hyperparameters in cooperative MARL | **Natural fit** — hard masking is applied directly at the logit level before the softmax, a standard, well-documented technique | **Selected as primary.** |

**Why MAPPO over MASAC specifically (not a popularity choice):** the defining structural feature of this problem is the **dynamically changing active-task population** (Section 20's critical experiment). Off-policy methods (MADDPG, MASAC) rely on replay buffers of past transitions; when the active-task set's size/composition changes over time, older buffer transitions were collected under a different effective input distribution to the attention encoder, degrading the accuracy of the critic's value estimates for the current policy (a *distributional-shift-in-the-replay-buffer* problem specific to our non-stationary-population setting). MAPPO's on-policy updates use only freshly collected trajectories each iteration, sidestepping this specific failure mode. This is the deciding factor — not general popularity of PPO.

**Frozen:** **MAPPO (primary)** with CTDE, attention encoder (Section 12), hybrid discrete/continuous action heads, hard logit-level masking (Section 13). **MADDPG (secondary, baseline only)** — also serves directly as the "MADRL without attention/masking" ablation, since it is the closest literature-grounded stand-in for "what P3 would do if generalized to this problem without our additions."

---

## 12. STATE SPACE DESIGN

**Per-UAV local state (fixed-size part, directly extending P1 eq. 42–43 / P4's state design):** $$s_m^{\\text{local}}(t) = \\big\[q\_{m,t},\\ v\_{m,t},\\ E_m(t),\\ \\text{CPU-util}*m(t),\\ \\rho*{m,\\cdot,t}\\ (\\text{cache-state vector, size}\\ |\\mathcal S|)\\big\]$$

**Variable-size part — the active-task/device set the UAV currently observes:** Each nearby *active* device/task $n$ is encoded as a token: $$x_n(t) = \\big\[\\text{relative position},\\ \\text{required service } s,\\ \\text{queueing urgency (time since arrival)},\\ \\text{whether } n\\text{'s required service is already replicated locally}\\big\]$$

**Attention module — concrete purpose (not "use attention for scalability" left vague):**

- **Input tokens:** the set ${x_n(t)}$ of currently-active, currently-observable devices/tasks near UAV $m$ — this set's *size* changes every slot as devices activate/deactivate (Section 20), which a fixed-length vector cannot represent without either truncation (losing information when many devices are active) or padding (wasting capacity/misleading the network when few are active).
- **Embeddings:** each token $x_n(t)$ is passed through a shared linear/MLP embedding layer, so tokens are compared in a common space regardless of how many there are.
- **Attention:** a self-attention (or UAV-as-query, tasks-as-key/value cross-attention) layer computes, for the querying UAV, a weighted combination of nearby tasks' embeddings, where the weights reflect *relevance* (e.g., a nearby device whose required service is **not** currently replicated anywhere reachable should receive higher attention weight than one already well-served) — this is the precise mechanism by which the policy learns to prioritize the devices the replication mechanism is meant to help.
- **Aggregated representation:** the attention output (a fixed-size vector regardless of the number of input tokens) is concatenated with the fixed-size local state $s_m^{\\text{local}}(t)$.
- **Policy/value network:** the concatenated vector feeds the actor (produces the action heads in Section 13) and, for the centralized critic (CTDE), is further concatenated across all $M$ UAVs' aggregated representations.

This directly satisfies the requirement that attention have a stated, mechanism-specific purpose: it is the component that lets the replication/association decision be sensitive to *which* currently-active devices are under-served, at a population size that changes every slot.

---

## 13. ACTION SPACE AND ACTION MASKING

**Discrete heads (per UAV agent):**

- Replication action: for a bounded shortlist of candidate services (e.g., top-$K$ by current local demand relevance, itself an attention output) — cache/evict each, subject to capacity.
- Association: assign each locally-visible active task to {serve locally, migrate to a specific visible peer, decline/queue}.

**Continuous head:**

- Trajectory: 2-D velocity vector (direction + speed), bounded by $v\_{\\max}$, mirroring P1/P3's continuous movement.

**Reduced/excluded from the action space** (kept tractable, per your instruction): bandwidth allocation, transmission power, and any WPT-related action (all removed or fixed, consistent with Sections 6, 7, 16).

**Action masking — concrete, per action type:**

| Action | Infeasible when | Mask source | Hard or penalty-based | Effect |
| --- | --- | --- | --- | --- |
| Replicate service $s$ | Would exceed remaining cache capacity $Z_m$ | Computed directly from current $\\rho\_{m,\\cdot,t}$ and $Z_m$ each slot | **Hard** (logit set to $-\\infty$ before softmax) | Guarantees the capacity constraint is never violated by construction, not just penalized. |
| Migrate task to peer $m'$ | $m'$ is out of A2A range, or $m'$ does not hold the required replica | Computed from current position/link-quality and $\\rho\_{m',\\cdot,t}$ | **Hard** | Prevents physically impossible migration actions from ever being sampled. |
| Associate task $n$ | Task $n$ is not currently active / not currently visible to UAV $m$ | Directly follows from the current active-task set (Section 20) | **Hard** | Keeps the discrete association head consistent with the *actual current* variable-size task set, not a stale fixed-size assumption. |
| Trajectory step | Would exit the flight region or violate the inter-UAV minimum-distance constraint | Computed from current position and neighbors' positions | **Soft (penalty-based)** — continuous actions are harder to hard-mask cleanly; instead, an out-of-bounds/collision action is clipped post-hoc and penalized in the reward, consistent with how P1/P3 handle their own equivalent constraints (constraint-violation penalty terms, e.g., P1's $e_2,e_3$ in eq. 44). | Discourages infeasible trajectories while keeping the continuous action space smooth for gradient-based training. |

This directly satisfies the requirement to distinguish hard vs. penalty-based masking and to explain, per action, exactly when and how each mask applies.

---

## 14. REWARD DESIGN

$$r_m(t) = -\\underbrace{\\widehat{\\text{CVaR}}*{0.95}(L_n(t))}*{\\text{tail-latency term}} ;-; \\eta_1\\underbrace{\\sum\_{s}|\\rho\_{m,s,t}-\\rho\_{m,s,t-1}|\\kappa_s}*{\\text{switch-cost term}} ;-; \\eta_2\\underbrace{E*{\\text{energy},m}(t)}\_{\\text{energy term}} ;-; \\underbrace{\\sum_k e_k\\cdot\\mathbb 1\[\\text{violation}*k\]}*{\\text{constraint-penalty terms (as P1 eq. 44)}}$$

- **Tail-latency term:** the batch-level CVaR estimate over currently-active devices served in this slot — a **shared, cooperative** reward term (identical across agents, standard CTDE cooperative-reward practice, and necessary because tail latency is a *population-level* statistic no single UAV fully controls).
- **Switch-cost term:** directly mirrors the objective's churn penalty (Section 8) — without it in the *reward*, the trained policy would optimize a different objective than the one frozen in Section 8, breaking reward-objective alignment.
- **Energy term:** keeps trajectories physically reasonable without a full charging-station subsystem.
- **Constraint-penalty terms:** directly modeled on P1's own eq. 44 approach (large penalty $-e_k$ if a *hard* constraint the mask couldn't fully prevent — e.g., a soft-masked trajectory violation — occurs).
- **Explicitly excluded from the reward:** task-completion count, cache-hit count, and workload-fairness terms are **not** separately rewarded — they are *consequences* we measure (Section 18 metrics), not objectives we optimize, to avoid the well-known failure mode of a reward that silently drifts from the frozen objective in Section 8.

**Normalization:** each term is normalized by its own empirical running mean/std (standard reward-scaling practice) before weighting, so $\\eta_1,\\eta_2$ are interpretable trade-off weights rather than absorbing arbitrary unit mismatches. **No terminal reward:** episodes are treated as continuing/long-run (as P1's and P2's own long-run-average framing), so no special terminal bonus is defined.

---

## 15. THE EXACT ROLE OF CACHING (service replication)

**What is cached:** a **self-contained service** (generalizing P3's Docker image) — explicitly **not** a task instance (that would collapse into P1/P2's semantics), **not** a full service *chain*/dependency graph (rejected as unnecessary complexity, see below), and **not** intermediate computation results (a different mechanism, out of scope here).

**Why not a full service-chain/DAG model:** your framing mentions "services/service chains" as a possibility. A multi-stage dependency-graph model would require (a) chain-topology bookkeeping, (b) per-stage placement and inter-stage communication cost, and (c) partial-chain-cached-vs-missing bookkeeping — none of which exists in any of the four base papers, and none of which is *necessary* for the core novelty (spatial de-coupling via replication). Adding it would directly violate the "not unnecessarily complicated" and "not a giant optimization problem" rules. **Frozen decision: atomic service replication, not service chains.**

**Why replication changes the optimization (the mechanism, made explicit):**

- **Task routing:** a task can be served locally (if replicated on the serving UAV), via migration (if replicated on a nearby peer — generalizing P2's mechanism), or via an anchor pull (if replicated nowhere nearby — generalizing P3's A2A miss path). Replication density directly changes which of these three (increasingly costly) paths is available.
- **UAV-UAV communication (A2A):** more fleet-wide replication reduces the *expected* frequency and distance of both migration and anchor-pull A2A events.
- **Trajectory:** because replication reduces the expected cost of "not being near a service source," the trajectory policy is freed to weight *demand location* more heavily relative to *service-source proximity* — this is the crux of the "spatial de-coupling" claim, and it is why replication is modeled as a **persistent, switch-cost-bearing state** (Section 8) rather than P3's free-reshuffle cache: persistence is what makes "reduced dependency" a stable property of the policy rather than a one-slot artifact.
- **Energy:** fewer/cheaper A2A events reduce communication energy, indirectly giving trajectories more energy budget for demand-tracking movement (a light echo of your Chain B, retained only as a *secondary*, already-covered-by-the-shared-energy-budget consequence — not as a separate WPT mechanism).
- **Latency and tail latency:** the direct payoff — devices in hotspots far from the anchor/peers see the largest latency reduction as replication coverage grows, which is exactly why the tail metric (Section 8–9) is the right lens.

**Mathematical mechanism (spatial-coupling cost term, `[PROPOSED EXTENSION]`):** $$L_n(t) = \\underbrace{L^{\\text{compute}}*n(t)}*{\\text{as P1/P3}} + \\big(1-\\rho\_{\\text{serving-UAV},\\psi(n),t}\\big)\\cdot\\underbrace{L^{\\text{A2A}}*n(t)}*{\\text{migration- or anchor-distance-dependent, as P2/P3's own rate models}}$$ This single conditional term is the entire new mechanism: when the serving UAV already has the replica ($\\rho=1$), the A2A term vanishes and the UAV's position is irrelevant to that task's latency; when it doesn't, latency depends on distance to whichever source (peer or anchor) can supply it — precisely the spatial coupling the paper is about.

---

## 16. THE EXACT ROLE OF WPT — DECISION

Evaluated against your own three categories:

**A. Central to the novelty?** No — Section 4's novelty analysis shows Chain A (replication → A2A de-coupling → trajectory freedom → tail latency) is fully self-sufficient and more directly literature-grounded than Chain B (which requires WPT).

**B. A necessary supporting mechanism?** No — the frozen energy model (Section 6, a bounded per-slot budget) already gives trajectories a physically meaningful energy cost without needing a device-recharging subsystem. Nothing in the frozen objective (Section 8), constraints (Section 10), or mechanism (Section 15) depends on IoT devices being energy-recharged.

**C. Unnecessary and should be removed?** **Yes — frozen decision: WPT is removed from the core model.**

**Justification, directly addressing your Chain-B text:**

- *IoT battery state → WPT decision → energy recovery → task execution ability → latency*: this whole path is `[NOT SUPPORTED BY PROVIDED PAPERS]` — none of the four papers models an IoT device battery constraint on its ability to *execute or request* a task, so there is nothing in the base literature for this path to attach to.
- *Service caching → changed workload → changed energy demand → changed WPT requirement → changed UAV residual energy → changed mobility capability*: this chain is real (replication does change UAV energy expenditure — captured by the shared energy-budget term in Sections 8/14) **up to** "changed mobility capability," but the *WPT* link specifically is an unsupported, unnecessary detour — the same "more energy available for mobility" payoff is already captured directly through the energy-budget term, without needing to introduce a device-recharging mechanism at all.
- The only WPT-like mechanism in the entire base literature (P4's ground-to-UAV laser charging) recharges the **wrong endpoint** (the UAV, not the IoT device) for your framing, so it cannot be repurposed without inventing a new, unvalidated IoT-device-side energy-harvesting model from scratch.

**Consequence:** WPT, IoT-device battery state, and any UAV→device power-transfer action are excluded from the frozen system model (Section 6), decision variables (Section 7), objective (Section 8), and reward (Section 14). It may be noted as explicit **future work** in the paper's conclusion, clearly labeled `[PROPOSED EXTENSION, out of scope for this paper]`, but is not part of the locked specification.

---

## 17. BASELINES (frozen suite)

| # | Baseline | What it isolates |
| --- | --- | --- |
| 1 | **No caching/replication** | Whether replication matters at all — every task pays the anchor-pull cost. |
| 2 | **Static caching (P1-style)** | Whether *persistence without adaptivity* is enough, vs. our persistent-but-adaptive replication. |
| 3 | **Independent per-UAV caching, no replication coordination (P3-style)** | Whether *coordination* across UAVs (not just caching itself) is what matters — directly isolates the "fleet-wide" part of the novelty. |
| 4 | **Average-latency objective (P1-style), same MADRL architecture** | Whether the tail-latency objective specifically (not just the replication mechanism) is necessary to see the claimed effect. |
| 5 | **MADDPG, no attention, no masking (P3-style)** | Isolates the value of the attention+masking additions from Section 12/13, using a literature-grounded, apples-to-apples algorithm. |
| 6 | **MAPPO with attention, no action masking** | Isolates masking's specific contribution (soft penalty vs. hard infeasibility prevention). |
| 7 | **MAPPO with masking, no attention (fixed-size, padded/truncated observation)** | Isolates attention's specific contribution under the dynamic-population setting (Section 20). |
| 8 | **Proposed (full model)** | The frozen contribution. |

Each baseline changes exactly one axis relative to the proposed model, so no baseline is "unfair" (none removes something *and* something else at once) — directly satisfying your "no unfair baselines" instruction. Classical optimization (e.g., a Lyapunov+BCD scheme in the style of P2) is **not** included as a full competing baseline, because P2's own formulation has no replication-coordination or tail objective to compare against fairly — it would need to be re-derived as a new classical baseline, which is scoped instead as an optional Section-11-style "simple heuristic" sanity check (a greedy "replicate the top-$K$ most-locally-demanded services, associate greedily by nearest-with-replica" rule), not a full baseline entry.

---

## 18. ABLATION STUDY

| Ablation | Removes | Causal link tested |
| --- | --- | --- |
| A. No replication (single-UAV local caching only, no fleet coordination) | The core mechanism | Does replication *itself* drive the effect? |
| B. No cooperative/coordinated replication (independent per-UAV, as P3) | Only the *coordination* part, keeping replication | Isolates coordination's marginal contribution over plain replication |
| C. No attention (fixed-size padded/truncated observation) | Attention module | Tests whether attention is needed for the dynamic-population setting specifically |
| D. No action masking (penalty-only for all actions) | Hard masking | Tests whether hard masking meaningfully improves sample efficiency/feasibility over penalty-only, for the *discrete* actions specifically |
| E. Average-latency objective instead of CVaR | Tail objective | Tests whether the tail effect is visible under an average objective (expected: much weaker/invisible, per Section 8's reasoning) |
| F. No trajectory adaptation (fixed/random flight pattern) | Trajectory freedom | Tests whether trajectory freedom is the actual channel through which replication helps (the heart of Chain A) |
| G. Static active-user population (all devices always active) vs. dynamic | Dynamicity | Tests whether the dynamic-active-user setting is where the proposed method's advantage is largest (motivating Section 20) |

**The causal-chain-proving experiment (most important, as requested):** compare **A → B → Proposed**, holding the tail objective and trajectory freedom fixed across all three. If the hypothesis (Section 5) holds, tail latency should improve monotonically A \< B \< Proposed, **and** the improvement should be concentrated in devices spatially far from the anchor (measured via a latency-vs-distance-to-anchor breakdown), which is the specific, checkable signature of the spatial-de-coupling mechanism — not just "more caching is generally better."

---

## 19. EXPERIMENTAL DESIGN

**Independent variables:** # UAVs, # IoT devices, active-device ratio, task arrival rate, service-catalog size/popularity skew (Zipf parameter, as P3), replication cache capacity $Z_m$, UAV battery/energy budget, UAV max speed, per-UAV CPU capacity, network density (region size vs. device count).

**Dependent metrics:**

- *Primary:* CVaR$\_{0.95}$ latency, P95 latency, P99 latency.
- *Secondary:* average latency, task completion rate, deadline-adjacent metric (if reported at all, computed post-hoc from the tail distribution rather than a hard deadline constraint), energy consumption, UAV mobility distance, replica-hit ratio (local/peer/anchor breakdown), A2A traffic volume (migration + anchor-pull events), fairness (Jain's index), reward/training-convergence curves, wall-clock training/inference time.

**Plots/tables needed:**

- Latency-CDF plots (showing the *tail* specifically, not just a bar chart of averages) per baseline.
- **Latency vs. distance-to-anchor** scatter/binned plot — the specific evidence for the spatial-de-coupling mechanism.
- Replica-hit-path breakdown (stacked bar: local/peer/anchor) vs. replication coverage.
- Training-convergence curves (reward vs. episode) for MAPPO vs. MADDPG.
- Sensitivity sweeps (one plot per independent variable listed above) showing CVaR latency vs. that variable, proposed vs. baselines.
- Ablation bar chart (Section 18's seven ablations vs. proposed, on the primary metric).

---

## 20. CRITICAL EXPERIMENT: DYNAMIC ACTIVE USERS

**Physical positions fixed; only the active/inactive and request state changes**, exactly as required.

**Frozen active-user generation model** `[PROPOSED EXTENSION]`, chosen for being realistic and mathematically clean (per your instruction):

- Each stationary device $n$ toggles to **active** according to an independent (or hotspot-correlated, see below) **Poisson arrival process** with rate $\\lambda_n(t)$; once active, it holds exactly one outstanding task until served, then returns to inactive (a simple, standard on/off-with-Poisson-arrivals model, avoiding an over-engineered custom process).
- **Bursty component:** $\\lambda_n(t)$ is modulated by a slowly-varying hotspot indicator (a small number of regions periodically "flare up," raising local $\\lambda_n(t)$ for a bounded duration) — this creates the spatially-shifting demand that is the direct trigger for the spatial-de-coupling mechanism to matter (if demand were spatially uniform and static, there would be no "hotspot far from the anchor" case to exercise).
- **Service-popularity component:** reuses P3's Zipf distribution directly, but sampled per-hotspot rather than globally, so *which* service is hot also shifts with location (motivating why static/independent per-UAV replication, baseline 3/ablation B, underperforms coordinated replication).

**Why a fixed-dimensional policy is problematic here:** the number of simultaneously active devices visible to a given UAV varies slot-to-slot (from the Poisson/hotspot process above) — a fixed-length observation vector must either truncate (dropping real, currently-relevant tasks when a hotspot flares, exactly when the policy most needs to see them) or pad with dummy entries (wasting capacity and potentially confusing the network when few devices are active). This is precisely why the attention module of Section 12 exists — not as a generic scalability trick, but because *this specific* active-user generation model produces a genuinely variable-size input the policy must handle correctly at both extremes (quiet periods and hotspot bursts).

---

## 21. COMPLEXITY AND SCALABILITY

- **Centralized-optimization complexity:** the discrete replication+association+migration sub-problem alone is a coupled multi-knapsack/assignment problem, worst-case combinatorial in $M\\times|\\mathcal S|$ and in the active-task count — infeasible to solve exactly online at the timescales required, which is the standard justification (shared with P1's own move away from pure classical optimization for its analogous sub-problems) for a learned policy.
- **Action-space growth:** discrete heads scale with (bounded shortlist size $K$) rather than the full catalog $|\\mathcal S|$ (Section 13's top-$K$ shortlisting), keeping the discrete action space from growing unboundedly with catalog size.
- **Attention complexity:** standard self-/cross-attention is $O(n^2)$ in the number of visible active tasks $n$; since each UAV only attends to *locally visible* active tasks (a spatial locality cutoff, not the whole network), $n$ stays bounded by local device density rather than total device count $N$, keeping this tractable even as $N$ grows.
- **Training complexity:** CTDE means the centralized critic's input grows linearly with $M$ (concatenated per-agent aggregated representations) — manageable for the fleet sizes typical in this literature (single-digit to low tens of UAVs, consistent with P1/P2/P3/P4's own experimental scales).
- **Inference complexity:** decentralized execution means each UAV only runs its own actor forward pass at deployment, independent of fleet size $M$ — this is the standard, well-understood CTDE scalability argument, not a novel claim of this paper.

**Why attention + masking specifically help scalability here (not just "helps" vaguely):** attention keeps the *input representation* bounded and meaningful under a variable active-task count (Section 20); hard masking keeps the *discrete action space* from wasting learning capacity on exploring actions that are always infeasible by construction (e.g., migrating to a peer with no replica), which matters more as $M$ and $|\\mathcal S|$ grow and the fraction of *feasible* discrete actions shrinks.

---

## 22. SIMULATOR REQUIREMENTS

| Module | Equations / logic | Inputs | Outputs |
| --- | --- | --- | --- |
| Environment state | Concatenation of Section 12's per-UAV local + attended states | — | Full joint state |
| Reset | Randomize device positions (fixed for the episode), initial UAV positions, empty replication state | Seed, region/device-count config | Initial state |
| Task generation | Poisson/hotspot process (Section 20) | $\\lambda_n(t)$, hotspot schedule | Active-device set, per-task ($s,D_n,C_n$) |
| User activation/deactivation | On/off logic tied to task completion | Task-completion events | Updated active set |
| Service popularity | Per-hotspot Zipf (Section 20) | Hotspot state, catalog $\\mathcal S$ | Per-region service-demand weights |
| Caching/replication | $\\rho\_{m,s,t}$ update, capacity check, switch-cost accounting | Agent actions, $Z_m$ | Updated $\\rho$, incurred switch cost |
| Task queues | Per-UAV backlog (P1/P2-style queue update) | Association/offload decisions, $f\_{n,m,t}$ | Queue lengths, service completions |
| Communication | IoT↔UAV, UAV↔UAV, UAV↔anchor rate models (as P1/P3's LoS-style path-loss) | Positions, fixed tx power | Achievable rates |
| Computation | CPU-cycle delay (as P1/P3) | $f\_{n,m,t}$, $C_n$ | Per-task compute delay |
| Energy | Propulsion+comm+comp accounting against $E\_{\\text{budget}}$ | Actions, distances, rates | Remaining energy, penalty if exceeded |
| UAV movement | Position update from continuous action, bounded by $v\_{\\max}$ | $q\_{m,t}$ | New position |
| Collision checking | Pairwise minimum-distance check (as P1/P3) | All UAV positions | Violation flags for the reward penalty |
| Latency calculation | Section 15's conditional formula (local vs. A2A path) | Association/migration outcome, rates | Per-task $L_n(t)$ |
| Reward | Section 14's full expression | All of the above | Per-agent scalar reward |
| Episode termination | Fixed horizon $T$ (as all four base papers use slotted, finite/long-run horizons) | Slot counter | Done flag |

**Implementation order (phased, modified from your template to match the frozen model — no WPT phase, since it's removed):**

1. **Phase 1 — Basic environment:** static devices, fixed UAV positions, no caching, no movement; verify communication/computation delay formulas against P1/P3-style sanity checks.
2. **Phase 2 — Task generation:** Poisson arrivals, active/inactive toggling (no hotspots yet).
3. **Phase 3 — Caching/replication:** add $\\rho\_{m,s,t}$, capacity constraint, and the conditional latency formula (Section 15) — validate against the "No caching" and "Static caching" baselines.
4. **Phase 4 — Communication/computation, migration path:** add UAV↔UAV migration (P2-style) and UAV↔anchor pull (P3-style) as the two A2A resolution paths.
5. **Phase 5 — Energy:** add the bounded energy budget and its penalty term (no WPT).
6. **Phase 6 — Trajectory:** add continuous UAV movement, collision constraint.
7. **Phase 7 — Tail-latency metrics:** implement CVaR/P95/P99 computation over the active-device population per slot/episode.
8. **Phase 8 — Hotspots/dynamic popularity:** add the spatially-shifting demand model (Section 20) — this is when the core research question becomes testable.
9. **Phase 9 — MADRL (MAPPO, MADDPG baseline):** implement the CTDE training loop for both algorithms.
10. **Phase 10 — Attention:** add the variable-size active-task encoder.
11. **Phase 11 — Action masking:** add hard masking for the discrete heads.
12. **Phase 12 — Experiments:** run Sections 17–19's baselines, ablations, and sweeps.

---

## 23. MATHEMATICAL CONSISTENCY CHECK

- **Every symbol defined:** yes, in Sections 6–10; the notation table (Table II, Section 26) will formalize this for the paper itself.
- **Every decision variable appears meaningfully:** $\\rho,\\alpha,\\mu,f,q$ each appear in the objective (via latency/switch-cost/energy) and in at least one constraint — verified against Section 10.
- **Every objective term measurable:** CVaR/latency (directly computable from simulated task completions), switch cost (directly computable from consecutive $\\rho$ states), energy (directly computable from the energy-accounting module) — all measurable.
- **Every constraint physically meaningful:** capacity (Section 7/10), migration feasibility (requires the *target* to actually hold the replica — Section 10), CPU cap, speed/area/collision — all tied to a physical quantity, not an arbitrary bound.
- **Energy cannot go negative:** enforced by the energy-budget constraint with a penalty (Section 8/14) rather than a hard clip, consistent with how P1 handles its reflecting energy-queue boundaries — **flagged inconsistency to resolve during implementation:** a pure penalty (not a hard clip) could in principle let an episode's *simulated* energy value go negative before the penalty registers; the simulator (Phase 5) must clip energy at zero and treat further action as infeasible (mask further movement/comm actions) rather than only penalizing after the fact, to avoid a physically meaningless negative-energy state — this must be implemented as a **hard mask**, not left as a soft penalty, correcting the otherwise-soft treatment implied by Section 8's objective term.
- **Battery cannot exceed capacity:** not applicable in the same way, since there is no recharging in the frozen model (Section 16) — energy only depletes from a fixed per-episode budget, so an "exceeds capacity" case cannot occur by construction; **note this explicitly in the paper** so reviewers don't expect a charging/capacity-ceiling mechanism that the frozen model deliberately excludes.
- **UAV movement obeys velocity limits:** enforced by construction in the action space (bounded continuous action), with the soft penalty in Section 13 catching residual boundary/collision cases.
- **Cache cannot exceed capacity:** enforced by hard masking (Section 13), not just a constraint on paper.
- **CPU/bandwidth cannot exceed capacity:** CPU enforced via constraint (Section 10) and should be additionally hard-masked/clipped in the simulator (currently only a soft constraint in the math — **flagged inconsistency**: Section 7/10 state it as a hard constraint, but no explicit masking mechanism is defined for it in Section 13, unlike replication/migration/association; **resolution:** treat the CPU-allocation continuous action as constrained via a normalized-then-scaled projection (softmax over requested shares × $W_m$) rather than an unconstrained continuous action, guaranteeing the constraint by construction).
- **No WPT to check:** consistent by construction (removed, Section 16).
- **A task not simultaneously computed, cached, and migrated:** enforced by the "task exclusivity" constraint (Section 10) and by construction in the conditional latency formula (Section 15) — a task follows exactly one path per attempt.
- **No latency double-counting:** the conditional formula (Section 15) adds the A2A term *only* when the replica is absent, and the compute term always applies once — verified structurally consistent.
- **Replication does not violate cache constraints:** enforced by hard masking (Section 13).
- **Dynamic active-user modeling consistency:** the Poisson/hotspot model (Section 20) is a standard, well-defined stochastic process; consistent as specified.
- **Reward aligned with objective:** Section 14 was constructed term-for-term from Section 8's objective — verified aligned, with the caveat noted above about hard-masking energy at zero rather than only penalizing it.
- **Evaluation metrics not accidentally directly optimized unless intended:** CVaR is *both* the objective and a primary evaluation metric **by design** (Section 8 explicitly chooses this); P95/P99 are evaluation-only, never part of the reward (Section 14) — correctly *not* directly optimized, only reported, avoiding the "reward hacking toward a reported number" failure mode for those two.

**Summary of identified inconsistencies requiring resolution during implementation:** (1) energy non-negativity should be hard-masked, not only soft-penalized; (2) CPU-allocation should be constrained by construction (normalized projection), not left as an unconstrained continuous action with only a written-down constraint. Both are noted explicitly so they are fixed in Phase 5/Phase 9 of the simulator, not discovered late.

---

## 24. NOVELTY-CLAIM AUDIT

**CLAIMS WE CAN SAFELY MAKE:**

- We are not aware of prior work, among these four representative papers, that models fleet-wide, coordinated (vs. independent) service replication as a mechanism for reducing a UAV's spatial dependency on its service-acquisition sources.
- We introduce a device-level tail-latency (CVaR/P95/P99) objective to this specific caching+trajectory joint-optimization problem class, where the closest prior work (P1) optimizes average latency and the closest caching-fairness work (P3) optimizes workload distribution, not latency percentiles.
- We provide an ablation-based causal analysis (Section 18) isolating replication, coordination, attention, masking, and the tail objective as separately-contributing factors — a level of causal isolation not present in any of the four base papers' own evaluations.

**CLAIMS WE SHOULD NOT MAKE:**

- "First" or "novel" caching+trajectory joint optimization (P1, P3 already exist).
- "First" MADRL approach to UAV-MEC (P1's multi-agent SAC, P3's MADDPG, P4's MA-DDQN already exist).
- "Optimal" solution (MADRL provides a learned, not provably optimal, policy).
- "Universally scalable" (our complexity analysis, Section 21, is argued for the fleet/catalog sizes typical of this literature, not proven for arbitrary scale).
- Any claim involving WPT, IoT-device energy harvesting, or IoT-device battery-driven latency (explicitly out of scope, Section 16).
- Any claim of "solving" tail latency or fairness in UAV-MEC generally — only within this specific replication-driven mechanism and the frozen assumptions above.

---

## 25. FINAL IEEE PAPER STRUCTURE

| Section | Purpose | Content | Equations | Figures | Tables | Experiments referenced | Citations needed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I. Introduction | Motivate tail latency in dynamic UAV-MEC; state the gap and contribution | Disaster/dynamic-hotspot motivating scenario; 3 frozen contributions (Section 5) | None | Fig. 1 (system) | — | — | P1–P4 + general UAV-MEC/caching/MARL surveys |
| II. Related Work | Position against P1–P4 and broader caching/fairness/MARL literature | Sub-sections: UAV-MEC caching, tail-latency/fairness in edge computing, attention/masking in MARL | None | — | Table I (literature comparison) | — | P1–P4 + supporting MARL/attention/CVaR literature |
| III. System Model | Freeze entities/assumptions (Section 6) | IoT devices, UAVs, services, task/communication/mobility/energy models | System-model equations (rates, energy, queues) | Fig. 2 (replication/A2A interaction) | Table II (notation) | — | Channel-model and Lyapunov-technique citations (from P1/P2/P3) |
| IV. Problem Formulation | State the full optimization problem (Section 10) | Objective, constraints, discrete/continuous/non-convex/combinatorial/stochastic breakdown | Full formulation (Section 10) | Fig. 3 (optimization coupling diagram) | — | — | CVaR/risk-measure citations |
| V. Proposed MADRL Framework | Describe MAPPO+attention+masking (Sections 11–14) | Algorithm choice justification, state/action/reward, attention mechanism, masking mechanism | State/action/reward equations | Fig. 4 (MADRL architecture), Fig. 5 (attention), Fig. 6 (action masking) | — | — | MAPPO/attention/masking method citations |
| VI. Experimental Setup | Describe simulator, baselines, ablations, metrics (Sections 17–22) | Parameters, baseline/ablation definitions, dynamic active-user model | — | Fig. 7 (dynamic active-user scenario) | Table III (sim. parameters), Table IV (baseline definitions) | — | — |
| VII. Results and Discussion | Present and interpret results | CDF/tail plots, distance-to-anchor analysis, sensitivity sweeps, ablations | — | Fig. 8+ (results) | Table V (main results), Table VI (ablation results) | Sections 17–19 | — |
| VIII. Conclusion | Summarize contribution, state limitations and future work (incl. WPT as explicit future work) | Restate frozen contributions; explicit WPT/service-chain future-work note (Sections 15–16) | None | — | — | — | — |

---

## 26. REQUIRED FIGURES

- **Fig. 1 — System architecture:** UAVs, IoT devices (stationary, some active/inactive), anchor/registry, showing the three link types (IoT↔UAV, UAV↔UAV, UAV↔anchor).
- **Fig. 2 — Service replication and A2A interaction:** one UAV with a local hit (no A2A needed) vs. one without (migration or anchor-pull path), visually showing Section 15's conditional-latency mechanism.
- **Fig. 3 — Optimization coupling diagram:** the causal chain from Section 4's boxed novelty statement, rendered as a flow diagram.
- **Fig. 4 — MADRL architecture:** CTDE structure — per-UAV actor (local state + attention output → discrete/continuous heads, with masking applied), centralized critic.
- **Fig. 5 — Attention mechanism:** tokens (active nearby tasks) → embeddings → attention weights → aggregated vector, annotated with a concrete example (a hotspot device receiving high attention weight).
- **Fig. 6 — Action masking:** example showing a replication action masked out due to capacity, and a migration action masked out due to the target lacking a replica.
- **Fig. 7 — Example dynamic active-user scenario:** a timeline/heatmap showing hotspot flare-ups against fixed device positions (illustrating Section 20's generation model).
- **Fig. 8 — Latency CDF (tail focus)** comparing baselines 1–8 (Section 17).
- **Fig. 9 — Latency vs. distance-to-anchor**, the key mechanism-evidence plot.
- **Fig. 10 — Replica-hit-path breakdown** (local/peer/anchor) vs. replication coverage.
- **Fig. 11 — Training convergence** (MAPPO vs. MADDPG).
- **Fig. 12+ — Sensitivity sweeps** (one per key independent variable from Section 19).
- **Fig. N — Ablation results** (bar chart, Section 18).

---

## 27. REQUIRED TABLES

- **Table I — Literature comparison:** essentially the Section 3 gap matrix, formatted for the paper.
- **Table II — System notation:** every symbol from Sections 6–10.
- **Table III — Simulation parameters:** region size, $M$, $N$, $|\\mathcal S|$, $Z_m$, $W_m$, $v\_{\\max}$, $E\_{\\text{budget}}$, Zipf parameter, Poisson rates, hotspot schedule parameters, $\\beta$ (CVaR level), $\\eta_1,\\eta_2$.
- **Table IV — Baseline definitions:** Section 17's table, formatted for the paper.
- **Table V — Main results:** CVaR/P95/P99/average latency, task completion, energy, fairness — one row per baseline (Section 17).
- **Table VI — Ablation results:** Section 18's seven ablations vs. proposed, primary metric + key secondary metrics.

---

## 28. LOCKED RESEARCH SPECIFICATION

# LOCKED RESEARCH SPECIFICATION

**Title:** Refresh-Persistent Cooperative Service Replication for Spatially De-Coupled, Tail-Latency-Aware Multi-UAV MEC

**Research Question:** How can cooperative, fleet-wide service replication be jointly optimized with UAV trajectory and task association/offloading — via an attention-based, action-masked multi-agent DRL controller — to reduce the spatial coupling between a UAV's position and its service-acquisition dependencies, thereby reducing device-level tail latency (CVaR/P95/P99) for a dynamically changing set of active IoT tasks?

**Core Problem:** Independent, memoryless, per-UAV caching (P3) and unrelieved inter-UAV migration dependency (P2) leave a UAV's trajectory spatially pinned to its service sources; no existing formulation lets fleet-wide replication relax this coupling or targets the resulting benefit at the tail of the device-level latency distribution.

**Core Novelty:** Persistent, coordinated, fleet-wide service replication as the mechanism that reduces A2A/spatial dependency, freeing trajectory to track dynamically-active demand and disproportionately improving tail (not average) device-level latency.

**System:** Multiple UAVs, one anchor/registry (HAP-equivalent), stationary IoT devices, one atomic-service catalog, three link types (IoT↔UAV, UAV↔UAV, UAV↔anchor).

**IoT Assumption:** Stationary; no device mobility; no device energy/battery state (WPT removed).

**Dynamicity:** Physical positions fixed; active/inactive state and task arrivals vary via a Poisson + spatially-shifting-hotspot process (Section 20).

**Caching Mechanism:** Persistent, switch-cost-bearing, fleet-coordinated **service replication** (atomic services, not chains); extends P3's independent Docker-image caching into a coordinated multi-UAV placement problem with an explicit spatial-coupling latency term (Section 15).

**WPT Mechanism:** **None — explicitly removed** (Section 16). Not part of the model, objective, reward, or decision variables. Noted only as labeled future work.

**Energy Model:** Bounded per-slot/episode propulsion+communication+computation budget (soft-penalized in objective/reward, hard-masked at zero in the simulator per Section 23); no recharging infrastructure.

**Trajectory Model:** Continuous 2-D movement, bounded speed/area, inter-UAV collision constraint; movement is the channel through which spatial de-coupling manifests (Section 15).

**Offloading Model:** Local (replica present) / migrate-to-peer (replica present at a reachable peer) / anchor-pull (replica absent) — one exclusive path per task (Section 15).

**Computation Model:** Per-UAV CPU budget, normalized-projection allocation across served tasks (Section 23's consistency fix).

**Primary Objective:** Minimize CVaR$\_{0.95}$ of device-level per-task latency + replication switch cost + energy penalty (Section 8).

**Primary Tail/Fairness Metric:** CVaR$\_{0.95}$ of per-device latency (primary, optimized).

**Secondary Metrics:** P95, P99 latency; Jain's fairness index; average latency; task completion rate; energy consumption; replica-hit-path breakdown; A2A traffic volume.

**Decision Variables:** $\\rho\_{m,s,t}$ (replication, discrete, persistent), $\\alpha\_{n,m,t}$ (association, discrete), $\\mu\_{n,mm',t}$ (migration, discrete), $f\_{n,m,t}$ (CPU share, continuous, constrained-by-construction), $q\_{m,t}$ (trajectory, continuous).

**Constraints:** Cache capacity, migration feasibility (target must hold replica), CPU capacity, speed/area/collision, energy budget (hard at zero), task exclusivity, non-negativity (Section 10).

**RL Paradigm:** Multi-agent, CTDE (centralized training, decentralized execution), cooperative shared tail-latency reward component.

**Primary MARL Algorithm:** **MAPPO** (justified in Section 11 specifically by its robustness to the non-stationary active-task population, via on-policy training).

**Secondary/Baseline Algorithm:** **MADDPG** (literature-grounded via P3; also serves as the "no attention/no masking" ablation stand-in).

**Attention:** Cross-/self-attention over currently-visible active-task tokens → fixed-size aggregated representation, concatenated with fixed-size local state (Section 12); purpose is explicitly to prioritize under-served, currently-active, non-replicated-nearby devices.

**Action Masking:** Hard, logit-level masking for replication (capacity), migration (target must hold replica), and association (task must be active/visible); soft, penalty-based handling for trajectory boundary/collision (Section 13).

**State:** Per-UAV: position, velocity, energy, CPU utilization, replication-state vector, plus attention-aggregated active-task representation (Section 12).

**Action:** Discrete: replicate/evict (shortlisted), associate/migrate/decline. Continuous: 2-D trajectory velocity.

**Reward:** Negative CVaR-latency term (shared/cooperative) − switch-cost term − energy term − constraint-violation penalties (Section 14).

**Baselines:** No caching; static caching (P1-style); independent per-UAV caching (P3-style); average-latency objective; MADDPG-no-attention-no-masking; MAPPO-no-masking; MAPPO-no-attention; proposed (Section 17).

**Ablations:** No replication; no coordination; no attention; no masking; average-objective instead of CVaR; no trajectory adaptation; static vs. dynamic active population (Section 18).

**Main Experiments:** Sensitivity sweeps over # UAVs, # devices, active ratio, arrival rate, catalog size/skew, cache capacity, energy budget, speed, CPU capacity, density (Section 19); the distance-to-anchor mechanism-evidence experiment (Section 19); the dynamic-active-user critical experiment (Section 20).

**Expected Causal Story:** Replication ↑ → local-hit rate ↑ / A2A dependency ↓ → trajectory increasingly demand-driven rather than anchor/peer-driven → tail (far-from-anchor, hotspot) devices' latency improves most → CVaR/P95/P99 improve more than average latency does, and more than in the independent-caching or average-objective baselines.

**Required Simulator Modules:** Section 22's full module table; 12-phase implementation order, WPT phase removed.

**Required Equations:** System-model rates/energy/queues (Section 6/10), conditional latency formula (Section 15), objective (Section 8), reward (Section 14), CVaR estimator.

**Required Figures:** Section 26's 12+ figures.

**Required Tables:** Section 27's six tables.

**Paper Structure:** Section 25's eight-section IEEE structure.

**Potential Threats to Validity:** Effect size may be small in low-device-density or spatially-uniform-demand settings (no hotspot far from anchor to exercise the mechanism); CVaR estimated from finite per-slot batches may be noisy at small active-device counts, requiring careful batch-size/episode-length tuning; MAPPO vs. MADDPG comparison could be confounded by implementation/tuning effort rather than a fundamental algorithmic difference unless both are tuned with comparable effort; the switch-cost weight $\\eta_1$ materially affects whether replication behaves "persistently" as intended — requires a dedicated sensitivity check, not just a fixed default.

**Claims We Can Make:** Section 24's "safely make" list.

**Claims We Must Avoid:** Section 24's "should not make" list.