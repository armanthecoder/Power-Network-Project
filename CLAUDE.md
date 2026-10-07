# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pygame-based simulation of multi-robot network formation: a swarm of robots grows a tree-shaped
communication network from a fixed `SOURCE` out to several `Sink` locations, splitting into
STAR/PORT branches at `PIVOT` nodes and dynamically reconfiguring (releasing/re-recruiting branches)
as sinks get touched and charge/discharge over time.

## Running

```
python main.py
```

- Visualization is controlled by `VISUALIZE_ON` in [constants.py](constants.py). With it `True` (default),
  a pygame window renders the sim live; with it `False`, `main()` runs headless until all sinks are
  touched or `MAX_SIM_TIME` is hit, then only `generate_report(...)` output is produced.
- `run_simulation(pivot_threshold, max_sim_time=80.0)` in [main.py](main.py) is a second, headless entry
  point parameterized by pivot threshold (returns `(network_complete_time, total_robots)`), intended for
  sweeping that threshold across many runs — it isn't currently called from anywhere else in this tree.
- Dependencies are `pygame` and `matplotlib`; there is no `requirements.txt` — install with
  `pip install pygame matplotlib`.
- There is no test suite and no linter/formatter configuration in this repo.

Every run ends by calling `generate_report(...)` ([post_processing.py](post_processing.py)), which writes
a new timestamped folder at the repo root named `{NUM_SINKS}sinks_{R}R_{timestamp}/` containing a text
report, a JSON state dump, and two matplotlib plots (full topology + Steiner skeleton). The many
`2sinks_*`/`3sinks_*` directories already in the repo root are accumulated output from past runs, not
source — they are safe to delete/regenerate and are not referenced by any code.

## Architecture

Four files form the simulation core, split by concern:

- **[constants.py](constants.py)** — the `Robot`, `Sink`, `Obstacle`, `Message` dataclasses, the
  `Role` (`MOVING`/`NETWORK`/`SOURCE`) and `NodeType` (`STRAIGHT`/`PIVOT`) enums, and every tunable
  parameter (void-controller gains `K_R`/`K_THETA`, pivot thresholds, boid flocking weights, sink
  layout, spawn rate). Tune simulation behavior here rather than hardcoding values in the logic files.
- **[assistive_functions.py](assistive_functions.py)** — all per-robot decision logic, i.e. what a
  robot does each tick:
  - `robot_step` dispatches to `_free_behavior` (boid flocking + guidance lock-on toward a recruiting
    network robot) for `MOVING` robots, or `_attached_behavior` (void/polar controller holding
    `desired_R`/`desired_dir_d` relative to its parent) for `NETWORK` robots.
  - `enforce_pivot_split` promotes a robot to `PIVOT`, splits its `sink_indices` into STAR/PORT sides
    via `split_sinks_star_port`, reassigns/reorients children, and releases duplicate children back to
    `MOVING`.
  - `calc_pivot_metric` / `is_local_pivot_candidate` score candidate pivots by how close their
    star/port branch angles are to a target angle (120° by default, or geometry-derived in
    `pivot_single_sink`); `main.py`'s loop uses these scores each tick to decide which robots to
    actually turn into pivots.
  - `remove_sink_from_subtree` / `release_branch_to_moving` / `tick_pivot_cooldowns` implement
    reconfiguration: once a subtree's assigned sinks are all touched/removed, its robots reset to
    `MOVING` and the vacating pivot gets a cooldown (`pivot_cooldown`) before it can recruit on that
    side again.
  - `deliver_messages` is a simple broadcast-within-`COMM_RANGE` mailbox (outbox → inbox); hop counts
    for `MOVING` robots are recomputed every tick via BFS from `NETWORK` robots in `compute_hop_counts`.
- **[main.py](main.py)** — owns the simulation loop: spawns flocks (`spawn_flock`), advances sink
  `charging_level` (a sink charges when touched, discharges otherwise — this currently drives
  reconfiguration, see below), computes per-robot pivot errors/scores and enforces pivot splits,
  delivers messages, steps every robot, then draws (if visualizing) or reports (on exit).
- **[post_processing.py](post_processing.py)** — `generate_report`, shared verbatim between both
  implementation variants (see below).

### Two parallel implementation variants

[Basic implementation/](Basic implementation/) is an earlier, separate snapshot of the same four files
(`main.py`, `constants.py`, `assistive_functions.py`, `post_processing.py` — the latter identical to the
root copy). It implements the originally-committed reconfiguration design: robots relay a
`BATTERY_REPORT` message up to their pivot (`robot.battery_reports["STAR"/"PORT"]`), and the pivot prunes
whichever branch has higher charge once both sides have reported. The root-level files are the actively
evolving variant — `Robot.battery_reports` still exists on the dataclass but is currently unused there;
instead, reconfiguration is driven by the simpler global `Sink.charging_level` model advanced directly in
`main.py`'s loop, combined with the branching-angle pivot score described above. When changing
reconfiguration/pruning logic, check which variant (root vs. `Basic implementation/`) is actually in
scope for the task — they diverge and are not meant to be kept in sync automatically.

### Traitor agents (Dutch-auction reconfiguration)

A third mover state, `Role.TRAITOR`, sits between `MOVING` and `NETWORK`: a settled `NETWORK` robot
whose own sink has already been touched can defect from its branch, travel like a mover, and become
`NETWORK` again wherever it next gets recruited. This is a continuous, score-driven reallocation of
already-finished capacity toward whichever sink is currently most charge-starved — it runs alongside,
not instead of, the pivot-formation logic described above.

- **Eligibility & auction** (`is_traitor_eligible`, `hop_count_to_own_sink`, `chain_toucher_dwell_ticks`,
  `pivot_distance`, all in [assistive_functions.py](assistive_functions.py)): once per tick, the auction
  tops up the number of robots currently holding `Role.TRAITOR` to `MAX_TRAITORS` (default `1`, but can be
  raised to run several defections concurrently). Every settled `STRAIGHT` `NETWORK` robot carrying
  exactly one sink qualifies if that sink is *either* currently touched *or* already charged to
  `TRAITOR_ELIGIBLE_CHARGE_THRESHOLD` or above. The charge-level fallback matters under robot scarcity:
  `compute_touched_sinks` only scans `Role.NETWORK` robots, so if a chain's own tip is the one that
  defects, `touched[]` flips False the instant it leaves — without the charge fallback, that would
  permanently strand the rest of the chain (never touched live again with no spare robot around to
  refill the tip, so never eligible again either). Each eligible robot is then scored against every other
  untouched sink by: `W_BATTERY` × that sink's charge deficit, `W_HOP` × inverse hop-count down to the
  sink-touching leaf of its own chain (closer to the sink = cheaper to lose; 0 if the chain is currently
  disconnected, since there's no leaf to measure to), `W_PIVOT` × inverse pivot-count between it and the
  candidate sink (discourages port-end ↔ starboard-end jumps), minus a flat `W_HYST` hysteresis term and
  a `W_DWELL`-scaled dwell penalty that protects a *just*-touched sink's whole chain from being
  immediately re-raided (without this, the highest-scoring candidate is always the literal toucher, which
  un-touches its own sink the instant it leaves — see `MIN_DWELL_TICKS`). The dwell penalty is a cliff,
  not a gentle taper: `W_DWELL * max(0, MIN_DWELL_TICKS - dwell)` stays far larger than every other term
  combined for nearly the whole window, then collapses to 0 in the last ~20 ticks before
  `MIN_DWELL_TICKS` — a chain is either fully untouchable or fully fair game, with almost no middle
  ground, and once dwell exceeds `MIN_DWELL_TICKS` the penalty is permanently 0 (long-settled chains are
  *more* available to donate from, not protected). The winning (robot, sink) pair converts via
  `convert_to_traitor` if its score clears `TRAITOR_SCORE_THRESHOLD`; when `MAX_TRAITORS > 1`, each
  additional winner in the same tick is picked by re-running the same scan excluding sinks already
  claimed that tick (`claimed_sids`) so concurrent defectors spread across different targets rather than
  piling onto one. Pulling a mid-chain (non-leaf) robot splices its child directly onto its old parent
  (`convert_to_traitor`'s splice) and puts that spliced node on a `pivot_cooldown` so it doesn't instantly
  recruit a replacement — a short `traitor_grace` window similarly stops the fresh defector from
  immediately re-accepting its own just-vacated slot.
- **Movement** (`_traitor_behavior`): identical to a free mover's reattachment, recruitment-response, and
  guide selection (`_try_accept_network_offer` / `_respond_to_recruitment`, shared with `_free_behavior`),
  with two differences layered on top. First, guide selection ranks candidates lexicographically by
  `(relevant, branch_val, -dist)` — a guide whose `sink_indices` contains the candidate sink always beats
  one that doesn't, regardless of branch depth or proximity; only among equally-relevant guides does the
  usual branch-depth/distance tie-break apply (without this, a Traitor near several guides could lock onto
  a deeper-but-irrelevant one purely by branch-depth, triggering unnecessary retreat/reapproach cycling).
  Second, once a guide is chosen: if its `sink_indices` doesn't contain the candidate sink *and* its own
  branch isn't already fully done, follow the reverse of its guidance instead of forward (so it doesn't
  retrace into the sink it already left); a guide whose branch is done is followed forward regardless,
  since it poses no such risk. The reverse direction is computed relative to the Traitor's own current
  position (`robot.pos - guidance_dir * R`), not anchored at the guide's position — anchoring at the guide
  produces a fixed nearby point to hover around forever instead of actually retreating (this was a real
  bug: a Traitor got permanently stuck for the rest of a 900s run before the fix). At a `PIVOT`,
  `guidance_dir_from_robot` takes an optional `candidate_sid` and, when given, returns whichever side
  (star/port) actually leads to that sink directly — bypassing its normal generic recruitment-priority
  logic, which has nothing to do with where one specific candidate sink is. If the current candidate sink
  gets touched by someone else first, it re-picks the best remaining untouched sink, gated by a short
  `candidate_cooldown` so it can't flip-flop between two sinks tick-to-tick.
- **Not-permanent exclusions**: a defector remembers `old_sink_id` (the sink it just left) and
  `old_parent_id` (the parent it just left), both off-limits as a re-pick/re-accept target — but only for
  `OLD_SINK_REVISIT_COOLDOWN_TICKS` / `OLD_PARENT_REVISIT_COOLDOWN_TICKS` respectively, not permanently.
  A permanent ban on `old_parent_id` caused a real bug: if nothing else was around, a Traitor could end up
  permanently refusing to refill the exact slot it just vacated. Both cooldowns clear the corresponding
  field back to `None` when they expire, after which that sink/parent is fair game again like any other.
- **Charge-biased pivot guidance**: separately, when a `PIVOT`'s star and port sides both still need
  growth, `guidance_dir_from_robot` steers ordinary (non-Traitor) recruitment toward whichever side's
  sinks have the lower **sum** of `charging_level` (not average — a side with more sinks counts for more,
  so it's weighted by total need rather than per-sink need), instead of a fixed default side. This only
  affects recruitment *order* — pivot positions and the STAR/PORT split itself stay purely geometric
  (`split_sinks_star_port`), so the tree's eventual shape is unchanged by charge dynamics; only the
  timing of which side fills first is affected.
- **Sink charge dynamics** (`main.py`'s main loop, not `run_simulation()` — see note below): each tick,
  every touched sink gains `SINK_CHARGE_RATE` and every untouched sink loses `SINK_DISCHARGE_RATE`, each
  applied as a flat per-sink rate — not a shared budget divided across however many sinks are
  simultaneously touched (the original `0.02 / n_touched` formula diluted toward parity with the 0.01
  discharge rate whenever 2+ sinks were touched at once; `SINK_CHARGE_RATE` is kept at 2x
  `SINK_DISCHARGE_RATE` regardless of how many sinks are touched).
- All Traitor/auction tunables (`W_BATTERY`, `W_HOP`, `W_PIVOT`, `W_HYST`, `W_DWELL`,
  `MIN_DWELL_TICKS`, `TRAITOR_SCORE_THRESHOLD`, `TRAITOR_ELIGIBLE_CHARGE_THRESHOLD`,
  `TRAITOR_GRACE_TICKS`, `SPLICE_COOLDOWN_TICKS`, `CANDIDATE_SWITCH_COOLDOWN_TICKS`,
  `OLD_SINK_REVISIT_COOLDOWN_TICKS`, `OLD_PARENT_REVISIT_COOLDOWN_TICKS`, `SINK_CHARGE_RATE`,
  `SINK_DISCHARGE_RATE`, `MAX_TRAITORS`) live in [constants.py](constants.py) alongside the pivot
  thresholds. Setting `W_HOP` to 0 (or `TRAITOR_ELIGIBLE_CHARGE_THRESHOLD` very low) removes the only
  term that discourages picking the literal, just-settled tip over other candidates — observed to cause
  the same robot to rapidly ping-pong between two sinks with almost no travel in between; the
  dwell/hysteresis terms alone don't fully compensate. Because `W_BATTERY`'s term (`0`–`2` at the default
  weight) dwarfs `W_HOP`/`W_PIVOT`'s (each well under `0.5` at typical weights), those two terms mostly
  only matter as tiebreaks between near-equal battery deficits, not as primary decision factors — keep
  their weights closer to `W_BATTERY`'s scale if they're meant to meaningfully shape which robot gets
  pulled.
- The original `SOURCE` phase-1 (single-sink trunk until an 80%-charge threshold) → phase-2 (branching)
  mechanism has been removed: the trunk now carries all sinks from its first robot, like every other
  branch, so `calc_pivot_metric` can form a pivot immediately instead of waiting on a threshold that
  blocked branching entirely while active.
- Under genuine robot scarcity (not enough total robots to simultaneously serve every sink), the auction
  can end up perpetually rotating the spare robot(s) among all the needy sinks forever (observed as a
  clean round-robin across all three sinks in a 3-sink layout) — this is expected behavior under real
  resource contention, not a bug to dampen further.
- This feature is root-level only; it has not been ported to [Basic implementation/](Basic implementation/).