"""Simulation code for "When Should AI Follow? Task Structure and Joint Adaptation
by Human and AI Agents".

Two agents adapt over binary decision sequences and differ in a single respect, the
memory regime that maps past decisions into the next decision:

* AI agent, uniform-memory (UM) adaptation: the next decision is 1 if the mean of the
  previous ``K`` decisions is at least 0.5, and 0 otherwise (ties resolve to 1).
* Human agent, recency-weighted (RW) adaptation: the previous ``K`` decisions receive
  linearly increasing weights (1 = oldest, K = most recent). The next decision is 1 if
  the weighted mean is at least 0.5, and 0 otherwise (ties resolve to 1).
* Memoryless stochastic adaptation (labelled "hallucination" in the output files): each
  decision is drawn uniformly at random, ignoring history. Used only in the HTA structure.

The first ``K`` states of an agent that starts without inheritance are drawn i.i.d.
Bernoulli(0.5) and are part of its sequence (and of its payoff).

Task structures
---------------
1. Modular (C = 0): the two agents work independently and their payoffs are averaged.
2. Sequential, AI-to-Human (ATH): the AI agent's sequence is built first. Its FIRST
   ``C`` states (which, whenever C <= K_AI, are the AI's random initial seed) become the
   first ``C`` states of the human sequence, and the human continues with RW adaptation
   over a window of length ``C``.
3. Sequential, Human-to-AI (HTA): the human sequence is built first. Its FIRST ``C``
   states become the first ``C`` states of the AI sequence, and the AI continues with UM
   adaptation over a window of length ``C``. In "hallucination" mode every AI decision
   at position ``N_Human`` or later (0-based) is replaced by a Bernoulli(0.5) draw; the
   positions between ``C`` and ``N_Human`` still follow the UM rule.

Parameters: N = task scope (sequence length), K = within-task coupling (window length),
C = cross-agent coupling. Subscripts AI / Human denote the agent.

Conventions of the code as run for the paper
--------------------------------------------
* Payoff (``PO``): mean over positions of the parity
  (sum mod 2) of ``K + 1`` consecutive, wrap-around decisions, with K the agent's
  within-task coupling. For K = 0 this equals the share of 1s. For K >= 1 it does not:
  an all-ones sequence scores 1 when K is even and 0 when K is odd.
* ``LP`` / ``NLP`` is the number of positions at least as large as both circular
  neighbours. For a binary sequence this is (#1s) + (#0s whose two neighbours are 0),
  so it scales almost one-for-one with sequence length.
* Joint payoff (``APO``) is the average of the two agents' payoffs.
* Downstream window = ``C``. In ATH the human payoff uses K = C and ``K_Human`` is
  reported as 0. In HTA ``K_AI`` is fixed at 0, so the AI payoff is its share of 1s.
* Sequences with K <= 0 are not updated and are all zeros (these rows are part of the
  grids below).
* Parameter grids: N_AI in 1..50 and N_Human in 1..10 (so N_AI/N_Human runs from 0.1
  to 50, including N_AI <= N_Human); modular: 0 <= K < N for both agents; ATH:
  0 <= K_AI < N_AI and 1 <= C < N_Human; HTA: K_AI = 0, 0 <= K_Human <= N_Human and
  1 <= C < N_Human.
* HTA update mode: drawn once per parameter combination per iteration with
  probability 1/2 each.

Usage
-----
    python nkc_models.py                        # all three structures, one seeded run
    python nkc_models.py --case hta --iterations 10 --seed 7 --outdir results
    python nkc_models.py --iterations 1000 --no-sequences   # smaller CSVs for many runs

Each structure writes one CSV (overwritten on every run) to ``--outdir``.
"""

# %% [markdown]
# # When Should AI Follow? Simulation code
#
# Companion code for *When Should AI Follow? Task Structure and Joint Adaptation by
# Human and AI Agents*. See the repository README for the model conventions, the output
# format and how each figure is produced (``make_figures.py``).

# %%
import argparse
import os
import random

import pandas as pd

# Parameter grids (see "Conventions of the code as run for the paper" in the docstring).
N_AI_MAX = 50
N_HUMAN_MAX = 10

OUTPUT_FILES = {
    "modular": "modular_simulation_results.csv",
    "ath": "sequential_simulation_results_ath.csv",
    "hta": "simulation_results_hta.csv",
}

SEQUENCE_COLUMNS = ("UA_AI", "UA_Human")

# %% [markdown]
# ## Adaptation rules (memory regimes)

# %%
def _random_seed_window(k, rng):
    """Random binary initial history of length k, i.i.d. Bernoulli(0.5)."""
    return [rng.choice([0, 1]) for _ in range(k)]


def uniform_memory_sequence(n, k, initial_seed=None, rng=random):
    """AI agent: build a length-``n`` sequence with uniform-memory (UM) adaptation.

    The next decision is 1 if the mean of the previous ``k`` decisions is >= 0.5.
    ``initial_seed`` (length ``k``) is the inherited history; if it is missing or has the
    wrong length, a random binary seed is drawn. With ``k <= 0`` there is no updating
    and the all-zero sequence is returned.
    """
    if k <= 0:
        return [0] * n

    if initial_seed is None or len(initial_seed) != k:
        initial_seed = _random_seed_window(k, rng)

    result = list(initial_seed)
    while len(result) < n:
        window = result[-k:]
        mean = sum(window) / k
        result.append(1 if mean >= 0.5 else 0)
    return result


def recency_weighted_sequence(n, k, initial_seed=None, rng=random):
    """Human agent: build a length-``n`` sequence with recency-weighted (RW) adaptation.

    The i-th oldest decision in the window of the previous ``k`` decisions has weight
    ``i`` (1..k). The next decision is 1 if the weighted sum is >= half of the total
    weight ``k(k + 1)/2``. Seeding and the ``k <= 0`` case follow
    :func:`uniform_memory_sequence`.
    """
    if k <= 0:
        return [0] * n

    if initial_seed is None or len(initial_seed) != k:
        initial_seed = _random_seed_window(k, rng)

    result = list(initial_seed)
    total_weight = sum(range(1, k + 1))
    while len(result) < n:
        window = result[-k:]
        weighted_sum = sum((i + 1) * window[i] for i in range(k))
        result.append(1 if weighted_sum >= total_weight / 2 else 0)
    return result


# %% [markdown]
# ## Payoff and local peaks

# %%
def compute_fitness(array, k):
    """Payoff of a sequence: mean parity of ``k + 1`` consecutive (wrap-around) decisions.
    Equals the share of 1s only when ``k = 0``."""
    n = len(array)
    if n == 0:
        return 0

    total_fitness = 0
    for i in range(n):
        interacting_values = [array[(i + j) % n] for j in range(k + 1)]
        total_fitness += sum(interacting_values) % 2
    return total_fitness / n


def count_local_peaks(array):
    """Number of positions at least as large as both circular neighbours."""
    n = len(array)
    if n == 0:
        return 0

    peak_count = 0
    for i in range(n):
        left_neighbor = array[i - 1] if i > 0 else array[n - 1]
        right_neighbor = array[i + 1] if i < n - 1 else array[0]
        if array[i] >= left_neighbor and array[i] >= right_neighbor:
            peak_count += 1
    return peak_count


def compute_payoff(array, k):
    """Payoff (``PO``) and number of local peaks (``NLP``) of a sequence."""
    return {"PO": compute_fitness(array, k), "NLP": count_local_peaks(array)}


# %% [markdown]
# ## 1. Modular (C = 0)
#
# AI (UM) and Human (RW) agents work independently; joint payoff is the average.

# %%
def simulate_modular(iteration, n_ai, k_ai, n_human, k_human, rng=random):
    """One modular run for the given scopes and coupling depths (C = 0)."""
    ua_ai = uniform_memory_sequence(n_ai, k_ai, rng=rng)
    ua_human = recency_weighted_sequence(n_human, k_human, rng=rng)

    po_ai = compute_payoff(ua_ai, k_ai)
    po_human = compute_payoff(ua_human, k_human)

    return {
        "Simulation": iteration,
        "N_AI": n_ai, "K_AI": k_ai, "N_Human": n_human, "K_Human": k_human, "C": 0,
        "LP_AI": po_ai["NLP"],
        "PO_AI": po_ai["PO"],
        "UA_AI": ua_ai,
        "LP_Human": po_human["NLP"],
        "PO_Human": po_human["PO"],
        "UA_Human": ua_human,
        "APO_Modular": (po_ai["PO"] + po_human["PO"]) / 2,
    }


def run_modular(iteration, rng=random):
    """All modular configurations with K_AI < N_AI and K_Human < N_Human."""
    configs = [
        (n_ai, k_ai, n_human, k_human)
        for n_ai in range(1, N_AI_MAX + 1) for k_ai in range(0, n_ai)
        for n_human in range(1, N_HUMAN_MAX + 1) for k_human in range(0, n_human)
    ]
    return pd.DataFrame(
        [simulate_modular(iteration, *config, rng=rng) for config in configs]
    )


# %% [markdown]
# ## 2. Sequential: AI-to-Human (ATH)
#
# The AI agent adapts first. Its first ``C`` decisions seed the human agent, whose
# window length is ``C``.

# %%
def simulate_ath(iteration, n_ai, k_ai, n_human, c, rng=random):
    """One ATH run: the AI sequence is built first and its first ``c`` decisions seed
    the human sequence (window ``c``). ``K_Human`` is reported as 0 and ``C`` as ``c``."""
    ua_ai = uniform_memory_sequence(n_ai, k_ai, rng=rng)

    if c <= 0 or c > len(ua_ai):
        ua_human = [0] * n_human
    else:
        ua_human = recency_weighted_sequence(
            n_human, c, initial_seed=ua_ai[:c], rng=rng
        )

    po_ai = compute_payoff(ua_ai, k_ai)
    po_human = compute_payoff(ua_human, c)

    return {
        "Simulation": iteration,
        "N_AI": n_ai, "K_AI": k_ai, "N_Human": n_human, "K_Human": 0, "C": c,
        "LP_AI": po_ai["NLP"],
        "PO_AI": po_ai["PO"],
        "UA_AI": ua_ai,
        "LP_Human": po_human["NLP"],
        "PO_Human": po_human["PO"],
        "UA_Human": ua_human,
        "APO_Sequential_ATH": (po_ai["PO"] + po_human["PO"]) / 2,
    }


def run_ath(iteration, rng=random):
    """All ATH configurations with K_AI < N_AI and 1 <= C < N_Human."""
    configs = [
        (n_ai, k_ai, n_human, c)
        for n_ai in range(1, N_AI_MAX + 1) for k_ai in range(0, n_ai)
        for n_human in range(1, N_HUMAN_MAX + 1) for c in range(1, n_human)
    ]
    return pd.DataFrame(
        [simulate_ath(iteration, *config, rng=rng) for config in configs]
    )


# %% [markdown]
# ## 3. Sequential: Human-to-AI (HTA), with Hallucination
#
# The human agent adapts first. Its first ``C`` decisions seed the AI agent, whose window
# length is ``C``. Beyond the human's task scope (positions after ``N_Human``) the AI
# agent either follows its UM rule (``rule-based``) or switches to memoryless stochastic
# adaptation (``hallucination``).

# %%
def simulate_hta(iteration, n_ai, k_ai, n_human, k_human, c,
                 ai_updation="rule-based", rng=random):
    """One HTA run.

    ``ai_updation`` is ``"rule-based"`` (AI keeps following its UM rule) or
    ``"hallucination"`` (AI decisions from position ``n_human`` onward are drawn
    uniformly at random).
    """
    # Step 1: the human generates its sequence.
    ua_human = recency_weighted_sequence(n_human, k_human, rng=rng)

    # Step 2: the AI starts from the first c human decisions and continues with UM
    # adaptation (window c).
    seed_for_ai = ua_human[:c]
    ua_ai = uniform_memory_sequence(n_ai, c, initial_seed=seed_for_ai, rng=rng)

    # Step 3: optionally replace the AI's decisions beyond the human's scope by
    # memoryless random draws.
    if len(ua_ai) > n_human and ai_updation == "hallucination":
        for i in range(n_human, n_ai):
            ua_ai[i] = rng.choice([0, 1])

    # Step 4: payoffs.
    po_ai = compute_payoff(ua_ai, k_ai)
    po_human = compute_payoff(ua_human, k_human)

    return {
        "Iteration": iteration,
        "N_AI": n_ai, "K_AI": k_ai, "N_Human": n_human, "K_Human": k_human, "C": c,
        "NLP_AI": po_ai["NLP"],
        "PO_AI": po_ai["PO"],
        "UA_AI": ua_ai,
        "NLP_Human": po_human["NLP"],
        "PO_Human": po_human["PO"],
        "UA_Human": ua_human,
        "APO": (po_ai["PO"] + po_human["PO"]) / 2,
        "AI_Updation_Type": ai_updation,
    }


def run_hta(iteration, rng=random):
    """All HTA configurations with K_AI = 0, 0 <= K_Human <= N_Human, 1 <= C < N_Human.

    For each configuration the AI update mode is "rule-based" or "hallucination" with
    probability 1/2 each.
    """
    configs = [
        (n_ai, 0, n_human, k_human, c)
        for n_ai in range(1, N_AI_MAX + 1)
        for n_human in range(1, N_HUMAN_MAX + 1)
        for k_human in range(0, n_human + 1)
        for c in range(1, n_human)
    ]

    results = []
    for n_ai, k_ai, n_human, k_human, c in configs:
        ai_updation = "rule-based" if rng.random() < 0.5 else "hallucination"
        results.append(
            simulate_hta(iteration, n_ai, k_ai, n_human, k_human, c,
                         ai_updation=ai_updation, rng=rng)
        )
    return pd.DataFrame(results)


# %% [markdown]
# ## Running the experiments

# %%
RUNNERS = {"modular": run_modular, "ath": run_ath, "hta": run_hta}


def run_case(case, iterations=1, outdir="results", seed=0, keep_sequences=True):
    """Run ``iterations`` replications of one task structure and write a single CSV.

    The output file is overwritten, so repeated invocations never append to stale
    results. ``keep_sequences=False`` drops the ``UA_AI`` / ``UA_Human`` columns (the
    full decision sequences) to keep files small for many iterations; it does not change
    any other value. Returns the path of the CSV.
    """
    rng = random.Random(seed)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, OUTPUT_FILES[case])
    if os.path.exists(path):
        os.remove(path)

    for iteration in range(1, iterations + 1):
        df = RUNNERS[case](iteration, rng)
        if not keep_sequences:
            df = df.drop(columns=list(SEQUENCE_COLUMNS))
        df.to_csv(path, mode="a", header=(iteration == 1), index=False)
        print(f"[{case}] iteration {iteration}/{iterations} done ({len(df)} rows)")
    return path


# %%
def main():
    parser = argparse.ArgumentParser(
        description="Simulate modular, ATH and HTA human-AI task structures."
    )
    parser.add_argument("--case", choices=["modular", "ath", "hta", "all"],
                        default="all", help="task structure to simulate (default: all)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="replications per structure (default: 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--outdir", default="results",
                        help="output directory (default: results)")
    parser.add_argument("--no-sequences", action="store_true",
                        help="omit the UA_AI / UA_Human sequence columns from the CSVs")
    args = parser.parse_args()

    cases = list(RUNNERS) if args.case == "all" else [args.case]
    for case in cases:
        path = run_case(case, args.iterations, args.outdir, args.seed,
                        keep_sequences=not args.no_sequences)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()