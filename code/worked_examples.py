"""Figure 2 worked example, computed with the simulation's own adaptation rules.

    python worked_examples.py

The same realized history (1, 1, 0, 0), oldest to most recent, is passed to the
recency-weighted rule (human archetype) and the uniform-memory rule (AI archetype)
with a window of K = 4. The two rules reach different next decisions.
"""
import nkc_models as m


def figure2(history=(1, 1, 0, 0)):
    history = list(history)
    k = len(history)
    rw = sum((i + 1) * x for i, x in enumerate(history)) / sum(range(1, k + 1))
    um = sum(history) / k
    next_rw = m.recency_weighted_sequence(k + 1, k, initial_seed=history)[-1]
    next_um = m.uniform_memory_sequence(k + 1, k, initial_seed=history)[-1]
    print("Figure 2: history (oldest to most recent) =", tuple(history))
    print(f"  recency-weighted (weights 1..{k}): weighted mean = {rw:.2f} -> next decision {next_rw}")
    print(f"  uniform-memory   (equal weights): mean          = {um:.2f} -> next decision {next_um}")


if __name__ == "__main__":
    figure2()
