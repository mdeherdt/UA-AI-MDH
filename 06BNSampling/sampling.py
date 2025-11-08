import random
from bn import DiscreteBN

def prior_sample(bn: DiscreteBN, rng: random.Random):
    """
        Generate one full sample from the Bayesian network using prior sampling.

        Go through all variables in topological order.
        For each variable X, sample a value according to P(X | parents(X))
        using the already sampled values of its parents.

        Returns:
            dict[str, str]: A full assignment mapping each variable name to its sampled state.
                            Example: {"asia": "no", "tub": "no", "smoke": "yes", ...}
        """
    sample = {}
    for X in bn.topo_order:
        states = bn.states[X]
        probs = [bn.local_prob(X, s, sample) for s in states]
        r = rng.random()
        cum = 0.0
        for s, p in zip(states, probs):
            cum += p
            if r <= cum:
                sample[X] = s
                break
    return sample


def rejection_sampling(bn: DiscreteBN, query, evidence, N, rng):
    """
    Estimate the probability P(query_var = query_val | evidence) using rejection sampling.

    Algorithm outline:
      generate N samples and only keep those that match the evidence.
      Return the fraction of kept samples where the query is true.

    Parameters:
        bn (DiscreteBN): The Bayesian network object.
        query (tuple):  (variable_name, value) to estimate, e.g. ("either", "yes").
        evidence (dict): Observed variables, e.g. {"xray": "yes"}.
        N (int): Number of samples to generate.
        rng (random.Random): Random number generator instance.

    Returns:
        float: An estimate of P(query | evidence)
               If no samples match the evidence, you may return 0.
    """
    """TODO: implement this function"""

    consistent = 0
    match = 0

    for _ in range(N):
        sample = bn.sample_prior(rng) # return assignment dict

        is_consistent = True

        for var,value in evidence.items():
            if sample[var] != value:
                is_consistent = False
                break

        if is_consistent:
            consistent += 1
            if sample[query[0]] == query[1]:
                match += 1

    if sample == 0:
        return None
    return match / consistent



def likelihood_weighting(bn: DiscreteBN, query, evidence, N, rng):
    """
    Estimate the probability P(query_var = query_val | evidence) using likelihood weighting.

    Algorithm outline:
      Generate N weighted samples where evidence variables are fixed
      to their observed values. Combine these weighted samples to
      estimate the conditional probability, making sure to normalize
      at the end.


    Parameters:
        bn (DiscreteBN): The Bayesian network object.
        query (tuple):  (variable_name, value) to estimate, e.g. ("either", "yes").
        evidence (dict): Observed variables, e.g. {"xray": "yes"}.
        N (int): Number of weighted samples to generate.
        rng (random.Random): Random number generator instance.

    Returns:
        float: An estimate of P(query | evidence)
               (Normalized weighted probability for the query variable being its target value.)
    """
    total_weight = 0.0
    query_weight = 0.0

    query_var = query[0]
    query_val = query[1]

    for _ in range(N):
        sample_weight = 1.0
        assignment = {}

        for X in bn.topo_order: #itereert over alle variabelen in topologische volgorde
            if X in evidence:
                x_val = evidence[X]
                assignment[X] = x_val

                p = bn.local_prob(X, x_val, assignment)

                sample_weight *= p

            else:
                states = bn.states[X]
                probs = [bn.local_prob(X, s, assignment) for s in states]
                r = rng.random() # willekeurig getal tussen 0 en 1
                cum = 0.0 # cumulatieve kans
                for s, p in zip(states, probs):
                    cum += p
                    if r <= cum:
                        assignment[X] = s
                        break
        total_weight += sample_weight

        if assignment[query_var] == query_val:
            query_weight += sample_weight

    if total_weight == 0:
            return 0.0

    return query_weight / total_weight