"""
Naive Bayes Category Classifier (PGM Concept)
----------------------------------------------
Implements a Naive Bayes Classifier — a fundamental Probabilistic Graphical Model (PGM).

Bayesian Network Structure:
    
         [Category]          ← Parent Node (INSTITUTIONAL / COMMITTEE)
         /    |    \
        /     |     \
    [Tag1] [Tag2] [Tag3]     ← Child Nodes (conditionally independent given Category)

Bayes' Theorem:
    P(Category | Tag1, Tag2, ..., TagN) ∝ P(Category) × ∏ P(Tagi | Category)

Where:
- P(Category) = Prior probability of each category
- P(Tagi | Category) = Likelihood of observing tag i given the category
- The product assumes conditional independence of tags given category (Naive Bayes assumption)

This classifier predicts whether an uploaded image belongs to an INSTITUTIONAL
or COMMITTEE event based on the AI-generated tags from CLIP.
"""

import math

# ══════════════════════════════════════════════════════════════════
# CONDITIONAL PROBABILITY TABLE (CPT)
# ══════════════════════════════════════════════════════════════════
# P(Tag | Category) — These are learned/hand-crafted likelihood values.
# In a real system, these would be estimated from historical data using MLE.
#
# Each value represents: "Given this category, what is the probability of
# this tag appearing?"

TAG_LIKELIHOODS = {
    "INSTITUTIONAL": {
        # High probability tags for institutional events
        "auditorium": 0.70, "seminar": 0.75, "graduation": 0.80,
        "award ceremony": 0.75, "stage": 0.60, "presentation": 0.70,
        "faculty": 0.65, "building": 0.50, "crowd": 0.55,
        "group photo": 0.50, "students": 0.55, "classroom": 0.60,
        "poster": 0.45, "cultural event": 0.55, "celebration": 0.50,
        # Low probability tags for institutional events
        "coding": 0.15, "hackathon": 0.10, "lab": 0.30,
        "sports": 0.20, "dance": 0.35, "music": 0.35,
        "food": 0.25, "workshop": 0.40, "competition": 0.30,
        "outdoor": 0.30, "nature": 0.20, "night event": 0.30,
        "portrait": 0.30, "single person": 0.25, "selfie": 0.20,
        "group of people": 0.50,
    },
    "COMMITTEE": {
        # High probability tags for committee events
        "coding": 0.75, "hackathon": 0.80, "workshop": 0.70,
        "competition": 0.70, "lab": 0.60, "sports": 0.65,
        "dance": 0.65, "music": 0.65, "food": 0.60,
        "night event": 0.60, "outdoor": 0.55, "group of people": 0.60,
        "celebration": 0.55, "cultural event": 0.60, "selfie": 0.50,
        # Low probability tags for committee events
        "auditorium": 0.35, "seminar": 0.25, "graduation": 0.10,
        "award ceremony": 0.25, "stage": 0.40, "presentation": 0.30,
        "faculty": 0.15, "building": 0.25, "crowd": 0.45,
        "group photo": 0.50, "students": 0.50, "classroom": 0.25,
        "poster": 0.40, "portrait": 0.40, "single person": 0.35,
        "nature": 0.35,
    }
}

# ══════════════════════════════════════════════════════════════════
# PRIOR PROBABILITIES — P(Category)
# ══════════════════════════════════════════════════════════════════
# These represent our belief about category distribution before seeing any evidence.
# Assume equal priors (uninformative prior / maximum entropy principle).

PRIOR_PROBABILITIES = {
    "INSTITUTIONAL": 0.5,
    "COMMITTEE": 0.5
}

# Laplace smoothing parameter (prevents zero-probability problem)
SMOOTHING_ALPHA = 0.1
DEFAULT_LIKELIHOOD = 0.3  # For unseen tags


def classify_category(tags: list[str]) -> dict:
    """
    Naive Bayes Classification using Bayes' Theorem.
    
    Given a set of observed tags, computes the posterior probability 
    for each category and returns the most likely category.
    
    Mathematical formulation (using log-probabilities for numerical stability):
        log P(C|T1,...,Tn) = log P(C) + Σ log P(Ti|C)
    
    Args:
        tags: List of AI-generated tags from CLIP (evidence / observed variables)
        
    Returns:
        Dictionary with:
        - predicted_category: The MAP (Maximum A Posteriori) estimate
        - confidence: Posterior probability of the predicted category
        - posteriors: Full posterior distribution over all categories
        - reasoning: Human-readable explanation of the classification
    """
    if not tags:
        return {
            "predicted_category": "INSTITUTIONAL",
            "confidence": 0.5,
            "posteriors": {"INSTITUTIONAL": 0.5, "COMMITTEE": 0.5},
            "reasoning": "No tags provided — defaulting to uniform prior."
        }
    
    # ─── Step 1: Compute log-posterior for each category ───
    # Using log-space to prevent floating-point underflow
    # log P(C|evidence) ∝ log P(C) + Σ log P(tag_i | C)
    
    log_posteriors = {}
    tag_contributions = {}
    
    for category in PRIOR_PROBABILITIES:
        # Start with log prior: log P(Category)
        log_posterior = math.log(PRIOR_PROBABILITIES[category])
        contributions = []
        
        for tag in tags:
            tag_lower = tag.lower().strip()
            # Get P(tag | category) from our CPT, with Laplace smoothing
            likelihood = TAG_LIKELIHOODS.get(category, {}).get(tag_lower, DEFAULT_LIKELIHOOD)
            # Apply Laplace smoothing
            smoothed_likelihood = (likelihood + SMOOTHING_ALPHA) / (1 + 2 * SMOOTHING_ALPHA)
            
            log_likelihood = math.log(smoothed_likelihood)
            log_posterior += log_likelihood
            
            contributions.append({
                "tag": tag_lower,
                "P(tag|{})".format(category): round(likelihood, 3),
                "log_contribution": round(log_likelihood, 4)
            })
        
        log_posteriors[category] = log_posterior
        tag_contributions[category] = contributions
    
    # ─── Step 2: Convert log-posteriors to normalized probabilities ───
    # Using the log-sum-exp trick for numerical stability
    max_log = max(log_posteriors.values())
    
    # Subtract max for numerical stability, then exponentiate
    posteriors = {}
    total = 0.0
    for category, log_p in log_posteriors.items():
        posteriors[category] = math.exp(log_p - max_log)
        total += posteriors[category]
    
    # Normalize to get valid probability distribution (sums to 1)
    for category in posteriors:
        posteriors[category] = round(posteriors[category] / total, 4)
    
    # ─── Step 3: MAP Estimate (Maximum A Posteriori) ───
    predicted = max(posteriors, key=posteriors.get)
    confidence = posteriors[predicted]
    
    # ─── Step 4: Generate human-readable reasoning ───
    reasoning_parts = [f"Observed tags: {tags}"]
    reasoning_parts.append(f"Prior P(INSTITUTIONAL) = {PRIOR_PROBABILITIES['INSTITUTIONAL']}")
    reasoning_parts.append(f"Prior P(COMMITTEE) = {PRIOR_PROBABILITIES['COMMITTEE']}")
    reasoning_parts.append(f"Posterior P(INSTITUTIONAL|tags) = {posteriors['INSTITUTIONAL']}")  
    reasoning_parts.append(f"Posterior P(COMMITTEE|tags) = {posteriors['COMMITTEE']}")
    reasoning_parts.append(f"MAP Estimate: {predicted} (confidence: {confidence:.1%})")
    
    result = {
        "predicted_category": predicted,
        "confidence": confidence,
        "posteriors": posteriors,
        "reasoning": " | ".join(reasoning_parts)
    }
    
    print(f"[PGM] ══════════════════════════════════════════")
    print(f"[PGM] Naive Bayes Classification Result:")
    print(f"[PGM]   Tags:       {tags}")
    print(f"[PGM]   P(INST|T):  {posteriors['INSTITUTIONAL']}")
    print(f"[PGM]   P(COMM|T):  {posteriors['COMMITTEE']}")
    print(f"[PGM]   Predicted:  {predicted} ({confidence:.1%})")
    print(f"[PGM] ══════════════════════════════════════════")
    
    return result
