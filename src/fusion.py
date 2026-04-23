def combine_scores(rule_results, ml_results):
    all_diseases = set(ml_results.keys())
    if rule_results:
        all_diseases.update(rule_results.keys())

    combined = {}
    for disease in all_diseases:
        rule_score = rule_results.get(disease, {}).get("confidence", 0) if rule_results else 0
        ml_score = ml_results.get(disease, 0)

        if rule_score > 70:
            final = (0.6 * rule_score) + (0.4 * ml_score)
        else:
            final = (0.4 * rule_score) + (0.6 * ml_score)

        combined[disease] = round(final, 2)

    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
