import pandas as pd
import csv
from apyori import apriori

with open("proj1.csv", "r") as f:
    csv_reader = csv.reader(f)

    transactions = list(csv_reader)

rules = apriori(transactions, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)


def prepare_output(results):
    Support, Confidence, Lift, Items, Antecedent, Consequent = [], [], [], [], [], []

    for RelationRecord in results:
        for ordered_stat in RelationRecord.ordered_statistics:
            Support.append(RelationRecord.support)
            Items.append(set(RelationRecord.items))
            Antecedent.append(set(ordered_stat.items_base))
            Consequent.append(set(ordered_stat.items_add))
            Confidence.append(ordered_stat.confidence)
            Lift.append(ordered_stat.lift)

    df = pd.DataFrame({
        'Items': Items,
        'Antecedent': Antecedent,
        'Consequent': Consequent,
        'Support': Support,
        'Confidence': Confidence,
        'Lift': Lift
    })

    return df


results = prepare_output(rules)

sorted_results = results.sort_values("Lift", ascending=False)

print(sorted_results)
