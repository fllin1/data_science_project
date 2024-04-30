import seaborn as sns
import matplotlib.pyplot as plt


def house_price(dataset):
    plt.figure(figsize=(9, 8))
    sns.histplot(dataset['SalePrice'], color='g', bins=100, kde=True, alpha=0.4)
    plt.savefig("output/house_prices")


def evalate_model(logs):
    plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("RMSE (out-of-bag)")
    plt.savefig("output/model_evalutation")
    plt.show()


def variable_weight(inspector):
    plt.figure(figsize=(12, 4))

    # Mean decrease in AUC of the class 1 vs the others.
    variable_importance_metric = "NUM_AS_ROOT"
    variable_importances = inspector.variable_importances()[variable_importance_metric]

    # Extract the feature name and importance values.
    #
    # `variable_importances` is a list of <feature, importance> tuples.
    feature_names = [vi[0].name for vi in variable_importances]
    feature_importances = [vi[1] for vi in variable_importances]
    # The feature are ordered in decreasing importance value.
    feature_ranks = range(len(feature_names))

    bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
    plt.yticks(feature_ranks, feature_names)
    plt.gca().invert_yaxis()

    # TODO: Replace with "plt.bar_label()" when available.
    # Label each bar with values
    for importance, patch in zip(feature_importances, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

    plt.xlabel(variable_importance_metric)
    plt.title("NUM AS ROOT of the class 1 vs the others")
    plt.tight_layout()
    plt.savefig("output/variable_importances")
    plt.show()
