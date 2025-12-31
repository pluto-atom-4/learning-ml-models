from pathlib import Path
from machine_learning.logistic_guesstimate import (
    load_insurance_data,
    guesstimate_logistic_coefficients,
)
from visualizations.logistic_plots import (
    plot_data_only,
    plot_with_logistic_curve,
)


def main() -> None:
    csv_path = Path("../../generated/data/raw/insurance_claim.csv")

    x, y = load_insurance_data(csv_path)

    # Run guesstimation
    head = guesstimate_logistic_coefficients(x, y, iterations=12)

    # Best estimate is at the head of the linked list
    best = head
    print(
        f"Best guesstimate → β0={best.beta0:.3f}, "
        f"β1={best.beta1:.3f}, accuracy={best.accuracy:.3f}"
    )

    # Plot raw data
    plot_data_only(x, y, title="Insurance Claim vs Age")

    # Plot logistic curve with best coefficients
    plot_with_logistic_curve(
        x,
        y,
        beta0=best.beta0,
        beta1=best.beta1,
        title="Best Logistic Guesstimate Fit",
    )


if __name__ == "__main__":
    main()
