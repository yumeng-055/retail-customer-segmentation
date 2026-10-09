# Data dictionary

The original course-provided dataset is included as [`raw/customer_segmentation.csv`](raw/customer_segmentation.csv). It contains 2,000 customer records and 15 fields. Only the filename has changed; the CSV contents are unchanged.

The dataset supports a simulated retail decision brief about category offers, channel engagement and membership design. It is a customer-level snapshot: the clustering describes observed spending and visits, rather than predicting purchases or estimating profitability.

The pipeline and Notebook use this file by default. An alternative CSV location can be supplied with `--data`.

| Fields | Meaning and units | Role |
| --- | --- | --- |
| customer_id | Customer identifier, one row per customer | Alignment checks only |
| gender, marital_status, education_level | Male/Female; Single/Married/Divorced; High School/Bachelor/Master/PhD | Profiles |
| age, num_children | Age in years, number of children | Profiles |
| monthly_income | Monthly income in SGD | Contextual purchasing capacity |
| spend_grocery, spend_fashion, spend_electronics, spend_dining, spend_online | Approximate average monthly spending in SGD | Clustering and profiles |
| visits_store_per_month, visits_online_per_month | Average monthly channel visits | Clustering and profiles |
| loyalty_score | Score constructed from visits and spending; observed range checked | Profiles only |

Online spending may overlap category spending. These five spending fields are not summed into revenue or total spending. The assignment does not document a sampling frame, observation period, transaction history, or the exact loyalty formula. No profitability or lifetime-value measure is available.

Source acknowledgment: the MAV552 January 2026 customer segmentation assignment at Singapore University of Social Sciences supplies the academic scenario and field definitions. The data and assignment are third-party course materials; no project license is applied to them.

## Reporting definitions

- **Observation unit:** one customer in the supplied cross-sectional snapshot. There is no documented transaction-level history, campaign exposure, country or observation-period field.
- **Model inputs:** eight features in fixed order: income, five spending fields, store visits and online visits. Demographics and loyalty are used for profiles.
- **Profile outputs:** numeric medians and means in original units after stated imputation; categorical counts and within-segment proportions. Segment shares use all 2,000 customers.
- **Money and channel measures:** monthly income and approximate average monthly spending are SGD values; visits are monthly averages. Spending overlap prevents a defensible combined revenue measure.
- **Outcome coverage:** visits and spending describe behavior. Conversion, repeat purchase, retention, incremental margin and experimental effects require additional event, transaction, assignment and cost data.
