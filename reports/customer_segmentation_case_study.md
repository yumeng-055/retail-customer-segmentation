# Retail Customer Segmentation and Marketing Strategy

## Executive summary

**Recommendation:** use four K-Means customer segments as a working framework for differentiated marketing pilots, with K=3 retained as a simpler comparison. Four profiles make grocery-oriented family purchasing and higher-income omnichannel activity separately visible, although K=3 has stronger internal clustering metrics. Whether that extra distinction improves business outcomes remains a question for testing.

The analysis covers **2,000 customers** and eight clustering features, with five principal components retaining **81.57%** of variance. The largest segment, fashion and technology shoppers, contains **744 customers (37.2%)**, with median online spending of **SGD 900/month** and **13 online visits/month**. This gives a clear starting hypothesis for an on-platform recommendation pilot. Family-focused shoppers have the highest median grocery spending; high-income omnichannel shoppers have higher spending across several categories and activity in both channels; low-engagement shoppers provide a separate repeat-purchase test group.

**Project scope:** an individual academic customer-analytics project with a simulated business brief. The business context describes a decision scenario, while the numerical evidence comes from executed analysis of course-provided data. The deliverables are a reproducible Python reporting workflow, verified customer profiles, model-selection evidence and a proposed KPI/experimental measurement specification. No retailer engagement, deployed system or measured campaign uplift is claimed.

## Business context and decision brief

**Simulated business setting.** An omnichannel retailer wants to plan differentiated category offers, customer engagement and membership benefits. Its customer file combines purchasing patterns, income and visits across physical and online channels. The simulated brief asks an analyst to move beyond a single broad marketing audience and give the marketing manager a practical way to define test groups.

The scenario represents a business decision problem; it does not document an actual retailer's campaign performance, budgets or operating practices. All numerical findings below come from the course-provided customer data.

| Decision gap in the simulated brief | Business question | Analytical response |
| --- | --- | --- |
| A broad audience offers little guidance on which products to emphasize | Which category-spending patterns warrant different offers? | Group customers using spending, channel visits and purchasing capacity; describe each group in original units |
| The appropriate engagement channel is unclear | Which groups show stronger online activity or activity in both channels? | Compare monthly online/store visits and online spending by segment |
| More segments add planning complexity | Does a fourth group support a meaningful additional decision? | Compare K=3 and K=4 using internal metrics, stability, memberships and business profiles |

The primary objective is to recommend an interpretable working segmentation and translate it into testable marketing hypotheses. The intended stakeholder is a marketing or customer-insights manager deciding which audiences, offers and channels to pilot. Success at this analytical stage means transparent profiles, justified model choices and a reproducible workflow. Commercial success would need purchase, exposure and cost data from a later controlled test.

## Data overview and source

The MAV552 January 2026 assignment at Singapore University of Social Sciences provides the course dataset and scenario. One row represents a customer; spending is approximate average monthly SGD, income is monthly SGD, and visits are monthly channel averages. The assignment's consulting scenario is an academic framing. The source CSV is included; the original report and assignment paper are retained separately. See [data definitions and availability](../data/README.md).

## Reproducible reporting and data-quality controls

The Python workflow converts the customer snapshot into a consistent set of segment profiles, diagnostic tables and English figures. A single command regenerates these outputs; the Notebook uses the same functions. This reduces dependence on separately edited spreadsheets or manually maintained chart values. Reporting is triggered by running the analysis, with no scheduled refresh or production service implemented.

| Reporting concern | Implemented control | Evidence available for review |
| --- | --- | --- |
| Input records may be incomplete or inconsistent | Required fields, numeric types, missingness, duplicate IDs/rows, finite values and category checks; explicit plausibility flags | Data-quality, data-check and observed-range tables |
| Transformations may obscure units or customer identity | Separate imputed original-unit profile frame; fixed model-feature order; checks for unchanged observed values and preserved customer order | Preprocessing functions and integrity checks |
| Segment reports may disagree on counts or definitions | Shares calculated from verified counts; medians and means exported separately; categorical counts and within-segment denominators provided | Segment overview, profile and proportion tables |
| A model change may produce misleading comparisons | Candidate/final partition checks, label-invariant ARI/NMI comparisons and customer-membership contingency tables | Partition-agreement and contingency tables |
| Outputs may be difficult to reproduce or hand over | Modular functions, explicit seeds/model settings, pinned packages, one command and focused tests | Source modules, complete Notebook and root execution instructions |

The analysis uses one supplied CSV with 2,000 customer records. It demonstrates local data preparation and reporting controls; warehouse modeling, SQL transformations, multi-source ingestion and large-scale processing are outside the implemented workflow.

For a marketing stakeholder, the [segment overview](tables/segment_overview.csv) answers who the groups are, the recommendations explain what to test, and the measurement specification below defines what evidence would support a rollout. For a technical reviewer, source functions and comparison tables expose how the profiles were produced. These artifacts are designed for handoff; no real cross-functional team collaboration is claimed by this individual project.

## Data preparation and analytical design

The dataset contains 2,000 customers and 15 fields. There are 200 missing cells: 50 each in income, children, electronics spending and online spending (2.5% per field). Checks found no duplicate rows or customer IDs, unsupported category values, negative observed numeric values, or nonfinite observed numeric values. Age and loyalty are checked against explicit plausibility ranges; these checks do not certify source accuracy. Income reaches SGD 999,999 per month and store visits reach 100 per month. These are unverified extremes or possible placeholders, retained without capping or deletion. The exported audit includes counts, ranges and category frequencies.

Income and children use median imputation; electronics and online spending use mean imputation. Imputed children are cast to integers. This assumes the observed central values are reasonable substitutes, without evidence that missingness is random. A separate imputed frame retains original units for profiling. The model frame applies `log1p` to income and store visits, then StandardScaler to the eight features in a fixed order. Logs compress differences at high values; scaling makes distance reflect standardized feature differences. Neither guarantees removal of outlier influence. Distribution and IQR summaries show their effects without removing observations.

Clustering uses income, five spending fields, and two channel-visit fields. Income represents purchasing capacity. The identifier and demographics are excluded to focus segmentation on purchasing patterns; demographics are summarized afterward. Loyalty is excluded because it is constructed from spending and visits, reducing redundancy. Pairwise correlations alone cannot establish absence of multicollinearity. Online spending is analyzed separately because overlap with category spending is undocumented.

PCA is fit to all eight transformed standardized features (`random_state=64`, no whitening, `svd_solver='auto'`). Five score columns retain 81.57% of variance. This is a compression tradeoff, not a guarantee of retained business value. The exported weights are `components_.T`, not component weights multiplied by explained standard deviation. Signs are arbitrary and do not change distances. No clustering in alternate representations is evaluated here.

PC1 combines fashion (0.435), dining (0.414), online spending (0.416), online visits (0.410) and other purchasing-capacity features. PC2 contrasts grocery (0.572) and store visits (0.553) with online visits (-0.324). PC3 emphasizes electronics (0.744) against fashion (-0.404) and store visits (-0.357). These are feature combinations, not customer segments.

![PCA variance](figures/pca_variance.png)
![PCA component weights](figures/pca_weights.png)

## Model selection

K-Means candidates cover K=2 through K=10 using k-means++ initialization, `n_init=20`, `random_state=64`, 300 maximum iterations, tolerance 0.0001 and Lloyd's algorithm. Final K=4 and the K=3 comparison use `n_init=30` with the same other settings. In this execution, the candidate and final partitions are identical at K=3 and K=4 (ARI=NMI=1). Inertia, CH and DB use the complete five-PC representation. Silhouette uses a seed-64 sample of 1,000 customers.

| K   | Silhouette   | Calinski-Harabasz   | Davies-Bouldin   | Minimum share   |
|:----|:-------------|:--------------------|:-----------------|:----------------|
| 3   | 0.336        | 1,011.484           | 1.164            | 0.217           |
| 4   | 0.326        | 923.144             | 1.222            | 0.165           |

Higher silhouette and CH, and lower DB, favor K=3. K=4 is an interpretability choice: it distinguishes a grocery-oriented family group from higher-income customers with greater dining, fashion and electronics spending. K=3's broader higher-capacity group has median income SGD 9,500, grocery spending SGD 900, and dining spending SGD 700. K=4 makes those two patterns separately visible in the original-unit profiles below. The K=3/K=4 contingency table documents the actual membership relationship; the solutions need not form a perfectly nested split. Balanced size alone does not establish better segments. There is no campaign-outcome evidence that four groups outperform three.

![Candidate K-Means metrics](figures/kmeans_candidates.png)

K-Means subsampling stability fits 12 independent 80% subsets without replacement, seeds 1000 through 1011 and `n_init=20`. Each later fit is compared to the first on correctly aligned shared row indices (11 comparisons). K=4 mean ARI is 0.974 (SD 0.012); mean NMI is 0.961 (SD 0.017). Initialization stability fits all customers 12 times with seeds 2000 through 2011 and `n_init=1`, comparing later runs to the first (11 comparisons). Its K=4 mean ARI/NMI is 0.986/0.976. Standard deviations use population `ddof=0`. These statistics measure agreement, not ground-truth accuracy. Subset runs retain the full-data scaler and PCA; stability therefore does not measure re-estimation of the entire preprocessing pipeline.

## Ward comparison

SciPy Euclidean/Ward linkage uses all 2,000 rows in the same five-PC representation. K=2 through K=8 are evaluated with `fcluster(..., criterion='maxclust')`. The dendrogram uses the complete tree without sampling. Cophenetic correlation describes correspondence between tree merge distances and original distances, not business validity.

| model               | silhouette   | CHI     | DBI   | minClusterRatio   |
|:--------------------|:-------------|:--------|:------|:------------------|
| K-Means (n_init=30) | 0.326        | 923.144 | 1.222 | 0.165             |
| Ward (SciPy)        | 0.305        | 844.971 | 1.267 | 0.116             |

The K=4 Ward profiles show broadly similar purchasing patterns after matching clusters by maximum customer-membership overlap. Matched median income is SGD 8,400 for the family pattern and SGD 15,725 for the higher-income pattern. Cross-method ARI/NMI is 0.823/0.787; memberships are similar, not identical. The aggregate contingency table and matched profiles expose the differences. SciPy and sklearn Euclidean/Ward partitions are equivalent up to relabeling at K=4 in this environment (ARI=NMI=1).

Ward stability uses 10 independent 80% subsets, seeds 64 through 73, compared to the full-data Ward partition restricted to each subset's indices. At K=4 mean ARI/NMI is 0.826/0.813, with SD 0.031/0.017 (`ddof=0`). This reference scheme and run count differ from K-Means. Ward's silhouette uses 2,000 customers while K-Means uses 1,000; neither the silhouette nor stability gap is a fully controlled comparison. These observed results support K-Means as a practical working segmentation, with no universal claim that Ward is more or less stable or robust. Both methods assign extremes to groups rather than identifying noise. A fitted K-Means model could assign future customers after matching preprocessing; no production assignment or drift monitoring system has been implemented or validated.

![Ward dendrogram](figures/ward_dendrogram.png)
![Stability under the stated protocols](figures/stability.png)

## Customer profiles

Counts and shares use all 2,000 customers. Numeric profiles below are within-segment medians in original units after imputation. Separate exported tables provide means and categorical modes; categorical proportion tables provide counts, denominators and shares for every observed level. Medians summarize each feature separately and do not represent a single customer.

| Segment                          | Customers   | Share   | Median income (SGD/month)   | Median store visits/month   | Median online visits/month   |
|:---------------------------------|:------------|:--------|:----------------------------|:----------------------------|:-----------------------------|
| Low-engagement shoppers          | 414         | 20.7%   | 3,550                       | 3                           | 3                            |
| Family-focused shoppers          | 512         | 25.6%   | 8,050                       | 6                           | 5                            |
| Fashion and technology shoppers  | 744         | 37.2%   | 6,300                       | 2                           | 13                           |
| High-income omnichannel shoppers | 330         | 16.5%   | 15,550                      | 6                           | 10                           |

Median category and online spending (SGD per month; columns may overlap):

| Segment                          | Grocery   | Fashion   | Electronics   | Dining   | Online   |
|:---------------------------------|:----------|:----------|:--------------|:---------|:---------|
| Low-engagement shoppers          | 600       | 200       | 250           | 300      | 250      |
| Family-focused shoppers          | 1,000     | 400       | 500           | 600      | 500      |
| Fashion and technology shoppers  | 500       | 600       | 550           | 600      | 900      |
| High-income omnichannel shoppers | 750       | 650       | 750           | 900      | 700      |

Low-engagement shoppers have the lowest median income and low spending and channel visits. Family-focused shoppers have higher grocery spending, a median of one child and 68.4% married customers (350/512); the name describes an observed pattern rather than every household. Fashion and technology shoppers have median age 30, stronger fashion/electronics spending than the low-engagement and family groups, the highest online spending and online visits, and fewer store visits. High-income omnichannel shoppers have the highest median income, high spending across several categories, and substantial activity in both channels. Income and spending are not profit or lifetime value.

Demographics should qualify the profiles. In the fashion/technology segment, married customers comprise 47.2% (351/744) and single customers 45.2% (336/744), so it is not predominantly single. The high-income segment is 52.7% male (174/330), a modest majority. Educational distributions are reported without assigning personality or motives. General online visits do not reveal social-media platform use.

![Segment sizes](figures/segment_sizes.png)
![Original-unit profile comparison](figures/profile_heatmap.png)

The heatmap first computes segment medians in original units, then standardizes each feature across the four medians using sample SD (`ddof=1`). It is not a customer-level standardized mean heatmap. Read the original-unit table for spending and visits. PC1/PC2 charts are two-dimensional projections of a five-PC clustering solution; projected separation does not show the full model geometry.

## Key findings and business implications

### 1. The largest segment provides a clear online targeting hypothesis

Fashion and technology shoppers account for **744 customers (37.2%)**. They have median online spending of **SGD 900 per month** and **13 online visits**, compared with **2 store visits**. This is the largest segment and has the highest median online spending and online visits among the four groups.

For the simulated marketing brief, these observations justify testing category recommendations or relevant bundles on the retailer's online platform. They do not establish conversion rates or use of particular social platforms. A proposed first pilot would evaluate category-purchase conversion and repeat online purchase, with margin as a guardrail once cost data is available.

### 2. Category needs and purchasing capacity support separate offer hypotheses

Family-focused shoppers have the highest median grocery spending, **SGD 1,000 per month**, and a median of **1 child**. High-income omnichannel shoppers instead have median income of **SGD 15,550**, dining spending of **SGD 900**, and **6 store / 10 online visits** per month.

These profiles motivate different hypotheses: grocery replenishment or bundles for the family-focused pattern, and coordinated cross-channel benefits or service options for the high-income omnichannel pattern. The extra K=4 distinction has a business use only if it leads to different offers and improves outcomes in testing. Higher income and spending do not establish higher profitability.

### 3. Low engagement identifies a group for a repeat-purchase test

Low-engagement shoppers comprise **414 customers (20.7%)**, with median **3 store visits, 3 online visits** and monthly income of **SGD 3,550**. Median fashion, electronics and dining spending are lower than in each of the other three groups.

This supports testing a simple, relevant repeat-visit offer through an available consented retailer channel. Their low activity is an observed cross-sectional pattern. Longitudinal purchases are needed to estimate churn, and cost data is needed to evaluate discount profitability.

### 4. A useful working segmentation requires a model-selection tradeoff

K=3 has a higher silhouette (**0.336**) than K=4 (**0.326**), as well as better CH and DB scores. K=4 is retained because the grocery/family and high-income omnichannel profiles suggest distinct decisions. The recommendation combines numerical evidence with interpretability; it does not claim K=4 is the statistical optimum or that four-group targeting has already improved sales.

## Recommended pilots and measurement plan

| Segment | Evidence-led hypothesis | Proposed action and channel | Proposed primary metric |
| --- | --- | --- | --- |
| Low-engagement shoppers | A relevant prompt could encourage another purchase | Simple repeat-visit offer via an available consented retailer communication channel and in-store checkout | Repeat purchase rate |
| Family-focused shoppers | Grocery-oriented offers may be more relevant to this purchasing pattern | Grocery bundles or replenishment reminders through stores and consented retailer messaging | Grocery repeat purchase rate |
| Fashion and technology shoppers | Recommendations could match observed online category interest | Fashion/electronics recommendations or bundles on the retailer's online platform | Category-purchase conversion |
| High-income omnichannel shoppers | Coordinated benefits could support activity across channels | Cross-channel membership benefits or service options via the retailer's stores and online platform | Cross-channel repeat purchase rate |

**Proposed pilot priority:** start with online category recommendations if exposure tracking and random assignment are feasible. The largest segment has a clear channel signal and a measurable on-platform intervention. This is a proposed starting point based on segment size and behavior, not a forecast of the highest ROI. Operational effort, consent, costs and sample-size requirements could change the priority.

Before launch, define eligible customers, the primary metric, randomization unit, observation window and required sample size. Compare assigned treatment and control customers over the same period, including customers who do not respond. Report the difference with uncertainty rather than counting only offer redemptions. Track discounts, fulfillment/service costs and returns so that increased purchases can be assessed alongside incremental margin.

Customer-level transactions are needed for repeat purchase and retention; recommendation exposures and purchases are needed for conversion; cost and return records are needed for margin. None of these outcomes is measured by the current customer-level summary dataset. No achieved campaign results, baseline rates or numerical targets are claimed. If family and high-income groups receive different offers, a later test can compare the K=4 targeting strategy with the simpler K=3 strategy.

### Proposed KPI definitions

The following is a specification for future campaign measurement. These metrics are not computed from the current snapshot. Define eligibility and a common observation window before assignment, and count distinct customers consistently in each test arm.

| Proposed KPI | Numerator | Denominator | Additional data required |
| --- | --- | --- | --- |
| Repeat purchase rate | Eligible prior purchasers assigned to an arm who make at least one qualifying purchase during the test window | All eligible prior purchasers assigned to that arm | Baseline and test-period customer-level transactions, purchase dates and assignment records |
| Grocery repeat purchase rate | Eligible prior grocery purchasers assigned to an arm who make a qualifying grocery purchase during the test window | All eligible prior grocery purchasers assigned to that arm | Category-coded transactions, baseline grocery purchases and assignment records |
| Category-purchase conversion | Eligible customers assigned to an arm who buy the promoted fashion/electronics category during the test window | All eligible customers assigned to that arm | Category-coded purchases, assignment records and recommendation-exposure logs for diagnostic checks |
| Cross-channel repeat purchase rate | Eligible prior purchasers assigned to an arm who make qualifying purchases in both physical and online channels during the test window | All eligible prior purchasers assigned to that arm | Linked customer identities, channel-coded purchases, dates and assignment records |

For the proposed online pilot, category-purchase conversion is the primary metric. Incremental margin per assigned customer is a proposed guardrail: compare treatment/control average contribution margin after discounts, returns and relevant campaign or service costs. Recommendation clicks and exposure can diagnose delivery, while the primary comparison retains all assigned customers rather than selecting only customers who saw or clicked the offer.

### Proposed experimental decision

Within the eligible fashion/technology segment, randomly assign customers to the category recommendation or the current online experience. Keep assignment at customer level so repeat visits use the same treatment. Before launch, set the observation window, minimum business-relevant effect and sample size using baseline outcome rates; the current data cannot supply these rates or a justified numerical target.

Estimate the treatment/control difference in the primary metric with uncertainty, and review the margin guardrail and exposure-delivery checks. A rollout decision would require evidence of a commercially worthwhile effect, acceptable cost and reliable data. An inconclusive result would motivate more evidence or a revised hypothesis. This is an experiment design proposal; no A/B test, significance test or causal effect estimate has been executed in this project.

### Future regional application

The present data uses SGD and contains no country or market identifier. Applying this reporting approach across markets would first require aligned customer identities, observation windows, category definitions, currency treatment and KPI denominators. Segment behavior would need validation within each market before reusing names or campaign rules. No cross-country marketing comparison is supported by the supplied snapshot.

## Limitations and next steps

This individual academic project uses course-provided cross-sectional data, with no verified real-company engagement, measured campaign outcomes or deployed system. The sampling frame and time coverage are undocumented. Mean/median imputation can reduce variation, and unexplained extreme income/store-visit values can affect the geometry despite logs. Spending overlap prevents a defensible total-revenue measure. Demographic labels and loyalty scores require source definitions; loyalty is not an independent validation target because it was constructed from model-related fields.

The segmentation depends on feature choice, StandardScaler, five retained PCs and distance-based algorithms. K=4 trades stronger K=3 internal metrics for potentially useful purchasing distinctions; internal scores do not establish natural customer types or actionable commercial value. The algorithms do not model noise or guarantee robust treatment of unusual customers. Stability uses transformed coordinates estimated from the full sample, and different protocols prevent a controlled cross-method stability comparison.

Future work could resolve extremes with the source owner, assess alternative treatments with approval, evaluate matched stability/silhouette protocols, re-fit preprocessing within resamples, and validate time persistence and targeting outcomes using authorized longitudinal transactions. Profitability, customer lifetime value, campaign ROI and causal uplift need additional data and suitable analysis. None is claimed here. The included CSV supports reproduction of this analysis.

## Analysis access

The [complete Notebook](../notebooks/customer_segmentation.ipynb) exposes all substantive stages. [Aggregate tables](tables/) include unrounded metrics, profiles, proportions and membership comparisons; [figures](figures/) are regenerated by the pipeline. Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib and seaborn support the analysis.
