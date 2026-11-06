# Steam Store-Page Features: An Exploratory Analysis

This project investigates associations between Steam store-page features and review volume in **1,984 indie games**. It combines data collection, feature engineering, OLS regression and machine-learning comparisons.

The original motivation was pre-launch prediction. The scraped snapshots were not verified as pre-launch observations. These results describe a retrospective exploration, rather than validated launch forecasts or causal effects of page design.

## Dataset and target

- **1,984 unique AppIDs**, including **675 zero-review games**. The notebook retains these rows.
- Numeric inputs: price, GIF/image/bullet/word counts and demo availability.
- Additional experiments use Steam tags and, in selected trials, game-title text.
- `Review_Count` defines the outcome. `Positive_Ratio` and `AppID` are excluded from predictors.
- Collection scripts use `requests`, Beautiful Soup and Steam endpoints. IDs come from top-rated, new-release and relevance search pages; this is not a representative random sample of all Steam games.
- The current scraper writes `steam_games_dataset_clean.csv` and applies a 20-review threshold. That differs from the notebook dataset `steam_indie_games_dataset.csv`, which includes zero-review rows. The scraper is not a byte-for-byte reproduction of that historical dataset.

The exact categories are **0-49**, **50-999**, and **1,000+** reviews. These are heuristic review-volume bins, not validated definitions of commercial failure, sustainability or success. The dataset contains no sales or profit labels.

The proposed visual-density metric is:

$$
\text{Visual Density} = \frac{2 \times \text{GIF Count} + \text{Image Count}}{\text{Word Count} + 1}
$$

The GIF weight of two is a heuristic. It does not measure visual quality, gameplay quality or a causal marketing effect.

## Original analysis: corrected interpretation

OLS regresses `log1p(Review_Count)` on visual density. Reproduction gives **p = 0.135** and **R² = 0.0011**. This model does not provide evidence of a nonzero linear slope at the conventional 5% level. It does not prove the absence of every possible relationship.

The original Random Forest uses price, visual density, structure ratio and demo availability. On its historical 397-row test split it obtains **52.90% accuracy** and **0.471 macro-F1**. Middle-category recall is only **18.95%**.

Reproduced impurity-based importances are approximately **0.339 for price** and **0.331 for visual density**. Visual density is not the highest-ranked feature in this run. Feature importance does not establish statistical significance, causality, nonlinearity or an optimal design. OLS and the classifier use different outcomes and inputs; their outputs are not a statistical contradiction.

Earlier claims that bad design guarantees failure, that these results prove nonlinearity, or that asset flips explain errors were unsupported and have been removed. Asset quality, gameplay and Steam algorithm behavior were not measured.

## Reproducible model comparison

`evaluate_models.py` evaluates **26 candidates in two development stages**:

1. **15 initial candidates:** majority baseline, regularized Random Forests, Extra Trees, histogram gradient boosting and logistic regression, with numeric features or numeric features plus tags.
2. **11 deeper candidates:** balanced logistic regression, title word/character TF-IDF, cumulative ordinal classification, direct log-review-count regression and boosted models with tags.

Model choice uses **five-fold stratified cross-validation within the 1,587 original training rows**. The original unstratified 80/20 split and seed 42 are preserved. Imputation, scaling and text vocabularies are fitted within each training fold. Numeric-only, accuracy and macro-F1 winners follow their CV scores; the original and stage-one models are reference comparisons.

| Model | Training CV accuracy | Historical test accuracy | Test macro-F1 | Test middle-category recall |
| --- | ---: | ---: | ---: | ---: |
| Majority-class baseline | 40.77% | 40.30% | 0.192 | 0.00% |
| Original Random Forest (4 numeric features) | 50.72% | 52.90% | 0.471 | 18.95% |
| CV-selected numeric-only Random Forest | 54.13% | 50.38% | 0.410 | 5.26% |
| Random Forest + tags (CV accuracy winner) | 62.51% | 59.19% | 0.453 | 1.05% |
| Log-review Extra Trees + tags (CV macro-F1 winner) | 56.96% | 57.43% | 0.543 | 46.32% |
| Logistic regression + tags (stage-one reference) | 58.92% | 58.44% | 0.531 | 23.16% |

**The largest accuracy gain comes from adding tags**, which broadens the question beyond visual composition. The accuracy-selected Random Forest reaches **59.19%**, but correctly identifies only **1 of 95** middle-category games. Its macro-F1 is lower than the original model's, so this is not an unqualified improvement.

The secondary CV macro-F1 winner predicts `log1p(Review_Count)` with Extra Trees, then maps predictions back to the same bins. It obtains **57.43% accuracy**, **0.543 macro-F1**, and **46.32% middle-category recall**. This offers a different tradeoff rather than the highest accuracy.

The CV-selected numeric-only alternative falls to **50.38%** on the historical test split. These experiments do not establish that tuning visual-count features alone improves generalization.

### What happened in the deeper trials?

These are **training-only CV scores**, not test-set rankings:

| Second-stage candidate | Mean CV accuracy | Mean CV macro-F1 |
| --- | ---: | ---: |
| `balanced_logistic_numeric_tags` | 56.71% | 0.542 |
| `logistic_numeric_tags_title_word` | 58.54% | 0.533 |
| `logistic_numeric_tags_title_char_wb` | 58.98% | 0.539 |
| `ordinal_logistic_numeric_tags` | 58.79% | 0.534 |
| `ordinal_hist_boost_numeric` | 51.92% | 0.448 |
| `log_reviews_hist_boost_numeric` | 46.57% | 0.450 |
| `ordinal_hist_boost_numeric_tags` | 59.29% | 0.542 |
| `log_reviews_hist_boost_numeric_tags` | 56.27% | 0.543 |
| `log_reviews_extra_trees_numeric_tags` | 56.96% | 0.549 |
| `hist_boost_numeric_tags_nodes7` | 60.49% | 0.538 |
| `hist_boost_numeric_tags_nodes15` | 59.67% | 0.536 |

Title text and ordinal classification did not beat the first-stage accuracy winner in these trials. Direct count regression improved the macro-F1 tradeoff but did not produce a large accuracy jump. A more complex method does not guarantee a higher useful score.

The full [machine-readable report](reports/model_comparison.json) includes all fold scores and standard deviations, model parameters, class-level test metrics, paired bootstrap intervals, dataset checksum, test AppIDs and package versions.

## Interpretation and remaining limits

- **Historical holdout:** This test split was already examined in the original notebook. It is a development comparison, not fresh external validation. Stage two and the secondary macro-F1 comparison were motivated by earlier findings.
- **Selection uncertainty:** Comparing many candidates and reusing CV folds can make the winning estimate optimistic. Bootstrap intervals only describe differences on this historical sample; they do not account for adaptive selection or sampling bias.
- **Temporal ambiguity:** Page design, tags and demo availability may have changed after launch or after a game gained attention. Excluding reviews from inputs does not resolve this.
- **Exposure and confounding:** Games have different ages, genres, budgets and publisher reach. Total reviews mix those factors with any design association.
- **Scope:** Tags and titles are additional metadata, not measurements of visual layout or description content. No trial proves a causal design effect or reliable pre-launch forecasting.

### What could improve the project further?

The stronger next step is better measurements and validation, rather than a promised accuracy target:

1. Collect timestamped, verified pre-launch pages and measure reviews at a common horizon, such as the first 90 days after release.
2. Capture actual description text and image/GIF content; counts and game titles cannot represent their meaning or quality.
3. Add developer/publisher history and release context using information available at the intended prediction time.
4. Validate across later release cohorts and developer/publisher groups, then use an independent holdout or nested CV for model-selection assessment.

The CSV lacks those observations. Post-release reviews or ratings are not added as predictors to inflate scores. Better data may help, but high accuracy is not guaranteed.

## Run locally

Use Python 3.11 and an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate_models.py
```

The default report is `reports/model_comparison.json`. Use `--data` and `--output` for other paths. The report records its environment; exact results can vary with package versions.

Run `analysis_notebook.ipynb` for the original analysis, corrected interpretations and comparison tables. Its comparison section verifies the dataset checksum against the report and requests a fresh evaluation if the data changes.

The tests cover exact class boundaries through the log transform, exclusion of outcome-derived inputs, duplicate AppID rejection and valid probabilities from the ordinal classifier.

## References

- [scikit-learn: leakage and common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: cross-validation and model selection](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: feature-importance limitations](https://scikit-learn.org/stable/modules/permutation_importance.html)

Author: Ali Umut Altas ([Toricicim](https://github.com/Toricicim)), Statistics undergraduate at Gazi University.
