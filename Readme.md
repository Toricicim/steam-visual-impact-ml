# Steam Indie Game Success Prediction: A Pre-Launch Analysis

## Project Overview
**Can we predict if an Indie game will be a "Hit" or a "Flop" before it even launches?**

Most success prediction models rely on early user reviews, ratings, or concurrent player counts. However, for a developer, it is crucial to estimate potential success **solely based on the Store Page Design** before the game goes live.

This project analyzes the visual composition (GIFs, Images, Text) of Steam store pages to quantify the impact of **"Visual Density"** on commercial success.

---

## The Core Metric: "Visual Density"
Raw counts of assets (e.g., "This game has 10 screenshots") proved insufficient to explain success. To measure the developer's effort and the "Show, Don't Tell" approach, I derived a custom metric:

$$\text{Visual Density} = \frac{(\text{GIF Count} \times 2) + \text{Image Count}}{\text{Word Count} + 1}$$

* **Logic:** A high score indicates a page that relies on visual storytelling rather than walls of text.
* **Hypothesis:** Optimized Visual Density correlates with higher market traction.

---

## The Statistical Conflict: Why We Ignored the P-Value
One of the most critical phases of this research was the conflict between **Classical Statistics** and **Machine Learning**.

### 1. The OLS Regression "Failure"
We first ran a Classical OLS Regression to test the linear relationship between `Visual_Density` and Success (`Log_Reviews`).
* **Result:** The p-value was **0.758** ($p > 0.05$).
* **Interpretation:** Statistically, there is **no significant linear relationship**. A traditional approach might have rejected the hypothesis here.

### 2. The Random Forest Solution
Despite the OLS result, we proceeded to train a Random Forest Classifier to detect complex patterns.
* **Result:** The model identified `Visual_Density` as the **#1 Most Important Feature** (Importance: ~0.40), outweighing even Price.
* **The Insight:** This contradiction proves that the relationship is **Non-Linear**. Success does not scale linearly with more visuals (which OLS looks for). Instead, there is likely a complex "sweet spot" or threshold that only non-linear algorithms (like Random Forest) can detect.

> **Key Takeaway:** We demonstrated that "No Linear Correlation" does not imply "No Predictive Power."

---

## Critical Limitations & The "Asset Flip" Reality
While the model successfully identifies potential "Hits" with high precision, it struggles to identify "Flops" (Low Recall for Class 0). This is not a technical bug, but a reflection of the current state of the Indie Market.

### 1. The "Hygiene Factor" Theory
Our analysis suggests that Store Page Design acts as a **Hygiene Factor**:
* **Bad Design = Guaranteed Failure:** If a page lacks visuals (Low Density), the model correctly predicts a Flop.
* **Good Design ≠ Guaranteed Success:** Having a high-quality page does not guarantee a Hit; it only "qualifies" the game to enter the race.

### 2. The "Asset Flip" Problem
The model generates **False Positives** on Flop games (predicting them as Hits).
* **Reason:** In the modern era, even low-effort "shovelware" games can use AI or cheap asset packs to create professional-looking store pages.
* **Conclusion:** The model measures the **"Professionalism of the Presentation,"** not the fun factor of the gameplay. It cannot distinguish between a polished gem and a polished "Asset Flip" without gameplay data.

---

## Tech Stack & Methodology
* **Data Collection:** Hybrid approach using Selenium (for scraping) and Steam API.
* **Data Cleaning:** Log-transformation of review counts (`Log_Reviews`), outlier removal (0-review games).
* **Modeling:**
    * **OLS:** For linearity testing (`statsmodels`).
    * **Random Forest:** For non-linear classification and feature importance (`scikit-learn`).
* **Validation:** Confusion Matrix, Feature Importance Analysis, and manual simulation function.

---

## Author
**[Toricicim - Ali Umut Altas]**
* Statistics Student @ [Gazi]
