# HarvestIQ Ghana
### An Agricultural Decision Support Tool for Ghanaian Smallholder Farmers

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Context](https://img.shields.io/badge/Context-Ghana%20Agriculture-gold)]()
[![Made in](https://img.shields.io/badge/Made%20in-Tamale%2C%20Ghana-red)]()

---

## The Problem

Every planting season, thousands of smallholder farmers across Ghana make critical decisions with almost no data. Which crop should I grow? Will my soil support it? How much should I expect to harvest? When should I tell buyers my produce will be ready?

These decisions are made by instinct, tradition, and hope.

A wrong crop choice, a missed harvest window, or an overestimated yield can mean months of lost income for a family that had no margin to lose. In a country where agriculture employs over 40% of the workforce and the majority of farmers operate below one hectare of land, the cost of a bad decision is not just financial — it is deeply personal.

**HarvestIQ Ghana** is a machine learning pipeline built to answer the three questions every Ghanaian farmer, agricultural buyer, and extension officer needs answered before a single seed goes into the ground:

> **What should I plant? How much will I get? When will it be ready?**

---

## What This Project Does

HarvestIQ Ghana trains two separate regression models on farm-level data from 10 Ghanaian regions:

| Model | Target | Question Answered |
|---|---|---|
| **Yield Predictor** | `Yield_kg_per_ha` | How much will I harvest per hectare? |
| **Harvest Timeline Predictor** | `days_to_harvest` | How many days from planting to harvest? |

Both models are trained on soil nutrient composition, rainfall, crop type, farm region, and planting season data. The pipeline covers the full ML workflow — raw data → cleaning → feature engineering → EDA → modelling → evaluation.

---

## Who This Is For

This tool is designed with three users in mind, and all three benefit from the same pipeline:

**The Farmer** — chooses a crop, inputs their soil readings and region, and gets an expected yield and harvest date. Enough information to plan their season, negotiate with buyers, and manage their household income.

**The Buyer / Aggregator** — uses farmer-level yield predictions to forecast regional supply and plan procurement ahead of harvest season. No more guessing how much produce will come out of Northern Ghana in November.

**The NGO / Extension Officer** — uses the feature importance outputs from the model to identify which soil inputs (nitrogen, phosphorus, potassium, pH) most strongly drive yield in their region. This becomes an evidence-based input recommendation — not generic advice, but data-backed guidance for the exact crop and region in question.

One model. Three stakeholders. Real decisions.

---

## Dataset

- **Records:** 2,060 farm-level observations (post-cleaning: ~2,000)
- **Regions covered:** 10 Ghanaian regions — Northern, Upper West, Upper East, Brong Ahafo, Ashanti, Volta, Eastern, Central, Western, Greater Accra
- **Crops:** Cassava, Yam, Maize, Rice, Soybean, Groundnut, Cocoa, Plantain
- **Features:**

| Column | Type | Description |
|---|---|---|
| `Farm_ID` | Identifier | Unique farm reference (dropped before modelling) |
| `Crop_Type` | Categorical | Type of crop planted |
| `Farm_Location` | Categorical | Ghanaian region |
| `Nitrogen_Percent` | Numeric | Soil nitrogen content (%) |
| `Phosphorus_ppm` | Numeric | Soil phosphorus (parts per million) |
| `Potassium_ppm` | Numeric | Soil potassium (parts per million) |
| `Soil_pH` | Numeric | Soil acidity/alkalinity |
| `Rainfall_mm` | Numeric | Seasonal rainfall (mm) — log-transformed |
| `Planting_Date` | Date | Date of planting (used for feature extraction) |
| `Harvest_Date` | Date | Date of harvest (used to compute target) |
| `Yield_kg_per_ha` | **Target 1** | Crop yield in kilograms per hectare |

**Engineered features:**

| Feature | Description |
|---|---|
| `days_to_harvest` | **Target 2** — (Harvest_Date − Planting_Date) in days |
| `Planting_Month` | Calendar month of planting (1–12) |
| `Planting_Season` | Ghana agricultural season: Major (Mar–Aug) or Minor (Sep–Feb) |

> **Note on dataset origin:** This is a structured synthetic dataset built to reflect realistic Ghanaian farm conditions. It is not sourced from a public repository. Real-world deployment would require integration with MoFA Ghana district-level records or verified sensor data.

---

## Pipeline Overview

```
Raw Data (unclean_data.csv)
        │
        ▼
 Data Inspection & Quality Audit
  ├── Shape, dtypes, descriptive stats
  ├── Duplicate detection and removal
  └── Missing value quantification (~5% across key columns)
        │
        ▼
 Feature Engineering
  ├── days_to_harvest  ← Harvest_Date − Planting_Date
  ├── Planting_Month   ← calendar month (1–12)
  ├── Planting_Season  ← Major / Minor based on Ghana rainfall calendar
  └── Data quality filter: remove impossible days_to_harvest (< 0 or > 400)
        │
        ▼
 Data Cleaning
  ├── Mean imputation for numeric nulls
  ├── Mode imputation for categorical nulls
  └── Region name standardisation (17 variants → 10 clean regions)
        │
        ▼
 Exploratory Data Analysis
  ├── Distribution histograms + skewness/kurtosis
  ├── Log transform on Rainfall_mm
  ├── IQR outlier detection (noted, not removed — agricultural extremes are real)
  ├── Categorical distribution plots (crop type, region)
  └── Correlation heatmap (numeric features vs both targets)
        │
        ▼
 Encoding & Preprocessing
  ├── One-hot encode: Crop_Type, Farm_Location, Planting_Season
  ├── Keep Planting_Month as numeric ordinal
  ├── Drop Farm_ID and original date columns
  └── StandardScaler for Lasso feature selection step
        │
        ▼
 Lasso Feature Selection
  └── Identifies features with non-zero coefficients → final feature list
        │
        ▼
 Modelling (two separate pipelines)
  ├── Model 1: Yield_kg_per_ha
  │    ├── Baseline: Linear Regression
  │    └── Main: Random Forest (n=100, random_state=42)
  └── Model 2: days_to_harvest
       ├── Baseline: Linear Regression
       └── Main: Random Forest (n=100, random_state=42)
        │
        ▼
 Evaluation
  ├── MAE, RMSE, R² for all four models
  ├── Actual vs Predicted scatter plots
  └── Feature importance bar charts
```

---

## Key Technical Decisions

**Why two separate models instead of one multi-output model?**
Yield and harvest duration are driven by different signals. Yield is dominated by soil chemistry and rainfall. Harvest duration is largely a function of crop biology — what you planted and when. Combining them into a single model would force one architecture to serve two fundamentally different prediction tasks and hurt both. Separate models, evaluated separately, is the cleaner, more interpretable approach.

**Why Random Forest over a simpler model?**
The relationship between soil nutrients, rainfall, crop type, and yield is non-linear. Nitrogen at 1.5% means something different under high rainfall than under low rainfall — a linear model cannot capture that interaction. Random Forest handles it naturally and also gives us feature importance, which is essential for the extension officer use case.

**Why keep Linear Regression as a baseline?**
To answer the question: does the complexity of Random Forest actually help? If both models perform similarly, the simpler one wins. Having a baseline prevents us from defaulting to complexity for its own sake.

**Why log-transform Rainfall_mm?**
Rainfall in Ghana is right-skewed — most farms receive moderate rainfall, with a few receiving very high amounts. Without transformation, those extreme values have outsized influence during training. Log-transforming compresses the scale and gives the model a more balanced view of rainfall's effect.

**Why Lasso for feature selection?**
After one-hot encoding, the feature space expands significantly. Lasso penalises irrelevant features by shrinking their coefficients to zero, leaving only the features with genuine predictive signal. This reduces model complexity and makes predictions more interpretable.

---

## Results


| Model | Target | Algorithm | MAE | R² |
|---|---|---|---|---|
| Baseline | Yield (kg/ha) | Linear Regression | 1,615.9 kg/ha | 0.7659 |
| Main | Yield (kg/ha) | Random Forest | 1,720.0 kg/ha | 0.7277 |
| Baseline | Days to Harvest | Linear Regression | 25 | 0.7798 |
| Main | Days to Harvest | Random Forest | 26 | 0.7591 |


---

## Honest Limitations

Good data science is transparent about what a model cannot do. This section exists because I believe a model's limitations are as important as its results.

**Missing agronomic drivers.** Crop yield depends on fertiliser timing and rate, pest and disease pressure, irrigation vs. rain-fed status, and seed variety. None of these are in the dataset. Their absence sets a hard ceiling on model performance that no algorithm can overcome — the information simply isn't there.

**Single global model across eight crops.** Cassava, Maize, and Cocoa have completely different yield scales, growth cycles, and soil requirements. A single model must generalise across all eight, which limits per-crop accuracy. Separate per-crop models would outperform this approach given sufficient data per crop.

**Dataset size.** ~2,000 records across 8 crops and 10 regions gives roughly 250 rows per crop — too thin for Random Forest to learn stable, generalisable patterns. This is likely why the baseline Linear Regression remains competitive. More data per crop is the single highest-leverage improvement available.

**Mean imputation for missing targets.** About 5% of `Yield_kg_per_ha` values were missing and filled with column means. Imputed target values carry no real farm-level signal and add noise to training. Model-based or KNN imputation would be a cleaner approach.

These are not excuses. They are the next set of problems to solve.

---

## Project Structure

```
harvest-predictor-ghana/
│
├── HarvestIQ_Ghana.ipynb      ← Full pipeline notebook (start here)
├── unclean_data.csv           ← Raw dataset
├── README.md                  ← This file
└── requirements.txt           ← Python dependencies
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/harvest-predictor-ghana.git
cd harvest-predictor-ghana

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook HarvestIQ_Ghana.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

---

## What's Next

- [ ] Collect real farm-level data from MoFA Ghana district offices
- [ ] Add fertiliser application, irrigation status, and seed variety as features
- [ ] Build per-crop models once sufficient data per crop is available
- [ ] Wrap both models in a simple Streamlit interface for field use
- [ ] Explore SHAP values for per-prediction explainability (farmer-friendly "why")
- [ ] Extend to post-harvest quality grading (HarvestIQ Post-Harvest)

---

## Contributors

HarvestIQ Ghana was developed collaboratively by a team of six researchers and ML practitioners committed to building data-driven tools for African agriculture.

| Name | Role |
|---|---|
| **Abdul Wahab Osman** | Project Lead & ML Practitioner — Tamale, Ghana |
| **Issifu Salma** | Researcher & ML Contributor |
| **Georgette Kweiki Narh** | Researcher & ML Contributor |
| **Abdulai Mohammed** | Researcher & ML Contributor |
| **Ewurabena Esther Appiah** | Researcher & ML Contributor |
| **Solomon Yenyenle** | Researcher & ML Contributor |

We build machine learning solutions grounded in African contexts because the problems here are real, the data is scarce, and the people who need these tools the most are rarely the ones the global AI community builds for.

This project is part of a broader portfolio of Ghana-focused ML work including DumsorData (power outage prediction), HarvestIQ Post-Harvest (crop quality grading), and FoodQwik (AI-powered food ordering).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/abdulwahabosman/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/Abdul-Pandev)

---

> *"The best time to give a farmer data was before planting. The second best time is now."*

---



*Built in Ghana, by a team that believes African farmers deserve better tools.*
## "EDA COMPLETED"