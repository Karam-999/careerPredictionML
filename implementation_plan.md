# Career Path Prediction — Target Variable Grouping

## Background

The "Career Path Prediction and Guidance System" uses the `Suggested Job Role` column as its classification target, which currently contains **12 distinct job roles** with approximately equal class distribution (~550-630 samples each, 6902 total). Despite trying multiple models (Random Forest, XGBoost, LightGBM, CatBoost), feature engineering, and class balancing, accuracy remains very low. Mutual Information analysis confirms that input features have almost no predictive power over these fine-grained roles.

Per the ML mentor's guidance, we will **group similar roles into broader categories** to reduce classification complexity.

---

## Step 1: Current Job Roles (Inspected from Dataset)

| # | Suggested Job Role | Count |
|---|---|---|
| 1 | Applications Developer | 551 |
| 2 | CRM Technical Developer | 567 |
| 3 | Database Developer | 581 |
| 4 | Mobile Applications Developer | 538 |
| 5 | Network Security Engineer | 630 |
| 6 | Software Developer | 587 |
| 7 | Software Engineer | 590 |
| 8 | Software Quality Assurance (QA) / Testing | 571 |
| 9 | Systems Security Administrator | 562 |
| 10 | Technical Support | 565 |
| 11 | UX Designer | 589 |
| 12 | Web Developer | 570 |

> [!NOTE]
> Classes are relatively balanced (ratio ~1.17). The poor accuracy is **not** caused by class imbalance — it's caused by the features being too generic to distinguish 12 fine-grained roles.

---

## Step 2: Proposed Grouping (12 → 4 Categories)

> [!IMPORTANT]
> **Please review this mapping carefully and approve/modify before I proceed with implementation.**

### Category 1: 🖥️ **Software Development**
| Original Role | Rationale |
|---|---|
| Applications Developer | Builds application software — core development |
| Software Developer | General software development — overlaps heavily with Applications Developer |
| Software Engineer | Engineering-focused software creation — same domain as above |
| CRM Technical Developer | Develops CRM platform solutions — still a software development role |
| Mobile Applications Developer | Develops mobile apps — a specialization of software development |
| Web Developer | Develops web applications — a specialization of software development |

**Why group these?** All six roles involve writing code to build software products. The features in our dataset (coding skills, certifications, hackathons, interested subjects) are too generic to differentiate between a "Web Developer" and a "Mobile Applications Developer" or a "Software Developer" and a "Software Engineer." These roles differ mainly in platform/technology stack, which our feature set doesn't capture with enough specificity.

**Sample count:** 551 + 567 + 538 + 587 + 590 + 570 = **3,403**

---

### Category 2: 🛡️ **Cybersecurity & Infrastructure**
| Original Role | Rationale |
|---|---|
| Network Security Engineer | Designs and maintains secure network infrastructure |
| Systems Security Administrator | Administers and secures IT systems |

**Why group these?** Both roles focus on protecting and securing IT systems/networks. They require similar security-oriented skills and mindsets. The dataset features like "interested career area" (security), "interested subjects" (hacking, networks), and certifications (information security) relate equally to both roles.

**Sample count:** 630 + 562 = **1,192**

---

### Category 3: 🧪 **Quality Assurance & Support**
| Original Role | Rationale |
|---|---|
| Software Quality Assurance (QA) / Testing | Tests software for quality and correctness |
| Technical Support | Provides technical troubleshooting and user support |

**Why group these?** Both roles are non-development, service-oriented tech roles focused on ensuring software works correctly for end users. QA ensures quality before release; Technical Support ensures quality after release. They share a service/support mindset distinct from building or securing software.

**Sample count:** 571 + 565 = **1,136**

---

### Category 4: 🗄️ **Data & Database Engineering**
| Original Role | Rationale |
|---|---|
| Database Developer | Designs and develops database systems |

**Why keep this separate?** Database Developer is distinctly data-focused (SQL, NoSQL, data modeling) compared to the general software development group. However, with only 1 role, this category has fewer samples.

**Sample count:** **581**

---

### Category 5: 🎨 **Design & User Experience**
| Original Role | Rationale |
|---|---|
| UX Designer | Designs user interfaces and user experiences |

**Why keep this separate?** UX Design is fundamentally different from all other tech roles — it's design-focused rather than code-focused. It requires creative/artistic skills and user psychology understanding that are distinct from engineering roles.

**Sample count:** **589**

---

## User Review Required

> [!WARNING]
> The 5-category split above results in an **imbalanced** distribution:
> - Software Development: 3,403 (49.3%)
> - Cybersecurity & Infrastructure: 1,192 (17.3%)
> - Quality Assurance & Support: 1,136 (16.4%)
> - Data & Database Engineering: 581 (8.4%)
> - Design & User Experience: 589 (8.5%)
>
> If you prefer a **more balanced 4-category** split, I can merge "Data & Database Engineering" into "Software Development" (since DB development is code-heavy) and merge "Design & User Experience" into "Quality Assurance & Support" (since both are non-core-development roles). This gives:
>
> | Category | Roles | Count |
> |---|---|---|
> | Software Development | Apps Dev, CRM Dev, DB Dev, Mobile Dev, SW Dev, SW Eng, Web Dev | 3,984 (57.7%) |
> | Cybersecurity & Infrastructure | Network Security Eng, Systems Security Admin | 1,192 (17.3%) |
> | QA, Support & Design | QA/Testing, Technical Support, UX Designer | 1,725 (25.0%) |

> [!IMPORTANT]
> **Which grouping do you prefer?**
> - **Option A**: 5 categories (keeps Database & UX as separate categories)
> - **Option B**: 3 categories (merges DB into Dev, UX into QA/Support)
> - **Option C**: 4 categories (e.g., merge just DB into Dev, keep UX separate)
> - **Your own custom grouping**

---

## Proposed Changes (After Approval)

### Training Pipeline

#### [NEW] [train_grouped.py](file:///d:/CareerPrediction/train_grouped.py)
- Complete training script that:
  1. Loads dataset
  2. Creates `Career_Category` column using approved mapping dictionary
  3. Shows class distribution before/after grouping
  4. Preserves all existing preprocessing (OrdinalEncoder, SimpleImputer, SMOTENC)
  5. Trains RandomForestClassifier with same hyperparameters
  6. Reports: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
  7. Compares results with original 12-class model
  8. Saves new model to `models/career_model_grouped.pkl`

---

### Streamlit App

#### [MODIFY] [app.py](file:///d:/CareerPrediction/app.py)
- Update model loading to use grouped model
- Update recommendations dictionary for new category names
- Update prediction display

---

## Verification Plan

### Automated Tests
- Run `train_grouped.py` and compare metrics (accuracy, precision, recall, F1) against the original 12-class model
- Verify confusion matrix shows improved classification
- Confirm SMOTENC balancing works correctly with reduced classes

### Manual Verification
- Compare classification reports side by side
- Verify Streamlit app loads and predicts with the new grouped model
