# Predictive Modelling for Heart Disease Risk
## Overview
This project demonstrates the development and evaluation of a machine learning model, focusing on ensemble classification techniques and application deployment. The objective was to build a Random Forest Classifier using the UCI Heart Disease dataset, resolve pipeline serialisation issues across development environments, evaluate traditional linear and distance-based baselines, and deploy the optimal predictive architecture within a functional Flask web application.

## Objectives
*   **Data Pre-processing:** Implemented continuous feature scaling (Standardisation), missing value imputation (median and mode variants), and categorical feature transformation (one-hot encoding).
*   **Pipeline Re-engineering:** Resolved cross-environment serialisation bugs by designing a runtime data processing workflow within the application layer.
*   **Hyperparameter Tuning:** Conducted a 5-fold cross-validated grid search to optimise algorithmic tree splitting and estimator boundaries.
*   **Web Deployment:** Built a fully operational Flask web prototype featuring client-side form parameter checking and localised feature-importance explanation layers.

## Technologies Used
*   **Language:** Python
*   **Web Services:** Flask, HTML, CSS
*   **Machine Learning:** Scikit-Learn
*   **Data Pipelines:** Pandas, NumPy
*   **Data Visualisation:** Matplotlib, Seaborn
*   **Version Control:** GitHub

## Experimental Framework
The machine learning pipeline was designed using a Jupyter Notebook framework and evaluates the following model architectures:
*   **Logistic Regression:** Implemented as an interpretable baseline to measure fundamental coefficient impact across linear health indicators.
*   **K-Nearest Neighbours (KNN):** Evaluated as a distance-based instance classifier to establish baseline clustering trends relative to multi-dimensional patient profiles.
*   **Random Forest Classifier:** Developed as the primary ensemble model to capture non-linear biological parameter correlations while mitigating data overfitting.

## Design Considerations
*   **Clinical Metric Alignment:** Prioritising model recall (sensitivity) over precision to minimise critical false-negative diagnostic outcomes.
*   **Dynamic Pre-processing Loader:** Overcoming environment-specific pickle serialisation errors by rebuilding the data transformation pipeline inside the Flask application script.
*   **Input Validation Boundary:** Integrating educational disclaimers alongside automated HTML boundaries to prevent unrealistic clinical inputs.
*   **User-Centred Transparency:** Mapping model features into an interactive, localised explanation panel to clearly illustrate the primary risk drivers behind predictions.

## Key Performance Results
*   **Optimal Architecture Parameters:** Default classifier settings outperformed deep structural parameters, delivering a **Test Accuracy of 86.96%** combined with a clinical **Recall score of 93.14%**.
*   **Baseline Comparative Benchmarks:** Logistic Regression (84.24% Accuracy / 86.12% F1) | K-Nearest Neighbours (85.33% Accuracy / 87.08% F1) | Random Forest (86.96% Accuracy / 88.79% F1).
*   **Primary Diagnostic Indicators:** Identification of serum cholesterol ranges, maximum heart rate thresholds, patient age, and exercise ST depression depths as the top risk features.

## Key Learnings
*   **Clinical Evaluation Strategy:** Understood the practical importance of optimising metrics based on clinical outcomes, learning why high recall is vital when a false negative carries severe consequences.
*   **Production Deployment Flow:** Gained practical experience translating a static model into a functional web service, resolving object serialisation errors by coding a dynamic data pipeline.
*   **Explainable AI Frameworks:** Practiced transforming multi-dimensional model weights into transparent, plain-text explanations suitable for standard web users.
*   **Software System Lifecycle:** Developed an end-to-end perspective on software systems, managing a complete pipeline project from data analysis through to functional application deployment.

## Future Improvements
*   **Advanced Model Explainability:** Embed SHAP (Shapley Additive exPlanations) or LIME components to output distinct feature calculations for individual clinic entries.
*   **Alternative Boosting Benchmarks:** Port the processing structures over to gradient boosting alternatives like XGBoost or LightGBM to evaluate potential classification gains.
*   **Continuous Evaluation Scaling:** Transition the deployment layer to support broader health history parameters to strengthen out-of-sample data generalisability.

## Credits
This project was developed as a comprehensive application of medical data analytics and ensemble learning concepts for predictive healthcare classification.

*   **Developer:** Joanne Uwera
*   **Dataset Reference:** UCI Machine Learning Repository (Anonymised Heart Disease Dataset)
