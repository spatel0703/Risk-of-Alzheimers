# Risk-of-Alzheimers Project

## Overview

The Risk-of-Alzheimers project, a collaborative effort led by Anthony Shajan, Dulce Osorio, and Shrey Patel, focused on evaluating the risk of Alzheimer's disease among patients using a Kaggle dataset. This project was conducted entirely in Python, leveraging data science concepts to construct a comprehensive data report on Alzheimer's and associated properties.

## Project Components

1. **Data Exploration and Quality Report:**
   - The project begins with loading and preprocessing data from the Kaggle dataset ('oasis_longitudinal.csv'). The [Data Quality Report](#data-quality-report) section provides insights into the dataset's statistics, missing values, and quality reports for both categorical and continuous features.

2. **Data Visualization:**
   - Visualizations are employed to gain a deeper understanding of the dataset. The [Data Visualization](#data-visualization) section includes histograms, boxplots, and scatter plots to represent the distribution of features and relationships between variables.

3. **Normalization of Original Dataset:**
   - The [Normalization](#normalization-of-original-dataset) process involves handling missing values and performing Min-Max scaling on continuous features.

4. **Imputation and Outliers:**
   - Missing values are imputed, and outliers are identified and removed in the [Imputation and Outliers](#imputation-and-outliers) section.

5. **Accuracy Matrix and Random Forest Classifier:**
   - A Random Forest Classifier is trained and evaluated to predict Alzheimer's risk. The [Accuracy Matrix and Random Forest Classifier](#accuracy-matrix-and-random-forest-classifier) section includes a ROC curve, accuracy metrics, and a classification report.

6. **User Input and Risk Prediction:**
   - Users can input data to predict Alzheimer's risk using the [User Input and Risk Prediction](#user-input-and-risk-prediction) section.

7. **Adding Risk Column to Imputed and Outlier Datasets:**
   - A 'Risk of Alzheimer' column is added to both imputed and outlier datasets, indicating the predicted risk level. This is detailed in [Adding Risk Column to Imputed and Outlier Datasets](#adding-risk-column-to-imputed-and-outlier-datasets).

8. **Data Visualization for Imputed and Outlier Datasets:**
   - The [Data Visualization for Imputed and Outlier Datasets](#data-visualization-for-imputed-and-outlier-datasets) section includes additional visualizations for both datasets, highlighting the impact of imputation and outlier removal.

9. **Correlation Heatmap:**
   - Correlation heatmaps are generated for both imputed and outlier datasets in the [Correlation Heatmap](#correlation-heatmap) section.

## Instructions for Running the Code

1. Clone the repository: `git clone https://github.com/your/repo.git`
2. Navigate to the project directory: `cd Risk-of-Alzheimers`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebook or Python script containing the code.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Results and Insights

- The project provides valuable insights into Alzheimer's risk prediction, with visualizations highlighting the impact of imputation and outlier removal on the dataset.
- The Random Forest Classifier demonstrates promising results in predicting Alzheimer's risk.

## Future Work

- Consider exploring additional feature engineering techniques.
- Investigate the performance of other classification algorithms.
- Enhance user interactivity for input validation and handling.

## References
- Kaggle Dataset: [OASIS Longitudinal Dataset](https://www.kaggle.com/jboysen/mri-and-alzheimers)

