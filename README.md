# **Classification of Daily Expenses**

**Author:** Maria Kasznar ([https://www.linkedin.com/in/biakasznar/](https://www.linkedin.com/in/biakasznar/))

**Date:** June 2025

## **Project Overview**

This machine learning project implements a multiclass classification pipeline designed to automatically categorize personal expense data. The dataset covers a year-long stay in Australia (July 2022 \- July 2023), with expenses from Brisbane and Perth serving as labeled training data. The key focus is to accurately classify unlabeled expenses from a third, unseen city (Jabiru).

## **Detailed Explanation in Jupyter Notebook**

For a comprehensive understanding of the project's methodology, including detailed explanations of data cleaning, exploratory data analysis (EDA), feature engineering, the full list of tested models, hyperparameter tuning with GridSearchCV, and model evaluation, please refer to the Classification of Daily Expenses.ipynb Jupyter Notebook.

## **Technologies Used**

These are the main libraries used; a full list of all dependencies can be found in the environment.yml file.

|  | Version |
| :---- | :---- |
| Python | 3.12.7 |
| pandas | 2.2.3 |
| numpy | 2.2.5 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |
| openpyxl | 3.1.5 |

## **How to Run**

1. **Clone the Repository:**  
   git clone \<repository\_url\>  
   cd \<repository\_name\>

2. Install Dependencies:  
   You can set up the environment using the provided environment.yml file with Conda:  
   > conda env create \-f environment.yml  
   > conda activate classification-env \
   
3. **Place Data:** Ensure your DailyExpenses\_PythonProject.xlsx file is in the specified data directory. There are 2 instances that refer to the path, both on the first code cell in the notebook.  
4. **Run Jupyter Notebook:**  
   > jupyter notebook "Classification of Daily Expenses.ipynb"

   Open the notebook in your browser and run all cells sequentially.