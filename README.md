# Wine Quality Prediction

A beginner machine learning project that estimates a wine quality score from physicochemical measurements such as acidity, sugar, density, and alcohol content.

## Internship ID

Add your own internship ID here before submitting this repository:

`YOUR_INTERN_ID`

## How it works

- Downloads the Wine Quality dataset from the UCI Machine Learning Repository when the script runs.
- Splits the measurements into training and test sets.
- Trains a random forest regression model to estimate the quality score.
- Reports mean absolute error, root mean squared error, and R-squared, then prints one example prediction.

## Run it

Requires Python 3.10 or newer and an internet connection for the first dataset download.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

## Dataset and attribution

The project uses the UCI Machine Learning Repository's Wine Quality dataset. It contains physicochemical measurements and sensory quality scores. The dataset is licensed under Creative Commons Attribution 4.0; please keep this attribution with the project.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality). DOI: [10.24432/C56S3T](https://doi.org/10.24432/C56S3T).

## Tools

Python, scikit-learn, ucimlrepo, and a random forest regressor.
