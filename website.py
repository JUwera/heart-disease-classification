from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


def load_and_train_artifacts():
    df = pd.read_csv("heart_disease_uci.csv")
    df = df.drop(columns=["id", "dataset"], errors="ignore")
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df = df.replace("nan", np.nan)

    X_raw = df.drop(columns=["num"])
    y = df["num"]

    categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_raw.select_dtypes(exclude=["object"]).columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_raw[numerical_cols] = num_imputer.fit_transform(X_raw[numerical_cols])
    X_raw[categorical_cols] = cat_imputer.fit_transform(X_raw[categorical_cols])

    X = pd.get_dummies(X_raw, drop_first=True)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    baseline_values = {
        col: (X_raw[col].mode()[0] if col in categorical_cols else float(X_raw[col].median()))
        for col in X_raw.columns
    }

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "importance_df": importance_df,
        "baseline_values": baseline_values,
    }

artifacts = load_and_train_artifacts()


def prepare_input(input_df, artifacts):
    numerical_cols = artifacts["numerical_cols"]
    categorical_cols = artifacts["categorical_cols"]

    prepared = input_df.copy()
    prepared[numerical_cols] = artifacts["num_imputer"].transform(prepared[numerical_cols])
    prepared[categorical_cols] = artifacts["cat_imputer"].transform(prepared[categorical_cols])

    encoded = pd.get_dummies(prepared, drop_first=True)
    encoded = encoded.reindex(columns=artifacts["feature_columns"], fill_value=0)
    return encoded


def get_top_contributors(user_input, artifacts, top_n=3):
    importance_df = artifacts["importance_df"]
    baseline = artifacts["baseline_values"]
    row = user_input.iloc[0]

    readable_names = {
        "chol": "cholesterol",
        "thalch": "maximum heart rate",
        "age": "age",
        "oldpeak": "ST depression",
        "trestbps": "resting blood pressure",
        "sex_Male": "male sex",
        "exang_True": "exercise-induced angina",
        "cp_atypical angina": "atypical angina chest pain",
        "cp_non-anginal": "non-anginal chest pain",
        "cp_typical angina": "typical angina chest pain",
        "cp_asymptomatic": "asymptomatic chest pain",
        "fbs_True": "fasting blood sugar above 120 mg/dl",
        "restecg_normal": "normal resting ECG",
        "restecg_st-t abnormality": "ST-T abnormality on resting ECG",
        "restecg_lv hypertrophy": "left ventricular hypertrophy",
        "slope_upsloping": "upsloping ST segment",
        "slope_flat": "flat ST segment",
        "slope_downsloping": "downsloping ST segment",
        "ca": "number of major vessels",
        "thal_normal": "normal thalassemia result",
        "thal_fixed defect": "fixed defect thalassemia result",
        "thal_reversable defect": "reversible defect thalassemia result",
    }

    contributors = []

    for _, item in importance_df.iterrows():
        feature = item["Feature"]
        score = item["Importance"]

        if feature in row.index:
            user_val = row[feature]
            base_val = baseline.get(feature)
            if isinstance(user_val, (int, float, np.integer, np.floating)) and isinstance(base_val, (int, float, np.integer, np.floating)):
                if abs(float(user_val) - float(base_val)) > 0:
                    contributors.append((readable_names.get(feature, feature), score))

        elif "_" in feature:
            original_col, category_val = feature.split("_", 1)
            if original_col in row.index and str(row[original_col]) == category_val:
                contributors.append((readable_names.get(feature, feature.replace("_", " ")), score))

    contributors = sorted(contributors, key=lambda x: x[1], reverse=True)
    return contributors[:top_n]


def get_validation_messages(user_input):
    row = user_input.iloc[0]
    messages = []

    if row["age"] < 18:
        messages.append("The dataset mainly represents adults, so predictions for under-18s should be treated with caution.")
    if row["chol"] > 400:
        messages.append("Very high cholesterol input detected. Ensure the value has been entered correctly.")
    if row["trestbps"] > 200:
        messages.append("Very high resting blood pressure input detected. Ensure the value has been entered correctly.")
    if row["oldpeak"] > 6:
        messages.append("A high ST depression value was entered. Please confirm that the value is correct.")

    return messages


@app.route("/")
def home():
    top_features = artifacts["importance_df"].head(10).to_dict(orient="records")
    return render_template("index.html", top_features=top_features)


@app.route("/predict", methods=["POST"])
def predict():
    user_input = pd.DataFrame([{
        "age": float(request.form["age"]),
        "sex": request.form["sex"],
        "cp": request.form["cp"],
        "trestbps": float(request.form["trestbps"]),
        "chol": float(request.form["chol"]),
        "fbs": True if request.form["fbs"] == "True" else False,
        "restecg": request.form["restecg"],
        "thalch": float(request.form["thalch"]),
        "exang": True if request.form["exang"] == "True" else False,
        "oldpeak": float(request.form["oldpeak"]),
        "slope": request.form["slope"],
        "ca": float(request.form["ca"]),
        "thal": request.form["thal"],
    }])

    validation_messages = get_validation_messages(user_input)

    processed_input = prepare_input(user_input, artifacts)
    scaled_input = artifacts["scaler"].transform(processed_input)

    prediction = artifacts["model"].predict(scaled_input)[0]
    probability = artifacts["model"].predict_proba(scaled_input)[0][1]
    contributors = get_top_contributors(user_input, artifacts)
    top_features = artifacts["importance_df"].head(10).to_dict(orient="records")

    prediction_text = "Higher heart disease risk detected" if prediction == 1 else "Lower heart disease risk detected"

    if probability >= 0.75:
        risk_interpretation = "High predicted risk"
    elif probability >= 0.50:
        risk_interpretation = "Moderate predicted risk"
    else:
        risk_interpretation = "Lower predicted risk"

    user_input_summary = user_input.to_dict(orient="records")[0]

    return render_template(
        "result.html",
        prediction_text=prediction_text,
        probability=round(probability * 100, 2),
        risk_interpretation=risk_interpretation,
        contributors=contributors,
        validation_messages=validation_messages,
        user_input_summary=user_input_summary,
        top_features=top_features,
    )


if __name__ == "__main__":
    app.run(debug=True)