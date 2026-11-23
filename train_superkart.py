import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "SuperKart.csv"
MODEL_DIR = ROOT / "backend_files"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_features(df):
    out = df.copy()
    out["Product_Id_char"] = out["Product_Id"].astype(str).str[:2]
    out["Store_Age_Years"] = 2026 - out["Store_Establishment_Year"]
    out["Product_Type_Category"] = out["Product_Type"].apply(
        lambda x: "Perishables" if x in {"Breads", "Breakfast", "Dairy", "Frozen Foods",
                                          "Fruits and Vegetables", "Meat", "Seafood", "Starchy Foods"}
        else "Non Perishables"
    )
    return out

raw = pd.read_csv(DATA)
raw["Product_Sugar_Content"] = raw["Product_Sugar_Content"].replace({"reg": "Regular", "regular": "Regular", "low sugar": "Low Sugar", "no sugar": "No Sugar"})
df = build_features(raw)

feature_cols = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category"
]
target_col = "Product_Store_Sales_Total"
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
numeric = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
categorical = [c for c in feature_cols if c not in numeric]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

candidates = {
    "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=-1), {
        "model__n_estimators": [150, 250], "model__max_depth": [12, None],
        "model__min_samples_leaf": [1, 3]
    }),
    "GradientBoosting": (GradientBoostingRegressor(random_state=42), {
        "model__n_estimators": [150, 250], "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3]
    }),
}

results = []
best = None
best_name = None
best_rmse = float("inf")

for name, (model, params) in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    search = GridSearchCV(pipe, params, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append({"model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2,
                    "best_params": search.best_params_})
    if rmse < best_rmse:
        best_rmse = rmse
        best = search.best_estimator_
        best_name = name

joblib.dump(best, MODEL_DIR / "superkart_model.joblib")
pd.DataFrame(results).sort_values("RMSE").to_csv(MODEL_DIR / "model_comparison.csv", index=False)
(MODEL_DIR / "model_metadata.json").write_text(json.dumps({
    "best_model": best_name,
    "feature_columns": feature_cols,
    "target_column": target_col,
    "numeric_columns": numeric,
    "categorical_columns": categorical,
    "rmse": best_rmse,
    "training_rows": len(X_train),
    "test_rows": len(X_test)
}, indent=2))
print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))
print(f"Selected: {best_name}; test RMSE: {best_rmse:.2f}")
