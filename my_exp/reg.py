# xgb_regress_csv.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)           # comma-separated
    return df

def build_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("name", "target")]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(float).values
    names = df["name"].astype(str).values
    return X, y, names, feature_cols

def evaluate(model, y_te, y_pred, name):
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred, squared=False)
    r2   = r2_score(y_te, y_pred)
    print(name, " Results")
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    print(f"Best n_estimators: {model.best_iteration + 1 if hasattr(model, 'best_iteration') else model.n_estimators}")


def main():
    datafile = "final_data/reg.csv "
    outdir = Path('resultt')# Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_csv(datafile)
    X, y, names, feature_cols = build_xy(df)

    # 2) Split (random; for molecules consider scaffold split later)
    X_tr, X_te, y_tr, y_te, n_tr, n_te = train_test_split(
        X, y, names, test_size=0.2, random_state=42
    )

    # 3) XGBoost regressor (trees don't need scaling)
    model = XGBRegressor(
        n_estimators=5000,          # large, rely on early stopping
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,          # very effective parameter 0.45 -> 0.43
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    # 4) Fit with early stopping using the hold-out as eval_set
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        # early_stopping_rounds=100,
        verbose=False
    )

    # 5) Evaluate
    y_pred = model.predict(X_te)
    evaluate(model=model, y_te=y_te, y_pred=y_pred, name= "XGBoost")

    # 3b) Random Forest (in parallel)
    rf = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    evaluate(model=rf,y_te=y_te, y_pred=y_pred_rf, name="Random Forest")

    '''
    # 6) Save predictions
    pd.DataFrame({"name": n_te, "y_true": y_te, "y_pred": y_pred})\
      .to_csv(outdir / "predictions.csv", index=False)

    # 7) Save feature importances
    imp = model.feature_importances_
    pd.DataFrame({"feature": feature_cols, "importance": imp})\
      .sort_values("importance", ascending=False)\
      .to_csv(outdir / "feature_importances.csv", index=False)

    # 8) Save model
    model.save_model(str(outdir / "xgb_model.json"))'''

if __name__ == "__main__":
    main()
