"""
Car Price Prediction — Local Training Script (VS Code version)
================================================================
Same model logic as the Colab version, adapted to run as a plain script:
 - No kagglehub (loads car_data_augmented.csv directly)
 - Shows 6 pop-up plot windows AND saves them as PNGs
 - Saves car_price_model.pkl, feature_cols.pkl, top_brands.pkl at the end
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from xgboost import XGBRegressor, XGBClassifier

sns.set_style("whitegrid")
RANDOM_STATE = 42
OUTPUT_DIR = "."

# ------------------------------------------------------------------
# 1. Load Data & Base Preprocessing
# ------------------------------------------------------------------
df = pd.read_csv("car_data_augmented.csv")

CURRENT_YEAR = 2026
df["car_age"] = CURRENT_YEAR - df["Year"]

df["brand"] = df["Car_Name"].str.split().str[0].str.lower()
top_brands = df["brand"].value_counts().head(10).index
df["brand"] = df["brand"].where(df["brand"].isin(top_brands), "other")

cat_cols = ["brand", "Fuel_Type", "Seller_Type", "Transmission"]
num_cols = ["car_age", "Present_Price", "Kms_Driven", "Owner"]

df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
feature_cols = [c for c in df_enc.columns if c not in ("Selling_Price", "Year", "Car_Name")]

X = df_enc[feature_cols]
y = df_enc["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scaler_transformer = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="passthrough"
)

# ==================================================================
# PART A — REGRESSION
# ==================================================================
print("=" * 65)
print("PART A: REGRESSION MODELS (predicting Selling_Price)")
print("=" * 65)

reg_results = {}

def evaluate_pipeline(name, pipeline, Xtr, Xte, ytr, yte):
    pipeline.fit(Xtr, ytr)
    preds = pipeline.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    rmse = np.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)

    cv_scores = cross_validate(
        pipeline, Xtr, ytr, cv=kf,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error"}
    )
    cv_r2_mean, cv_r2_std = cv_scores["test_r2"].mean(), cv_scores["test_r2"].std()
    cv_mae_mean = -cv_scores["test_mae"].mean()

    reg_results[name] = {
        "Test_MAE": mae, "Test_RMSE": rmse, "Test_R2": r2,
        "CV_R2_mean": cv_r2_mean, "CV_R2_std": cv_r2_std, "CV_MAE_mean": cv_mae_mean,
    }
    print(f"{name:32s} Test R2: {r2:.4f}  Test MAE: {mae:.3f}  |  "
          f"CV R2: {cv_r2_mean:.4f} (+/-{cv_r2_std:.4f})  CV MAE: {cv_mae_mean:.3f}")
    return pipeline, preds

lr_pipe = Pipeline([("lr", LinearRegression())])
lr_model, lr_preds = evaluate_pipeline("Linear Regression (baseline)", lr_pipe, X_train, X_test, y_train, y_test)

ridge_pipe = Pipeline([
    ("scaler", scaler_transformer),
    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=kf))
])
ridge_model, ridge_preds = evaluate_pipeline("Ridge Regression", ridge_pipe, X_train, X_test, y_train, y_test)

lasso_pipe = Pipeline([
    ("scaler", scaler_transformer),
    ("lasso", LassoCV(alphas=np.logspace(-3, 1, 50), cv=kf, max_iter=20000, random_state=RANDOM_STATE))
])
lasso_model, lasso_preds = evaluate_pipeline("Lasso Regression", lasso_pipe, X_train, X_test, y_train, y_test)

poly_transformer = ColumnTransformer(
    transformers=[("poly_num", PolynomialFeatures(degree=2, include_bias=False), num_cols)],
    remainder="passthrough"
)
poly_ridge_pipe = Pipeline([
    ("poly", poly_transformer),
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
param_grid_poly = {"ridge__alpha": np.logspace(-2, 3, 20)}
poly_grid = GridSearchCV(poly_ridge_pipe, param_grid_poly, cv=kf, scoring="r2")
poly_model, poly_preds = evaluate_pipeline("Polynomial (deg=2) + Ridge", poly_grid, X_train, X_test, y_train, y_test)

xgb_base = XGBRegressor(random_state=RANDOM_STATE)
xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
xgb_search = RandomizedSearchCV(
    xgb_base, param_distributions=xgb_param_dist, n_iter=15,
    cv=kf, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model, xgb_preds = evaluate_pipeline("XGBoost Regressor", xgb_search, X_train, X_test, y_train, y_test)

reg_results_df = pd.DataFrame(reg_results).T.sort_values("Test_R2", ascending=False)
reg_results_df.to_csv(f"{OUTPUT_DIR}/regression_results_real.csv")
print("\nRegression comparison (sorted by Test R2):")
print(reg_results_df.round(4))

best_reg_name = reg_results_df.index[0]
best_model_lookup = {
    "Linear Regression (baseline)": lr_model,
    "Ridge Regression": ridge_model,
    "Lasso Regression": lasso_model,
    "Polynomial (deg=2) + Ridge": poly_model,
    "XGBoost Regressor": xgb_model,
}
final_model = best_model_lookup[best_reg_name]

best_xgb_reg = xgb_model.best_estimator_
importances = pd.Series(
    best_xgb_reg.feature_importances_, index=feature_cols
).sort_values(ascending=False).head(12)

# ==================================================================
# PART B — CLASSIFICATION
# ==================================================================
print("\n" + "=" * 65)
print("PART B: CLASSIFICATION MODELS (price bracket)")
print("=" * 65)

q1, q2 = y_train.quantile([0.33, 0.66])
print(f"Train Bucket Cutoffs (lakhs INR) -> Budget: < {q1:.2f} | Mid-Range: {q1:.2f}-{q2:.2f} | Premium: > {q2:.2f}")

def bucket_price(p):
    if p < q1:
        return 0
    elif p < q2:
        return 1
    return 2

y_train_class = y_train.apply(bucket_price)
y_test_class = y_test.apply(bucket_price)
labels_order = ["Budget", "Mid-Range", "Premium"]
yc_test_bin = label_binarize(y_test_class, classes=[0, 1, 2])

def evaluate_classifier(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)
    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="macro")
    print(f"\n{name}  ->  Accuracy: {acc:.4f}   Macro F1: {f1:.4f}")
    print(classification_report(yte, preds, target_names=labels_order))
    return model, preds, proba, acc, f1

logreg_pipe = Pipeline([
    ("scaler", scaler_transformer),
    ("logreg", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])
logreg_param_grid = {"logreg__C": [0.01, 0.1, 1.0, 10.0]}
logreg_grid = GridSearchCV(logreg_pipe, logreg_param_grid, cv=kf, scoring="f1_macro")
logreg_model, logreg_preds, logreg_proba, logreg_acc, logreg_f1 = evaluate_classifier(
    "Logistic Regression", logreg_grid, X_train, X_test, y_train_class, y_test_class
)

xgb_clf_base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss")
xgb_clf_param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9]
}
xgb_clf_search = RandomizedSearchCV(
    xgb_clf_base, param_distributions=xgb_clf_param_dist, n_iter=10,
    cv=kf, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1
)
xgb_clf_model, xgb_clf_preds, xgb_clf_proba, xgb_acc, xgb_f1 = evaluate_classifier(
    "XGBoost Classifier", xgb_clf_search, X_train, X_test, y_train_class, y_test_class
)

class_results_df = pd.DataFrame({
    "Logistic Regression": {"Accuracy": logreg_acc, "Macro F1": logreg_f1},
    "XGBoost Classifier": {"Accuracy": xgb_acc, "Macro F1": xgb_f1},
}).T
class_results_df.to_csv(f"{OUTPUT_DIR}/classification_results_real.csv")
print("\nClassification comparison:")
print(class_results_df.round(4))

# ==================================================================
# VISUALIZATIONS — pop-up windows + saved PNGs
# ==================================================================
best_reg_preds = {
    "Linear Regression (baseline)": lr_preds, "Ridge Regression": ridge_preds,
    "Lasso Regression": lasso_preds, "Polynomial (deg=2) + Ridge": poly_preds,
    "XGBoost Regressor": xgb_preds,
}[best_reg_name]
best_clf_preds = xgb_clf_preds if xgb_f1 >= logreg_f1 else logreg_preds
best_clf_name = "XGBoost Classifier" if xgb_f1 >= logreg_f1 else "Logistic Regression"
colors = ["#e07a5f", "#3d5a80", "#81b29a"]

fig1, ax1 = plt.subplots(figsize=(7, 6))
ax1.scatter(y_test, best_reg_preds, alpha=0.5, s=25, color="#3b6ea5")
lims = [min(y_test.min(), best_reg_preds.min()), max(y_test.max(), best_reg_preds.max())]
ax1.plot(lims, lims, "r--", linewidth=1.5)
ax1.set_xlabel("Actual Selling Price (lakhs)")
ax1.set_ylabel("Predicted Selling Price (lakhs)")
ax1.set_title(f"Best Regressor: {best_reg_name}")
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(8, 6))
x_pos = np.arange(len(reg_results_df))
width = 0.35
ax2.bar(x_pos - width/2, reg_results_df["Test_R2"], width, label="Test R2", color="#3b6ea5")
ax2.bar(x_pos + width/2, reg_results_df["CV_R2_mean"], width, label="5-fold CV R2", color="#8fb8dd")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(reg_results_df.index, rotation=35, ha="right", fontsize=8)
ax2.set_ylabel("R2 Score")
ax2.set_title("Regression: Test R2 vs Cross-Validated R2")
ax2.legend()
plt.tight_layout()
plt.savefig("plot2_regression_comparison.png", dpi=150)

fig3, ax3 = plt.subplots(figsize=(7, 6))
ax3.barh(importances.index[::-1], importances.values[::-1], color="#5a9367")
ax3.set_title("Top Feature Importances (XGBoost Regressor)")
ax3.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("plot3_feature_importances.png", dpi=150)

fig4, ax4 = plt.subplots(figsize=(6, 6))
cm = confusion_matrix(y_test_class, best_clf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order,
            yticklabels=labels_order, ax=ax4)
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_title(f"Confusion Matrix: {best_clf_name}")
plt.tight_layout()
plt.savefig("plot4_confusion_matrix.png", dpi=150)

def compute_roc(proba, y_bin):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["macro"], tpr["macro"], _ = roc_curve(y_bin.ravel(), proba.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

fpr_log, tpr_log, auc_log = compute_roc(logreg_proba, yc_test_bin)
fpr_xgb, tpr_xgb, auc_xgb = compute_roc(xgb_clf_proba, yc_test_bin)

fig5, ax5 = plt.subplots(figsize=(7, 6))
for i, c in zip(range(3), colors):
    ax5.plot(fpr_log[i], tpr_log[i], color=c, label=f"{labels_order[i]} (AUC={auc_log[i]:.2f})")
ax5.plot(fpr_log["macro"], tpr_log["macro"], "k--", label=f"Macro-avg (AUC={auc_log['macro']:.2f})")
ax5.plot([0, 1], [0, 1], color="gray", linestyle=":")
ax5.set_xlabel("False Positive Rate")
ax5.set_ylabel("True Positive Rate")
ax5.set_title("ROC Curve: Logistic Regression (OvR)")
ax5.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plot5_roc_logistic_regression.png", dpi=150)

fig6, ax6 = plt.subplots(figsize=(7, 6))
for i, c in zip(range(3), colors):
    ax6.plot(fpr_xgb[i], tpr_xgb[i], color=c, label=f"{labels_order[i]} (AUC={auc_xgb[i]:.2f})")
ax6.plot(fpr_xgb["macro"], tpr_xgb["macro"], "k--", label=f"Macro-avg (AUC={auc_xgb['macro']:.2f})")
ax6.plot([0, 1], [0, 1], color="gray", linestyle=":")
ax6.set_xlabel("False Positive Rate")
ax6.set_ylabel("True Positive Rate")
ax6.set_title("ROC Curve: XGBoost Classifier (OvR)")
ax6.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plot6_roc_xgboost.png", dpi=150)

print("\nOpening 6 plot windows — close each one to continue...")
plt.show()

# ==================================================================
# EXPORT — saves the 3 files app.py needs, right in this same folder
# ==================================================================
joblib.dump(final_model, "car_price_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(top_brands.tolist(), "top_brands.pkl")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Best regressor: {best_reg_name} "
      f"(Test R2={reg_results_df.loc[best_reg_name,'Test_R2']:.4f}, "
      f"CV R2={reg_results_df.loc[best_reg_name,'CV_R2_mean']:.4f})")
print(f"Best classifier: {best_clf_name} (Macro F1={max(xgb_f1, logreg_f1):.4f})")
print("\nSaved: car_price_model.pkl, feature_cols.pkl, top_brands.pkl")