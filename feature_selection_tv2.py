"""
THÀNH VIÊN 2 - Bài 2: Feature Selection (Correlation) + Linear Regression + MAE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Đọc & tiền xử lý dữ liệu
df = pd.read_csv("DATA/student-mat.csv", sep=";")
df = df.dropna()

# Dự đoán G3 (hồi quy); bỏ G1, G2 để tránh rò rỉ dữ liệu (data leakage)
target = "G3"
X_raw = df.drop(columns=["G1", "G2", "G3"])
y = df[target]

X_encoded = X_raw.copy()
for col in X_encoded.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

print("Tiền xử lý dữ liệu thành công.")

# 2. Ma trận tương quan + Heatmap
corr_data = X_encoded.copy()
corr_data[target] = y
corr_matrix = corr_data.corr()

plt.figure(figsize=(16, 13))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False,
            linewidths=0.4, cbar_kws={"label": "Hệ số tương quan"})
plt.title("Ma trận tương quan - Student Performance Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap_full.png", dpi=150)
plt.close()

corr_with_target = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
print("\nTương quan với G3 (giảm dần theo |corr|):")
print(corr_with_target.to_string())

plt.figure(figsize=(9, 10))
colors = ["#d62728" if v < 0 else "#1f77b4" for v in corr_with_target.values]
plt.barh(corr_with_target.index[::-1], corr_with_target.values[::-1], color=colors[::-1])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Hệ số tương quan với G3")
plt.title("Tương quan giữa từng đặc trưng và điểm G3", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_with_target.png", dpi=150)
plt.close()

# 3. Chọn các tập features theo mức độ tương quan
abs_corr = corr_with_target.abs()
feature_sets = {
    "Toàn bộ features (Full)": list(X_encoded.columns),
    "Tương quan cao (|corr| >= 0.10)": abs_corr[abs_corr >= 0.10].index.tolist(),
    "Top 5 features tương quan mạnh nhất": abs_corr.sort_values(ascending=False).head(5).index.tolist(),
    "Tương quan thấp (|corr| < 0.05) - loại các feature mạnh": abs_corr[abs_corr < 0.05].index.tolist(),
}

print("\nCác tập features được chọn:")
for name, cols in feature_sets.items():
    print(f" - {name} ({len(cols)} features): {cols}")

# 4. Linear Regression trên từng tập features + so sánh MAE
results = []
for set_name, cols in feature_sets.items():
    if len(cols) == 0:
        continue

    X_sub = X_encoded[cols]
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    results.append({"Feature set": set_name, "Số features": len(cols), "MAE": mae})

results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
print("\n===== SO SÁNH MAE GIỮA CÁC TẬP FEATURES =====")
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]
print(f"\n>>> Tốt nhất: '{best_row['Feature set']}' - MAE = {best_row['MAE']:.4f} "
      f"({best_row['Số features']} features)")

plt.figure(figsize=(9, 5))
bars = plt.bar(results_df["Feature set"], results_df["MAE"], color="#2ca02c")
plt.ylabel("MAE (thấp hơn = tốt hơn)")
plt.title("So sánh MAE giữa các tập features (Linear Regression)", fontsize=13, fontweight="bold")
plt.xticks(rotation=20, ha="right")
for b, v in zip(bars, results_df["MAE"]):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("mae_comparison.png", dpi=150)
plt.close()

results_df.to_csv("bai2_mae_results.csv", index=False)
print("\nĐã lưu: correlation_heatmap_full.png, correlation_with_target.png, mae_comparison.png, bai2_mae_results.csv")
