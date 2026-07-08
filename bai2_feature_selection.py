import pandas as pd, numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("DATA/student-mat.csv", sep=";")
df = df.dropna()

# Mã hoá toàn bộ cột dạng chữ để tính correlation
df_enc = df.copy()
for col in df_enc.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])

# Mục tiêu hồi quy: điểm cuối kỳ G3
target = "G3"
corr = df_enc.corr(numeric_only=True)
corr_target = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
print("Top tương quan với G3:")
print(corr_target.head(12))

# Vẽ heatmap tương quan (toàn bộ)
plt.figure(figsize=(13,11))
sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, linewidths=0.3,
            cbar_kws={"shrink":0.8}, xticklabels=True, yticklabels=True)
plt.title("Ma tran tuong quan (Correlation Heatmap) - Student Performance", fontsize=14)
plt.tight_layout()
plt.savefig("corr_heatmap_full.png", dpi=150)
plt.close()

# Heatmap thu gọn: top 10 biến tương quan mạnh nhất với G3 + G3
top_feats = corr_target.head(10).index.tolist()
sub = df_enc[top_feats + [target]]
plt.figure(figsize=(8,7))
sns.heatmap(sub.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, square=True)
plt.title("Top 10 dac trung tuong quan manh nhat voi G3", fontsize=13)
plt.tight_layout()
plt.savefig("corr_heatmap_top10.png", dpi=150)
plt.close()

# ==== Feature Selection dựa trên Correlation, thử nghiệm với Linear Regression ====
X_all = df_enc.drop(columns=[target])
y = df_enc[target]

feature_sets = {
    "Toan bo features (30)": X_all.columns.tolist(),
    "Top 10 tuong quan cao nhat": corr_target.head(10).index.tolist(),
    "Top 5 tuong quan cao nhat": corr_target.head(5).index.tolist(),
    "Nguong |corr| > 0.1": corr_target[abs(corr_target) > 0.1].index.tolist(),
    "Nguong |corr| > 0.2": corr_target[abs(corr_target) > 0.2].index.tolist(),
}

mae_results = {}
for name, feats in feature_sets.items():
    Xs = X_all[feats]
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred = lr.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)
    mae_results[name] = {"n_features": len(feats), "MAE": round(mae, 4)}
    print(f"{name} | n_features={len(feats)} | MAE={mae:.4f}")

with open("bai2_results.json", "w") as f:
    json.dump({
        "top_corr_with_G3": {k: round(v,3) for k,v in corr_target.head(12).items()},
        "mae_results": mae_results
    }, f, indent=2, ensure_ascii=False)

# Bar chart so sánh MAE giữa các tập feature
plt.figure(figsize=(9,5))
names = list(mae_results.keys())
maes = [mae_results[n]["MAE"] for n in names]
colors = ["#0F766E","#14B8A6","#2DD4BF","#5EEAD4","#99F6E4"]
bars = plt.bar(range(len(names)), maes, color=colors[:len(names)])
plt.xticks(range(len(names)), [n.replace(" ","\n") for n in names], fontsize=9)
plt.ylabel("MAE (Mean Absolute Error)")
plt.title("So sanh MAE - Linear Regression tren cac tap feature khac nhau")
for b, m in zip(bars, maes):
    plt.text(b.get_x()+b.get_width()/2, m+0.02, f"{m:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("mae_comparison.png", dpi=150)
plt.close()

print("DONE")
