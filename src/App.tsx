import { useState } from "react";

const models = [
  {
    category: "🔵 SUPERVISED — Regression",
    color: "#3b82f6",
    items: [
      {
        name: "Linear Regression",
        lib: "scikit-learn",
        when: "Quan hệ tuyến tính, dự đoán số liên tục",
        theory: "Linear Regression mô hình hoá quan hệ tuyến tính giữa biến độc lập và biến phụ thuộc. Nó học hệ số góc và hệ số chặn bằng cách tối thiểu hóa sai số bình phương, phù hợp cho dự đoán biến liên tục khi dữ liệu không quá phức tạp.",
        code: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dữ liệu: dự đoán giá nhà theo diện tích
X = np.array([[50], [60], [70], [80], [90], [100], [120], [150]])
y = np.array([150, 180, 210, 240, 270, 300, 360, 450])  # giá (triệu)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
print(f"Hệ số góc (w): {model.coef_[0]:.2f}")
print(f"Hệ số chặn (b): {model.intercept_:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Dự đoán nhà 110m²
print(f"Dự đoán nhà 110m²: {model.predict([[110]])[0]:.0f} triệu")`,
      },
      {
        name: "Ridge / Lasso Regression",
        lib: "scikit-learn",
        when: "Dữ liệu có nhiều feature tương quan (multicollinearity), tránh overfitting",
        code: `from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Dữ liệu nhiều feature (diện tích, số phòng, tuổi nhà, tầng)
X = np.array([
    [50, 1, 10, 2], [70, 2, 5, 3], [90, 3, 2, 1],
    [60, 2, 8, 4], [100, 3, 1, 5], [80, 2, 3, 2],
    [110, 4, 0, 6], [75, 2, 6, 3]
])
y = np.array([150, 210, 280, 180, 320, 240, 360, 225])

# Ridge Regression (L2) — giữ tất cả features, giảm trọng số lớn
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
ridge.fit(X, y)
print("Ridge coef:", ridge.named_steps["model"].coef_)

# Lasso Regression (L1) — tự động loại bỏ feature không quan trọng
lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=0.5))
])
lasso.fit(X, y)
print("Lasso coef:", lasso.named_steps["model"].coef_)
# Coef = 0 → feature đó bị loại!

# ElasticNet — kết hợp L1 + L2
elastic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.5, l1_ratio=0.5))
])
elastic.fit(X, y)
print("ElasticNet coef:", elastic.named_steps["model"].coef_)`,
      },
      {
        name: "XGBoost Regressor",
        lib: "xgboost",
        when: "Dữ liệu tabular, cần độ chính xác cao, thi Kaggle",
        theory: "XGBoost là một thuật toán boosting hiệu suất cao, xây dựng dần dần các cây quyết định. Nó tối ưu hóa sai số theo hàm mất mát và regularization, phù hợp với dữ liệu phức tạp và có thể xử lý tương quan giữa đặc trưng tốt.",
        code: `import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

# Dữ liệu
X = np.random.rand(1000, 10)  # 1000 mẫu, 10 features
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.randn(1000)*0.1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện XGBoost
model = xgb.XGBRegressor(
    n_estimators=200,      # số cây
    max_depth=6,           # độ sâu tối đa mỗi cây
    learning_rate=0.1,     # learning rate
    subsample=0.8,         # lấy mẫu 80% data mỗi cây
    colsample_bytree=0.8,  # lấy mẫu 80% feature mỗi cây
    random_state=42,
    verbosity=0
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Đánh giá
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Feature importance
import pandas as pd
feat_imp = pd.Series(model.feature_importances_,
                     index=[f"f{i}" for i in range(10)])
print(feat_imp.sort_values(ascending=False))`,
      },
    ],
  },
  {
    category: "🔵 SUPERVISED — Classification",
    color: "#6366f1",
    items: [
      {
        name: "Logistic Regression",
        lib: "scikit-learn",
        when: "Binary/multiclass classification, cần giải thích được, baseline nhanh",
        theory: "Logistic Regression ước lượng xác suất cho từng lớp bằng hàm sigmoid/softmax. Nó là mô hình tuyến tính cho classification, dễ hiểu và thường dùng làm baseline khi dữ liệu không quá phức tạp.",
        code: `from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Dataset: phân loại ung thư vú (lành tính / ác tính)
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale dữ liệu (quan trọng với Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện
model = LogisticRegression(
    C=1.0,           # nghịch đảo regularization (lớn = ít regularize)
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Đánh giá
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred,
      target_names=["Ác tính", "Lành tính"]))

# Xác suất dự đoán
proba = model.predict_proba(X_test[:3])
print("Xác suất [ác tính, lành tính]:", proba)`,
      },
      {
        name: "Random Forest Classifier",
        lib: "scikit-learn",
        when: "Baseline mạnh, ít tuning, dữ liệu vừa-lớn, ít quan tâm tốc độ",
        theory: "Random Forest là tập hợp nhiều cây quyết định độc lập, mỗi cây học trên bootstrap sample và chỉ xét một subset đặc trưng. Việc lấy trung bình dự đoán giúp giảm overfitting và tăng độ ổn định so với một cây đơn.",
        code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Dataset: phân loại hoa Iris (3 loài)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # số cây
    max_depth=None,        # không giới hạn độ sâu
    min_samples_split=2,
    max_features="sqrt",   # căn bậc 2 số features mỗi split
    random_state=42,
    n_jobs=-1              # dùng tất cả CPU cores
)
rf.fit(X_train, y_train)

# Đánh giá
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Feature Importance
feat_names = load_iris().feature_names
for name, imp in sorted(zip(feat_names, rf.feature_importances_),
                        key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")`,
      },
      {
        name: "SVM",
        lib: "scikit-learn",
        when: "Dữ liệu nhỏ-vừa, high-dimensional, binary classification",
        code: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

# Tạo dữ liệu phân loại
X, y = make_classification(
    n_samples=500, n_features=20,
    n_informative=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM với RBF kernel (mạnh nhất)
svm = Pipeline([
    ("scaler", StandardScaler()),  # SVM rất cần scale dữ liệu!
    ("model", SVC(
        kernel="rbf",  # rbf | linear | poly | sigmoid
        C=1.0,         # penalty (lớn = fit chặt hơn)
        gamma="scale", # bandwidth của RBF kernel
        probability=True  # bật để lấy predict_proba
    ))
])
svm.fit(X_train, y_train)

# Đánh giá
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

# Tìm hyperparameter tốt nhất
param_grid = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", "auto", 0.1]
}
grid = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)`,
      },
      {
        name: "Naive Bayes",
        lib: "scikit-learn",
        when: "Text classification, spam filter, dữ liệu nhỏ, cần train nhanh",
        code: `from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ===== Ví dụ 1: Phân loại email spam =====
emails = [
    "Chúc mừng! Bạn đã trúng thưởng 1 tỷ đồng",
    "Nhận ngay ưu đãi độc quyền hôm nay",
    "Họp team lúc 3 giờ chiều nhé",
    "Báo cáo tháng 6 đã xong, anh review giúp",
    "Mua ngay giảm 90% chỉ hôm nay",
    "Tài liệu họp đính kèm bên dưới",
]
labels = [1, 1, 0, 0, 1, 0]  # 1=spam, 0=ham

# Pipeline: TF-IDF → Naive Bayes
spam_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB(alpha=1.0))  # alpha: Laplace smoothing
])
spam_model.fit(emails, labels)

test_emails = ["Trúng thưởng iphone 15 miễn phí",
               "Xem lại meeting notes hôm qua"]
preds = spam_model.predict(test_emails)
for email, pred in zip(test_emails, preds):
    print(f"{'SPAM' if pred else 'HAM'}: {email}")

# ===== Ví dụ 2: GaussianNB cho dữ liệu liên tục =====
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
gnb = GaussianNB()
gnb.fit(X[:120], y[:120])
print(f"\\nGaussianNB Accuracy: {gnb.score(X[120:], y[120:]):.4f}")`,
      },
      {
        name: "KNN",
        lib: "scikit-learn",
        when: "Dữ liệu nhỏ, không cần train, lazy learning, recommender đơn giản",
        code: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine
import numpy as np

# Dataset: phân loại rượu vang (3 loại)
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale dữ liệu (KNN dùng khoảng cách nên phải scale!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tìm K tốt nhất
k_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    k_scores.append(scores.mean())

best_k = np.argmax(k_scores) + 1
print(f"K tốt nhất: {best_k}, CV Accuracy: {max(k_scores):.4f}")

# Huấn luyện với K tốt nhất
knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean",  # euclidean | manhattan | minkowski
    weights="uniform"    # uniform | distance
)
knn.fit(X_train, y_train)
print(f"Test Accuracy: {knn.score(X_test, y_test):.4f}")`,
      },
    ],
  },
  {
    category: "🟢 UNSUPERVISED — Clustering",
    color: "#10b981",
    items: [
      {
        name: "K-Means",
        lib: "scikit-learn",
        when: "Biết trước số cụm, cụm hình tròn, dữ liệu lớn",
        code: `from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Dữ liệu: phân khúc khách hàng (tuổi, thu nhập, chi tiêu)
np.random.seed(42)
customers = np.array([
    *np.random.randn(50, 3) + [25, 30, 20],  # nhóm trẻ, thu nhập thấp
    *np.random.randn(50, 3) + [45, 70, 80],  # nhóm trung niên, thu nhập cao
    *np.random.randn(50, 3) + [60, 50, 40],  # nhóm lớn tuổi, trung bình
])

# Scale dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(customers)

# Tìm số K tối ưu bằng Elbow Method
inertias = []
silhouettes = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))
    print(f"K={k}: Inertia={km.inertia_:.0f}, "
          f"Silhouette={silhouette_score(X, km.labels_):.4f}")

# Chạy với K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Phân tích từng cụm
for i in range(3):
    cluster = customers[labels == i]
    print(f"\\nCụm {i+1} ({sum(labels==i)} khách):")
    print(f"  Tuổi TB: {cluster[:,0].mean():.1f}")
    print(f"  Thu nhập TB: {cluster[:,1].mean():.1f}")`,
      },
      {
        name: "DBSCAN",
        lib: "scikit-learn",
        when: "Cụm hình dạng bất kỳ, có outlier/noise, không biết số cụm",
        code: `from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import numpy as np

# Dữ liệu hình lưỡi liềm (K-Means không handle được)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Thêm outliers
outliers = np.random.uniform(-2, 3, (20, 2))
X = np.vstack([X, outliers])

# DBSCAN
dbscan = DBSCAN(
    eps=0.3,        # bán kính vùng lân cận
    min_samples=5,  # số điểm tối thiểu để tạo core point
    metric="euclidean"
)
labels = dbscan.fit_predict(X)

# Phân tích kết quả
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(labels == -1)
print(f"Số cụm phát hiện: {n_clusters}")
print(f"Điểm nhiễu (noise): {n_noise}")
print(f"Phân bổ cụm: {dict(zip(*np.unique(labels, return_counts=True)))}")
# label = -1 nghĩa là outlier/noise

# Lọc ra outliers
clean_mask = labels != -1
X_clean = X[clean_mask]
labels_clean = labels[clean_mask]
print(f"Dữ liệu sạch sau khi loại outlier: {len(X_clean)} điểm")`,
      },
      {
        name: "Gaussian Mixture Model",
        lib: "scikit-learn",
        when: "Cụm chồng lấp, cần xác suất thuộc cụm, dữ liệu có phân phối Gaussian",
        code: `from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np

# Dữ liệu với cụm chồng lấp
X, y_true = make_blobs(
    n_samples=300, centers=3,
    cluster_std=1.5, random_state=42
)

# Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,     # số cụm
    covariance_type="full",  # full | tied | diag | spherical
    max_iter=200,
    random_state=42
)
gmm.fit(X)

# Dự đoán cụm
labels = gmm.predict(X)

# XÁC SUẤT thuộc từng cụm (khác K-Means — soft assignment)
proba = gmm.predict_proba(X)
print("Xác suất 5 điểm đầu thuộc từng cụm:")
print(np.round(proba[:5], 3))
# Mỗi điểm có XS thuộc nhiều cụm → "mềm" hơn K-Means

# Thông số phân phối
print("\\nTâm của từng cụm:")
print(gmm.means_)
print("\\nBIC (chọn K tối ưu — thấp hơn = tốt hơn):")
for k in range(2, 6):
    g = GaussianMixture(n_components=k, random_state=42)
    g.fit(X)
    print(f"  K={k}: BIC={g.bic(X):.1f}")`,
      },
    ],
  },
  {
    category: "🟢 UNSUPERVISED — Dimensionality Reduction",
    color: "#f59e0b",
    items: [
      {
        name: "PCA",
        lib: "scikit-learn",
        when: "Giảm chiều tuyến tính, tiền xử lý trước khi train, visualize",
        code: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Dataset: ảnh chữ số viết tay 8x8 = 64 chiều
X, y = load_digits(return_X_y=True)
print(f"Shape gốc: {X.shape}")  # (1797, 64)

# Scale trước PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: giữ 95% variance
pca = PCA(n_components=0.95)  # hoặc n_components=30
X_pca = pca.fit_transform(X_scaled)
print(f"Shape sau PCA: {X_pca.shape}")  # giảm còn ~30 chiều
print(f"Variance giải thích: {pca.explained_variance_ratio_.sum():.2%}")

# Xem contribution của từng component
print("\\nTop 5 components - variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")

# Giảm xuống 2D để visualize
pca2d = PCA(n_components=2)
X_2d = pca2d.fit_transform(X_scaled)
print(f"\\nShape 2D: {X_2d.shape}")
# Có thể vẽ scatter plot phân cụm 10 chữ số`,
      },
      {
        name: "t-SNE",
        lib: "scikit-learn",
        when: "Visualize dữ liệu nhiều chiều trong 2D/3D — KHÔNG dùng để train",
        code: `from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Dataset: 1797 ảnh chữ số, 64 chiều
X, y = load_digits(return_X_y=True)

# Dùng subset để nhanh hơn
X_sub, y_sub = X[:500], y[:500]

# Scale dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sub)

# t-SNE — giảm xuống 2D
tsne = TSNE(
    n_components=2,
    perplexity=30,      # ~số hàng xóm (5-50), ảnh hưởng cấu trúc
    learning_rate=200,  # hoặc "auto"
    n_iter=1000,        # số vòng lặp tối ưu
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)
print(f"Shape sau t-SNE: {X_tsne.shape}")  # (500, 2)

# Kết quả: mỗi chữ số tạo thành 1 cụm riêng biệt
for digit in range(10):
    mask = y_sub == digit
    center = X_tsne[mask].mean(axis=0)
    print(f"Chữ số {digit}: center = ({center[0]:.1f}, {center[1]:.1f})")

# LƯU Ý: t-SNE không thể transform dữ liệu mới!
# Chỉ dùng để VISUALIZE, không dùng cho downstream task`,
      },
    ],
  },
  {
    category: "🟣 ENSEMBLE METHODS",
    color: "#8b5cf6",
    items: [
      {
        name: "XGBoost Classifier (đầy đủ)",
        lib: "xgboost",
        when: "Dữ liệu tabular — thường cho kết quả tốt nhất, chuẩn Kaggle",
        code: `import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification
import numpy as np

# Dữ liệu phân loại nhị phân
X, y = make_classification(
    n_samples=2000, n_features=20,
    n_informative=15, n_redundant=3,
    weights=[0.7, 0.3],  # imbalanced data
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tính scale_pos_weight cho imbalanced data
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # xử lý imbalanced
    use_label_encoder=False,
    eval_metric="auc",
    early_stopping_rounds=20,          # dừng sớm nếu không improve
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Đánh giá
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
feat_imp = model.feature_importances_
top_features = np.argsort(feat_imp)[::-1][:5]
print("\\nTop 5 features:", top_features)`,
      },
    ],
  },
  {
    category: "🔴 REINFORCEMENT LEARNING",
    color: "#ef4444",
    items: [
      {
        name: "Q-Learning",
        lib: "numpy (thuần)",
        when: "Môi trường rời rạc nhỏ, bảng Q có thể lưu hết, bài toán đơn giản",
        code: `import numpy as np

# ===== Q-Learning: Maze 4x4 =====
# 0=đi được, -1=tường, 10=đích
maze = np.array([
    [ 0,  0,  0,  0],
    [ 0, -1, -1,  0],
    [ 0,  0,  0,  0],
    [-1,  0,  0, 10]
])
ROWS, COLS = maze.shape
n_states = ROWS * COLS
n_actions = 4  # 0=lên, 1=xuống, 2=trái, 3=phải
MOVES = [(-1,0), (1,0), (0,-1), (0,1)]

# Q-Table: state x action
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.1    # learning rate
gamma = 0.95   # discount factor
epsilon = 1.0  # exploration rate

def state_id(r, c): return r * COLS + c
def valid(r, c): return 0 <= r < ROWS and 0 <= c < COLS

# Training
for episode in range(5000):
    state = (0, 0)  # start
    epsilon = max(0.01, epsilon * 0.999)  # decay epsilon

    for _ in range(100):  # max steps
        sid = state_id(*state)

        # Epsilon-greedy: khám phá vs khai thác
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)  # random
        else:
            action = np.argmax(Q[sid])  # tốt nhất theo Q

        # Thực hiện action
        dr, dc = MOVES[action]
        nr, nc = state[0]+dr, state[1]+dc

        if not valid(nr, nc) or maze[nr,nc] == -1:
            nr, nc = state  # không đi được, đứng yên

        reward = maze[nr, nc]
        done = reward == 10

        # Cập nhật Q-Table (Bellman Equation)
        next_sid = state_id(nr, nc)
        Q[sid, action] += alpha * (
            reward + gamma * np.max(Q[next_sid]) - Q[sid, action]
        )
        state = (nr, nc)
        if done: break

# Test: tìm đường đi tối ưu
print("Q-Table học được — đường đi tối ưu:")
state, path = (0, 0), [(0,0)]
for _ in range(20):
    sid = state_id(*state)
    action = np.argmax(Q[sid])
    dr, dc = MOVES[action]
    state = (state[0]+dr, state[1]+dc)
    path.append(state)
    if maze[state] == 10: break
print("Path:", path)`,
      },
    ],
  },
  {
    category: "🧠 DEEP LEARNING",
    color: "#ec4899",
    items: [
      {
        name: "Neural Network (PyTorch)",
        lib: "pytorch",
        when: "Dữ liệu lớn phức tạp, ảnh, văn bản, âm thanh, quan hệ phi tuyến sâu",
        code: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ===== Dữ liệu =====
X, y = make_classification(n_samples=2000, n_features=20,
                           n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Chuyển sang tensor
X_tr = torch.FloatTensor(X_train)
y_tr = torch.FloatTensor(y_train)
X_te = torch.FloatTensor(X_test)
y_te = torch.FloatTensor(y_test)
train_loader = DataLoader(
    TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True
)

# ===== Định nghĩa mô hình =====
class NeuralNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),   # chuẩn hoá batch
            nn.ReLU(),
            nn.Dropout(0.3),       # dropout tránh overfit
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()           # output xác suất 0-1
        )
    def forward(self, x):
        return self.net(x).squeeze()

model = NeuralNet(in_features=20)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# ===== Training Loop =====
for epoch in range(50):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = (model(X_te) > 0.5).float()
            acc = (test_pred == y_te).float().mean()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Test Acc={acc:.4f}")`,
      },
    ],
  },
];

const KEYWORDS = new Set([
  "import","from","as","def","class","return","for","in","if","else",
  "elif","while","True","False","None","with","not","and","or","print",
  "pass","break","continue","lambda","try","except","finally","raise",
  "yield","global","nonlocal","del","assert","is"
]);

function TokenLine({ line }: { line: string }) {
  // If line is a comment
  if (line.trimStart().startsWith("#")) {
    return <span style={{ color: "#8b949e", fontStyle: "italic" }}>{line}</span>;
  }

  const tokens = [];
  let i = 0;
  while (i < line.length) {
    // Comment mid-line
    if (line[i] === "#") {
      tokens.push(<span key={i} style={{ color: "#8b949e", fontStyle: "italic" }}>{line.slice(i)}</span>);
      break;
    }
    // String: single or double quote
    if (line[i] === '"' || line[i] === "'") {
      const q = line[i];
      let j = i + 1;
      while (j < line.length && line[j] !== q) { if (line[j] === "\\") j++; j++; }
      j++;
      tokens.push(<span key={i} style={{ color: "#a5d6ff" }}>{line.slice(i, j)}</span>);
      i = j; continue;
    }
    // Number
    if (/[0-9]/.test(line[i]) && (i === 0 || /\W/.test(line[i-1]))) {
      let j = i;
      while (j < line.length && /[0-9.]/.test(line[j])) j++;
      tokens.push(<span key={i} style={{ color: "#79c0ff" }}>{line.slice(i, j)}</span>);
      i = j; continue;
    }
    // Word (keyword, builtin, identifier)
    if (/[a-zA-Z_]/.test(line[i])) {
      let j = i;
      while (j < line.length && /[a-zA-Z0-9_]/.test(line[j])) j++;
      const word = line.slice(i, j);
      let color = "#e6edf3";
      if (KEYWORDS.has(word)) color = "#ff7b72";
      else if (/^[A-Z]/.test(word)) color = "#79c0ff";
      else if (word === word.toUpperCase() && word.length > 1) color = "#79c0ff";
      tokens.push(<span key={i} style={{ color }}>{word}</span>);
      i = j; continue;
    }
    // Default char
    tokens.push(<span key={i}>{line[i]}</span>);
    i++;
  }
  return <>{tokens}</>;
}

export default function MLCodeViewer() {
  const [selectedCat, setSelectedCat] = useState(0);
  const [selectedModel, setSelectedModel] = useState(0);
  const [copied, setCopied] = useState(false);

  const cat = models[selectedCat];
  const model = cat.items[selectedModel];

  const handleCopy = () => {
    navigator.clipboard.writeText(model.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      background: "#0d1117",
      minHeight: "100vh",
      color: "#e6edf3",
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        padding: "20px 24px 0",
        borderBottom: "1px solid #21262d",
      }}>
        <h1 style={{
          margin: "0 0 4px",
          fontSize: "18px",
          fontWeight: 700,
          color: "#58a6ff",
          letterSpacing: "0.5px",
        }}>
          🤖 Machine Learning — Code Examples
        </h1>
        <p style={{ margin: "0 0 16px", fontSize: "11px", color: "#8b949e" }}>
          Chi tiết từng model • Python scikit-learn / XGBoost / PyTorch
        </p>

        {/* Category tabs */}
        <div style={{ display: "flex", gap: "4px", overflowX: "auto", paddingBottom: "0" }}>
          {models.map((cat, i) => (
            <button
              key={i}
              onClick={() => { setSelectedCat(i); setSelectedModel(0); }}
              style={{
                padding: "8px 14px",
                border: "none",
                borderRadius: "6px 6px 0 0",
                background: selectedCat === i ? cat.color + "22" : "transparent",
                color: selectedCat === i ? cat.color : "#8b949e",
                cursor: "pointer",
                fontSize: "11px",
                fontWeight: selectedCat === i ? 700 : 400,
                whiteSpace: "nowrap",
                borderBottom: selectedCat === i ? `2px solid ${cat.color}` : "2px solid transparent",
                transition: "all 0.15s",
              }}
            >
              {cat.category}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        {/* Sidebar — model list */}
        <div style={{
          width: "220px",
          minWidth: "220px",
          background: "#161b22",
          borderRight: "1px solid #21262d",
          padding: "12px 8px",
          overflowY: "auto",
        }}>
          {cat.items.map((item, i) => (
            <button
              key={i}
              onClick={() => setSelectedModel(i)}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "10px 12px",
                marginBottom: "4px",
                border: "none",
                borderRadius: "6px",
                background: selectedModel === i ? cat.color + "20" : "transparent",
                borderLeft: selectedModel === i ? `3px solid ${cat.color}` : "3px solid transparent",
                color: selectedModel === i ? "#e6edf3" : "#8b949e",
                cursor: "pointer",
                fontSize: "12px",
                fontWeight: selectedModel === i ? 600 : 400,
                transition: "all 0.15s",
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: "3px" }}>{item.name}</div>
              <div style={{ fontSize: "10px", color: cat.color, opacity: 0.8 }}>
                📦 {item.lib}
              </div>
            </button>
          ))}
        </div>

        {/* Main content */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          {/* Model header */}
          <div style={{
            padding: "16px 20px",
            background: "#161b22",
            borderBottom: "1px solid #21262d",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <h2 style={{
                  margin: "0 0 6px",
                  fontSize: "16px",
                  color: cat.color,
                  fontWeight: 700,
                }}>
                  {model.name}
                </h2>
                <div style={{
                  fontSize: "11px",
                  color: "#8b949e",
                  background: "#21262d",
                  padding: "4px 10px",
                  borderRadius: "4px",
                  display: "inline-block",
                  marginBottom: "8px",
                }}>
                  📦 {model.lib}
                </div>
                <div style={{
                  fontSize: "12px",
                  color: "#adbac7",
                  marginTop: "6px",
                }}>
                  <span style={{ color: cat.color }}>💡 Dùng khi: </span>
                  {model.when}
                </div>
                {model.theory ? (
                  <div style={{
                    fontSize: "11px",
                    color: "#8b949e",
                    marginTop: "10px",
                    lineHeight: "1.6",
                    maxWidth: "720px",
                  }}>
                    <strong style={{ color: "#c9d1d9" }}>📘 Lý thuyết: </strong>
                    {model.theory}
                  </div>
                ) : null}
              </div>
              <button
                onClick={handleCopy}
                style={{
                  padding: "7px 16px",
                  background: copied ? "#238636" : "#21262d",
                  border: `1px solid ${copied ? "#2ea043" : "#30363d"}`,
                  borderRadius: "6px",
                  color: copied ? "#3fb950" : "#8b949e",
                  cursor: "pointer",
                  fontSize: "11px",
                  fontWeight: 600,
                  transition: "all 0.2s",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                  marginLeft: "16px",
                }}
              >
                {copied ? "✅ Copied!" : "📋 Copy"}
              </button>
            </div>
          </div>

          {/* Code block */}
          <div style={{ flex: 1, overflow: "auto", padding: "0" }}>
            <pre style={{
              margin: 0,
              padding: "20px 24px",
              fontSize: "13px",
              lineHeight: "1.8",
              background: "#0d1117",
              color: "#e6edf3",
              whiteSpace: "pre",
              overflowX: "auto",
              fontFamily: "'JetBrains Mono','Fira Code','Consolas',monospace",
            }}>
              <code style={{ display: "block" }}>
                {model.code.split("\n").map((line, i) => (
                  <div key={i} style={{ display: "flex", minHeight: "1.8em" }}>
                    <span style={{
                      color: "#3d444d", userSelect: "none",
                      marginRight: "20px", minWidth: "32px",
                      textAlign: "right", fontSize: "11px",
                      lineHeight: "1.8", flexShrink: 0,
                    }}>{i + 1}</span>
                    <span style={{ whiteSpace: "pre", flex: 1, textAlign: "left" }}>
                      <TokenLine line={line} />
                    </span>
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{
        padding: "10px 24px",
        background: "#161b22",
        borderTop: "1px solid #21262d",
        fontSize: "11px",
        color: "#3d444d",
        display: "flex",
        justifyContent: "space-between",
      }}>
        <span>Python 3.8+ • scikit-learn 1.3 • xgboost 2.0 • pytorch 2.0</span>
        <span>{models.reduce((acc, c) => acc + c.items.length, 0)} models • {models.length} categories</span>
      </div>
    </div>
  );
}
