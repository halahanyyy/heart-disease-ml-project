import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib


columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
"thalach","exang","oldpeak","slope","ca","thal","num"]
files = [
"processed.cleveland.data",
"processed.hungarian.data",
"processed.switzerland.data",
"processed.va.data"
]
dfs = [pd.read_csv(f, header=None, names=columns,na_values="?") for f in files]
df = pd.concat(dfs, ignore_index=True)
#preprocessing
df=df.drop_duplicates()
df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())
df["chol"]=df["chol"].fillna(df["chol"].median())
df["thalach"]=df["thalach"].fillna(df["thalach"].median())
df["oldpeak"]=df["oldpeak"].fillna(df["oldpeak"].median())
df["thal"]=df["thal"].fillna(df["thal"].mode()[0])
df["fbs"]=df["fbs"].fillna(df["fbs"].mode()[0])
df["exang"]=df["exang"].fillna(df["exang"].mode()[0])
df["slope"]=df["slope"].fillna(df["slope"].mode()[0])
df["ca"]=df["ca"].fillna(df["ca"].mode()[0])
df["restecg"]=df["restecg"].fillna(df["restecg"].mode()[0])
categorical_cols = ["cp","restecg","slope","ca","thal"]
df2= pd.get_dummies(df, columns=categorical_cols, drop_first=True)
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df3=df2.copy()
scaler = StandardScaler()
df3[numeric_cols] = scaler.fit_transform(df3[numeric_cols])
#EDA
df3.hist(figsize=(15,10))
plt.suptitle("Histograms of Features", fontsize=16)  
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(df3.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()
plt.figure(figsize=(8,6))
sns.boxplot(x="num", y="age", data=df3)
plt.title("Boxplot of Age vs Heart Disease (num)", fontsize=16)
plt.show()
print(df3.head())
#PCA
X = df3.drop("num", axis=1)
y = df3["num"]
pca = PCA(n_components=0.95)  
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.grid()
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="plasma", edgecolor="k", s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Scatter Plot (First 2 Components)")
plt.colorbar(label="num")
plt.show()
#feature_selection
#random_forest
X = df3.drop("num", axis=1)
y = df3["num"].apply(lambda x: 1 if x > 0 else 0)  
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Random Forest (Feature Importance)")
plt.show()
#recursive_feature_elimination
lr = LogisticRegression(max_iter=1000, solver="liblinear")
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("Selected Features by RFE:", list(selected_features_rfe))
#chi-square_test
X_chi = df2.drop("num", axis=1)
y_chi = df2["num"].apply(lambda x: 1 if x > 0 else 0)
scaler = MinMaxScaler()
X_chi_scaled = scaler.fit_transform(X_chi)
chi2_selector = SelectKBest(score_func=chi2, k=10)
X_kbest = chi2_selector.fit_transform(X_chi_scaled, y_chi)
selected_features_chi2 = X_chi.columns[chi2_selector.get_support()]
print("Selected Features by Chi-Square:", list(selected_features_chi2))
#selection
rf_features = X.columns[indices[:10]]
rfe_features = set(selected_features_rfe)
chi2_features = set(selected_features_chi2)
intersection = set(rf_features).intersection(rfe_features, chi2_features)
print("Intersection (common features):", intersection)
selected_final_features = rf_features  
print("The selected features:", list(selected_final_features))

#supervisedlearning
X_final = X[selected_final_features]
y_final = y

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}
plt.figure(figsize=(8,6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = [acc, prec, rec, f1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

print("Model Performance:\n")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Accuracy : {metrics[0]:.2f}")
    print(f"  Precision: {metrics[1]:.2f}")
    print(f"  Recall   : {metrics[2]:.2f}")
    print(f"  F1-score : {metrics[3]:.2f}")
    print("-"*40)

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#unsupervisedlearning
X_cluster = X_final  
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for KMeans")
plt.show()
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)
pca = PCA(n_components=2)
X_pca_cluster = pca.fit_transform(X_cluster)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca_cluster[:,0], y=X_pca_cluster[:,1], hue=y_kmeans, palette="viridis")
plt.title("KMeans Clustering (PCA 2D)")
plt.show()

plt.figure(figsize=(10,7))
dendrogram = sch.dendrogram(sch.linkage(X_cluster, method='ward'))
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Samples")
plt.ylabel("Euclidean distances")
plt.show()

hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_cluster)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca_cluster[:,0], y=X_pca_cluster[:,1], hue=y_hc, palette="coolwarm")
plt.title("Hierarchical Clustering (PCA 2D)")
plt.show()
print("KMeans vs True Labels")
print(confusion_matrix(y_final, y_kmeans))
print("Accuracy:", accuracy_score(y_final, y_kmeans))
print("\nHierarchical vs True Labels")
print(confusion_matrix(y_final, y_hc))
print("Accuracy:", accuracy_score(y_final, y_hc))

#tuning
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"]
    }
}
best_models = {}
for name, model in models.items():
    print(f"🔍 Tuning {name} ...")
    param_grid = param_grids[name]
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"Best params for {name}: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.3f}")
    print("-"*50)
    print("\n📊 Performance After Tuning:\n")
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  Accuracy : {acc:.2f}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall   : {rec:.2f}")
    print(f"  F1-score : {f1:.2f}")
    print("-"*40)

#model_export
final_model = best_models["SVM"]  
joblib.dump(final_model, "final_model.pkl")
loaded_model = joblib.load("final_model.pkl")
y_pred = loaded_model.predict(X_test)
print("Accuracy after loading:", accuracy_score(y_test, y_pred))
