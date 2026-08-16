import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score
import joblib, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROLE_TO_CATEGORY = {
    "Applications Developer": "Software Development",
    "CRM Technical Developer": "Software Development",
    "Database Developer": "Software Development",
    "Mobile Applications Developer": "Software Development",
    "Software Developer": "Software Development",
    "Software Engineer": "Software Development",
    "Web Developer": "Software Development",
    "Network Security Engineer": "Cybersecurity",
    "Systems Security Administrator": "Cybersecurity",
    "Software Quality Assurance (QA) / Testing": "QA & Support",
    "Technical Support": "QA & Support",
    "UX Designer": "Design",
}

class CareerModelWrapper:
    def __init__(self, preprocessor, model, label_encoder):
        self.preprocessor = preprocessor
        self.model = model
        self.label_encoder = label_encoder
    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        preds = self.model.predict(X_proc)
        return self.label_encoder.inverse_transform(preds)

df = pd.read_csv("data/PS2_Dataset.csv")
X = df.drop(columns=["Suggested Job Role"])
y_raw = df["Suggested Job Role"].map(ROLE_TO_CATEGORY)

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_cols)
])

X_train_proc = preprocessor.fit_transform(X_train_rus)
X_test_proc = preprocessor.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_proc, y_train_rus)

y_pred = rf.predict(X_test_proc)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

final = CareerModelWrapper(preprocessor=preprocessor, model=rf, label_encoder=le)
joblib.dump(final, "models/career_model_grouped.pkl")
print("Saved.")
