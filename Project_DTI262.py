import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo

# โหลดข้อมูลจาก UCI
maternal_health_risk = fetch_ucirepo(id=863)
df = maternal_health_risk.data.original

# ตัดแถวข้อมูลที่ HeartRate เท่ากับ 7 (ผิดปกติ)
df = df[df["HeartRate"] != 7]

# ตัดแถวข้อมูลมีมีอายุต่ำกว่า 12 และมากกว่า 50 ปี
df = df[(df['Age'] >= 12) & (df['Age'] <= 50)]

# แสดงข้อมูลสรุปของ DataFrame เพื่อตรวจสอบชนิดข้อมูลและจำนวนค่าที่ไม่ใช่ค่าว่างในแต่ละคอลัมน์
print(df.info())

# แสดงค่าสถิติพื้นฐานของคอลัมน์เชิงตัวเลข เช่น ค่าเฉลี่ย, ค่าสูงสุด, ต่ำสุด และส่วนเบี่ยงเบนมาตรฐาน
print(df.describe())

# แยก features และ target
X = df.drop(columns=["RiskLevel"])
y = df["RiskLevel"]

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ฟังก์ชันประเมินผลของการทดลอง
def evaluate_model(n_estimators, max_features):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_list = []
    f1_macro_list = []
    all_f1 = []
    all_precision = []
    all_recall = []
    all_y_true = []
    all_y_pred = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # สร้างโมเดล
        if (n_estimators != None) and (max_features == None):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42
                )
        elif (n_estimators != None) and (max_features != None):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=42
                )
        else: model = RandomForestClassifier(random_state=42)

        # เทรน
        model.fit(X_train, y_train)

        # ทำนายทั้ง train และ test
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # แสดง classification report ของ training
        print(f"Fold {fold} Classification Report (Train):")
        report_train = classification_report(
            y_train, y_pred_train,
            target_names=label_encoder.classes_,
            labels=model.classes_,
            digits=4
        )
        print(report_train)

        # แสดง classification report ของ testing
        print(f"Fold {fold} Classification Report (Test):")
        report_test = classification_report(
            y_test, y_pred_test,
            target_names=label_encoder.classes_,
            labels=model.classes_,
            digits=4
        )
        print(report_test)

        # เก็บผลลัพธ์รวม
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_test)

        # คำนวณ metrics ต่าง ๆ จาก test
        acc = accuracy_score(y_test, y_pred_test)
        f1_macro = f1_score(y_test, y_pred_test, average="macro")
        f1_each = f1_score(y_test, y_pred_test, average=None)
        precision_each = precision_score(y_test, y_pred_test, average=None)
        recall_each = recall_score(y_test, y_pred_test, average=None)

        accuracy_list.append(acc)
        f1_macro_list.append(f1_macro)
        all_f1.append(f1_each)
        all_precision.append(precision_each)
        all_recall.append(recall_each)

        fold += 1

    # แสดงผลรวมของทุก fold
    print("\nAll Classification Report (Test):")
    report_all = classification_report(all_y_true, all_y_pred, target_names=label_encoder.classes_, digits=4)
    print(report_all)
    avg_accuracy = np.mean(accuracy_list)
    return avg_accuracy

# โมเดลเริ่มต้น
print("[ Baseline ]")
print(evaluate_model(None, None))

# การทดลองที่ 1 ทอสอบค่าพารามิเตอรื n_estimators
print("\n[ Experiment 1 ]")
print("\n[ n_estimators: 110 ]")
ex1_1 = evaluate_model(110, None)
print("\n[ n_estimators: 251 ]")
ex1_2 = evaluate_model(251, None)
print("\n[ n_estimators: 501 ]")
ex1_3 = evaluate_model(501, None)

# สร้าง dictionary สำหรับเก็บค่า accuracy ของแต่ละ Activation
n_estimators_results = {
    "110": ex1_1,
    "251": ex1_2,
    "501": ex1_3
}
# แล้วเลือกโครงสร้างที่มีค่า accuracy สูงที่สุด
best_n_estimators = max(n_estimators_results, key=n_estimators_results.get)
print(f"\nBest n_estimators: {best_n_estimators}")

# แปลงค่าตัวแปรจาก str เป็น int เพื่อนำไปใช้ต่อในการทดลองที่ 2
best_n_estimators = int(best_n_estimators)

# การทดลองที่ 2 ทอสอบค่าพารามิเตอรื max_features
print("\n[ Experiment 2 ]")
print("\n[ max_features: 0.7 ]")
ex2_1 = evaluate_model(best_n_estimators, 0.7)
print("\n[ max_features: 0.9 ]")
ex2_2 = evaluate_model(best_n_estimators, 0.9)
print("\n[ max_features: 'log2' ]")
ex2_3 = evaluate_model(best_n_estimators, 'log2')

# สร้าง dictionary สำหรับเก็บค่า accuracy ของแต่ละ Activation
max_features_results = {
    "0.7": ex2_1,
    "0.9": ex2_2,
    "log2": ex2_3
}
# แล้วเลือกโครงสร้างที่มีค่า accuracy สูงที่สุด
best_max_features = max(max_features_results, key=max_features_results.get)
print(f"\nBest max_features: {best_max_features}")

print(f"\nBest method:\nn_estimators = {best_n_estimators}\nmax_features = {best_max_features}")
