# =========================================
# GuardianX AI Threat Analyzer
# Day 3 – Final Stable Version (Modified)
# with retrain() function
# =========================================

import re
import pickle
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression


# ==============================
# إعدادات المشروع
# ==============================

DATA_FILE = "dataset.csv"
MODEL_FILE = "guardian_model.pkl"

LABELS = {
    "safe": 0,
    "scam": 1,
    "threat": 2
}


# ==============================
# الكلاس الرئيسي
# ==============================

class ThreatAnalyzer:

    # ---------------------------------
    # INIT
    # ---------------------------------
    def __init__(self):

        print("Loading fastText model...")

        self.vectors = KeyedVectors.load_word2vec_format(
            r"D:\مشروع\cc.ar.300.vec",   # عدّلي المسار فقط لو مختلف
            binary=False
        )

        print("fastText loaded ✔")

        self.model = LogisticRegression(max_iter=1000)

        # لو في موديل محفوظ → حمليه
        try:
            with open(MODEL_FILE, "rb") as f:
                self.model = pickle.load(f)
                print("Model loaded from file ✔")

        # لو ما في → درّبي
        except:
            print("Training new model...")
            self.train()


    # ---------------------------------
    # تحويل النص إلى متجه
    # ---------------------------------
    def text_to_vector(self, text):

        words = re.findall(r"\w+", str(text))

        vecs = [self.vectors[w] for w in words if w in self.vectors]

        if not vecs:
            return np.zeros(self.vectors.vector_size)

        return np.mean(vecs, axis=0)


    # ---------------------------------
    # تدريب النموذج (نسخة آمنة)
    # ---------------------------------
    def train(self):

        df = pd.read_csv(DATA_FILE)

        X = []
        y = []

        for text, label in zip(df["text"], df["label"]):

            # تنظيف label (يمنع KeyError) + تحويل لحروف صغيرة
            label = str(label).strip().lower()

            if label not in LABELS:
                print(f"⚠ Skipped unknown label: {label}")
                continue

            X.append(self.text_to_vector(text))
            y.append(LABELS[label])

        X = np.array(X)
        y = np.array(y)

        self.model.fit(X, y)

        print("Training complete ✔")

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)

        print("Model saved ✔")


    # ---------------------------------
    # دالة إعادة التدريب (الجديدة)
    # ---------------------------------
    def retrain(self, new_file=None):
        """
        إعادة تدريب النموذج على كل البيانات
        new_file: (اختياري) ملف CSV إضافي فيه جمل جديدة
        """
        
        print("🔄 جاري إعادة التدريب على كل البيانات...")
        
        # قراءة البيانات الأساسية
        df = pd.read_csv(DATA_FILE)
        
        # لو في ملف إضافي، ندمجه مع الأساسي
        if new_file:
            df_new = pd.read_csv(new_file)
            df = pd.concat([df, df_new], ignore_index=True)
            # (اختياري) نحفظ الملف المدمج عشان نستفيد منه بعدين
            df.to_csv(DATA_FILE, index=False)
            print(f"📁 تم دمج ملف {new_file} مع البيانات الأساسية")
        
        X = []
        y = []
        skipped = 0
        
        for text, label in zip(df["text"], df["label"]):
            label = str(label).strip().lower()
            
            if label not in LABELS:
                skipped += 1
                continue
                
            X.append(self.text_to_vector(text))
            y.append(LABELS[label])
        
        X = np.array(X)
        y = np.array(y)
        
        # تدريب النموذج
        self.model.fit(X, y)
        
        # حفظ النموذج الجديد
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        
        # طباعة تقرير
        print(f"✅ تم إعادة التدريب بنجاح!")
        print(f"📊 إجمالي الجمل: {len(df)}")
        print(f"📊 جمل مستخدمة في التدريب: {len(y)}")
        if skipped > 0:
            print(f"⚠ جمل تم تخطيها (تسمية غير معروفة): {skipped}")
        print(f"📊 توزيع التصنيفات:")
        for label, count in df['label'].value_counts().items():
            print(f"   • {label}: {count}")


    # ---------------------------------
    # التنبؤ
    # ---------------------------------
    def predict(self, text):

        vec = self.text_to_vector(text).reshape(1, -1)

        pred = self.model.predict(vec)[0]

        # عكس القاموس
        inv = {v: k for k, v in LABELS.items()}

        return inv[pred]


# ==============================
# اختبار مباشر
# ==============================

if __name__ == "__main__":

    analyzer = ThreatAnalyzer()

    # ----------------------------
    # مثال: إعادة التدريب (لو تبغى)
    # ----------------------------
    
    # analyzer.retrain()  # أزل التعليق لو تبي تعيد التدريب فوراً
    
    # analyzer.retrain("جمل_جديدة.csv")  # لو عندك ملف إضافي
    
    # ----------------------------
    # اختبار التنبؤ
    # ----------------------------
    
    tests = [
        "مرحبا كيف حالك",
        "ارسل المال والا بفضحك",
        "فزت بجائزة اضغط الرابط",
        "هات الرقم السري الآن",
        "وينك يا صاحبي"
    ]

    print("\n🔍 اختبار التنبؤ:")
    for t in tests:
        print(f"  • {t} ➡ {analyzer.predict(t)}")