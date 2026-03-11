# =========================================
# GuardianX AI Threat Analyzer
# Version 2.0 - مع AraBERT (500MB)
# =========================================

import re
import pickle
import numpy as np
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression


# ==============================
# إعدادات المشروع
# ==============================

DATA_FILE = "dataset.csv"
MODEL_FILE = "guardian_model.pkl"
ARABERT_PATH = "./arabic_model"  # المسار المحلي للنموذج

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
    # INIT - مع AraBERT
    # ---------------------------------
    def __init__(self):

        print("="*50)
        print("🚀 جاري تحميل نموذج AraBERT...")
        print("="*50)

        # التحقق من وجود النموذج المحفوظ
        if not os.path.exists(ARABERT_PATH):
            print("⚠ لم أجد النموذج المحفوظ. سيتم تحميله من الإنترنت (قد يستغرق دقائق)")
            # إذا لم يكن موجوداً، نحمله من Hugging Face
            model_name = "aubmindlab/bert-base-arabertv2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            # نحفظه للمستقبل
            os.makedirs(ARABERT_PATH, exist_ok=True)
            self.tokenizer.save_pretrained(ARABERT_PATH)
            self.bert_model.save_pretrained(ARABERT_PATH)
            print("✅ تم تحميل وحفظ النموذج محلياً")
        else:
            # تحميل النموذج من المسار المحلي
            print(f"📂 تحميل النموذج من: {ARABERT_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(ARABERT_PATH)
            self.bert_model = AutoModel.from_pretrained(ARABERT_PATH)
            print("✅ تم تحميل النموذج المحلي بنجاح!")

        # تحديد الجهاز (GPU إن وجد)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
        print(f"💻 الجهاز المستخدم: {self.device}")

        # تحميل نموذج التصنيف (Logistic Regression)
        self.classifier = LogisticRegression(max_iter=1000)
        
        # محاولة تحميل النموذج المدرب مسبقاً
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                self.classifier = pickle.load(f)
                print("✅ تم تحميل نموذج التصنيف من ملف")
        else:
            print("🔄 لا يوجد نموذج تصنيف محفوظ. سيتم تدريب نموذج جديد...")
            self.train()


    # ---------------------------------
    # تحويل النص إلى متجه باستخدام AraBERT
    # ---------------------------------
    def text_to_vector(self, text):
        """
        تحويل النص إلى متجه (768 بعداً) باستخدام AraBERT
        """
        # تجهيز النص
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)

        # تمرير النص عبر النموذج
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # استخدام متوسط التشفيرات (Mean Pooling)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding


    # ---------------------------------
    # تدريب النموذج
    # ---------------------------------
    def train(self):

        print("📚 جاري قراءة بيانات التدريب...")
        df = pd.read_csv(DATA_FILE)

        X = []
        y = []
        skipped = 0

        for idx, (text, label) in enumerate(zip(df["text"], df["label"])):
            # عرض التقدم كل 100 جملة
            if idx % 100 == 0:
                print(f"⏳ معالجة الجملة {idx}/{len(df)}")

            label = str(label).strip().lower()

            if label not in LABELS:
                skipped += 1
                continue

            # تحويل النص إلى متجه
            vec = self.text_to_vector(text)
            X.append(vec)
            y.append(LABELS[label])

        X = np.array(X)
        y = np.array(y)

        print(f"\n📊 إجمالي الجمل: {len(df)}")
        print(f"📊 جمل مستخدمة في التدريب: {len(y)}")
        print(f"⚠ جمل تم تخطيها: {skipped}")
        print(f"📊 شكل مصفوفة التدريب: {X.shape}")

        # تدريب النموذج
        print("\n🧠 جاري تدريب النموذج...")
        self.classifier.fit(X, y)
        print("✅ تم التدريب بنجاح!")

        # حفظ النموذج
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.classifier, f)
        print("💾 تم حفظ النموذج في ملف")


    # ---------------------------------
    # التنبؤ
    # ---------------------------------
    def predict(self, text):

        # تحويل النص إلى متجه
        vec = self.text_to_vector(text).reshape(1, -1)

        # التنبؤ
        pred = self.classifier.predict(vec)[0]

        # تحويل الرقم إلى تسمية
        inv = {v: k for k, v in LABELS.items()}
        
        return inv[pred]


    # ---------------------------------
    # إعادة التدريب (اختياري)
    # ---------------------------------
    def retrain(self, new_file=None):
        """
        إعادة تدريب النموذج على كل البيانات
        new_file: (اختياري) ملف CSV إضافي
        """
        print("\n🔄 جاري إعادة التدريب...")
        
        df = pd.read_csv(DATA_FILE)
        
        if new_file:
            df_new = pd.read_csv(new_file)
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            print(f"📁 تم دمج ملف {new_file}")
        
        X = []
        y = []
        skipped = 0
        
        for text, label in zip(df["text"], df["label"]):
            label = str(label).strip().lower()
            
            if label not in LABELS:
                skipped += 1
                continue
                
            vec = self.text_to_vector(text)
            X.append(vec)
            y.append(LABELS[label])
        
        X = np.array(X)
        y = np.array(y)
        
        self.classifier.fit(X, y)
        
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.classifier, f)
        
        print(f"✅ تم إعادة التدريب!")
        print(f"📊 الجمل المستخدمة: {len(y)}")


# ==============================
# اختبار مباشر
# ==============================

if __name__ == "__main__":

    analyzer = ThreatAnalyzer()
    
    tests = [
        "مرحبا كيف حالك",
        "ارسل المال والا بفضحك",
        "فزت بجائزة اضغط الرابط",
        "هات الرقم السري الآن",
        "وينك يا صاحبي"
    ]

    print("\n🔍 اختبار التنبؤ:")
    for t in tests:
        result = analyzer.predict(t)
        print(f"  • {t} ➡ {result}")