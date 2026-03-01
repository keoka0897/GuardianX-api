from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from ai_threat_analyzer import ThreatAnalyzer
import os
import time
import logging
from fastapi_advanced_rate_limiter import SlidingWindowRateLimiter

# إعداد التسجيل (Logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("guardianx")

app = FastAPI(docs_url=None, redoc_url=None)

# قراءة مفتاح API من المتغيرات البيئية
API_KEY = os.getenv("API_KEY", "guardian123")

# ✅ إضافة Rate Limiter - 10 طلبات في الدقيقة لكل مستخدم
limiter = SlidingWindowRateLimiter(
    capacity=10,      # 10 طلبات كحد أقصى
    fill_rate=10/60,  # لكل 60 ثانية
    scope="user",     # لكل مستخدم (بناءً على المفتاح)
    backend="memory"  # للتجربة المحلية
)

analyzer = ThreatAnalyzer()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "GuardianX API is running"}

@app.post("/predict")
async def predict(request: Request, data: TextRequest, x_api_key: str = Header(None)):
    # 1. التحقق من وجود المفتاح
    if not x_api_key:
        logger.warning("طلب بدون مفتاح API")
        raise HTTPException(status_code=401, detail="API Key مفقود")
    
    # 2. التحقق من صحة المفتاح
    if x_api_key != API_KEY:
        logger.warning(f"محاولة بمفتاح غير صالح: {x_api_key[:5]}...")
        raise HTTPException(status_code=403, detail="مفتاح غير صالح")
    
    # ✅ 3. التحقق من عدد الطلبات (Rate Limiting)
    client_id = x_api_key
    if not limiter.allow_request(client_id):
        wait_time = limiter.get_wait_time(client_id)
        logger.warning(f"كثرة طلبات من المستخدم: {x_api_key[:5]}...")
        raise HTTPException(
            status_code=429,
            detail=f"عدد الطلبات كبير جداً. حاول بعد {wait_time:.0f} ثانية"
        )
    
    # 4. تنفيذ التحليل
    start_time = time.time()
    try:
        result = analyzer.predict(data.text)
        processing_time = time.time() - start_time
        
        # 5. تسجيل الطلب الناجح
        logger.info(f"مستخدم: {x_api_key[:5]}... | نص: {data.text[:30]}... | نتيجة: {result} | وقت: {processing_time:.2f}ث")
        
        return {"result": result}
    
    except Exception as e:
        logger.error(f"خطأ في التحليل: {str(e)}")
        raise HTTPException(status_code=500, detail="خطأ داخلي في السيرفر")

# للتشغيل المحلي
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)