import requests

url = "http://127.0.0.1:8000/predict"
headers = {"X-API-Key": "guardian123", "Content-Type": "application/json"}
data = {"text": "كيف الحال"}

response = requests.post(url, json=data, headers=headers)
print(response.json())