from fastapi import FastAPI #FastAPI → creates APIs
from fastapi.responses import HTMLResponse #HTMLResponse → helps return HTML web pages (like index.html)
import joblib #joblib → loads your saved ML model (model.pkl)
import uvicorn #uvicorn → runs your FastAPI app
import logging #logging → used to store user activity requests inside a log file

# ----------------------------------------------------
#  Configure Logging (Stores logs in requests.log)
# ----------------------------------------------------
logging.basicConfig(
    filename="requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

model = joblib.load(r"D:\end to end python\End to end CICD pipeline\source\model.pkl")

@app.get("/")
def home():
    logging.info("Home page accessed")
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/predict/{Experience}")
def salary_predict(Experience: float):
    # Log incoming request
    logging.info(f"Prediction request received | Experience: {Experience}")

    # Predict
    salary = model.predict([[Experience]])[0]

    # Log output
    logging.info(f"Prediction completed | Experience: {Experience}, Salary: {salary}")

    return {"experience": Experience, "predicted_salary": salary}

if __name__ == "__main__":
    uvicorn.run("prediction:app", host="127.0.0.1", port=8000, reload=True) #reload=True → auto reload on code change
