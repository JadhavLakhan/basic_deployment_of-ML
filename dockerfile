FROM python:3.11

# Create working directory
WORKDIR /apps

# Copy requirement file
COPY requirement.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

# Copy all project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "source.prediction:app", "--host", "0.0.0.0", "--port", "8000"]
