# Step 1: Upgrade to Python 3.11 to satisfy modern NumPy requirements
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /code

# Install required system libraries for YOLOv11/OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy the requirements file into the container first
COPY ./requirements.txt /code/requirements.txt

# Step 4: Install the Python dependencies (keeping the extended timeout active)
RUN pip install --no-cache-dir --default-timeout=1000 --upgrade -r /code/requirements.txt

# Step 5: Copy your root files and folders into the container
COPY . /code

# Step 6: Expose the port your app runs on
EXPOSE 8080

# Step 7: Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]