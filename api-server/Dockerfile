# Use an official Python image as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# --- venv setup ---
# 1. Set environment variables for the virtual environment
#    Using ENV makes it easy to reference these paths later
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 2. Create the virtual environment
RUN python3 -m venv $VIRTUAL_ENV

# --- Dependency Installation ---
# 3. Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# 4. Install dependencies into the virtual environment
#    Because we updated the PATH, 'pip' now refers to the pip inside the venv.
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# 5. Copy the application code into the container
COPY ./app /app/app

# --- Run Command ---
# 6. Command to run the application
#    This 'uvicorn' is the one installed inside the venv because of the PATH.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5501", "--reload"]