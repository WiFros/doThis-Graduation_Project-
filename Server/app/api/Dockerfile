# Select python version
FROM python:3.9.13


# Copy requirements.txt for installation
COPY requirements.txt .

# Install 'requirements.txt'
RUN pip install --upgrade pip
RUN apt update; apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

# Copy all of docker image contents
COPY . .

# Open port 5000
EXPOSE 5000

# Start Flask Server (app.py)
CMD ["python","app.py"]