FROM python

WORKDIR /app

# Copy the requirements.txt file into the image
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]