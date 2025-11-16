# 1️⃣ Base image with Python
FROM python:3.13-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and project files
COPY requirements.txt .
COPY . .

# 4️⃣ Install system dependencies for Playwright
RUN apt-get update && \
    apt-get install -y curl wget gnupg && \
    apt-get install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 libgtk-3-0 libx11-xcb1 libxss1 libxtst6 libxkbcommon0 fonts-liberation lsb-release && \
    rm -rf /var/lib/apt/lists/*

# 5️⃣ Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 6️⃣ Install Node.js (required by Playwright)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# 7️⃣ Install Playwright browsers
RUN npx playwright install --with-deps

# 8️⃣ Expose port (Flask default 5000)
EXPOSE 5000

# 9️⃣ Start command
CMD ["python", "app.py"]
