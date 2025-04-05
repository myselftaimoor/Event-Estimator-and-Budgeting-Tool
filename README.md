# Event Estimator and Budgeting Tool

This repository contains the implementation of an **Event Estimator and Budgeting Tool** designed to predict event costs based on key parameters such as guest count, venue, catering, and additional services. The tool leverages machine learning to provide accurate cost predictions, enabling event planners to make informed decisions and manage budgets effectively.

---

## ⭐ Features

- **Cost Prediction**: Accurately estimates event costs based on user inputs.
- **Parameters**: Takes inputs such as guest count, venue, catering, and additional services.
- **Machine Learning Models**: Trained models to provide reliable cost estimates.
- **User-Friendly Interface**: Simple interface for easy use by event planners.

---

## 🧠 Methodology

1. **Data Collection**: Gathers event data including guest count, venue type, catering services, and additional event details.
2. **Model Training**: Various machine learning models are trained using this data to predict the total event cost.
3. **Cost Estimation**: Users input event details (guest count, venue, services), and the tool estimates the total cost based on trained models.
4. **Evaluation**: The tool evaluates the cost prediction accuracy to ensure reliability for event planners.

---

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/event-estimator-tool.git
   cd event-estimator-tool

2. **Create & Activate Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the Application**
   ```bash
   python app.py

5. Access the Tool

Open your browser and go to:
http://127.0.0.1:5000/

## 🛠️ Tech Stack

Backend: Python, Flask
Machine Learning: Scikit-learn (for model training)

## 📂 Directory Structure

app.py: Main application file

models/: Contains trained machine learning models

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements or features. If you have suggestions or bug reports, please open an issue.

## 📄 License

This project is licensed under the MIT License.



