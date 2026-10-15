# 💊 Getz Pharma Customer Segmentation

## 📌 Project Overview

This project uses **Unsupervised Machine Learning** to identify different customer groups based on their purchasing behaviour. The goal is to discover patterns in customer data using the K-Means Clustering algorithm without predefined labels.

## 🎯 Objectives

- Explore and understand the customer dataset.
- Clean and prepare the data for machine learning.
- Use the Elbow Method to help select the number of clusters.
- Apply K-Means Clustering to group similar customers.
- Visualize customer segments and understand their behaviour.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## 📊 Features

- Customer dataset overview and summary statistics.
- Missing value analysis.
- Interactive Elbow Method chart.
- Adjustable number of customer clusters.
- Customer segmentation visualizations.
- Cluster summaries to compare customer behaviour.
- Individual customer filtering by segment.

## 🤖 Machine Learning Algorithm

**K-Means Clustering**

K-Means is an unsupervised learning algorithm that groups similar data points into clusters. In this project, customer purchasing and payment behaviour is used to identify different segments.

The data is scaled using `StandardScaler` before applying the algorithm. The Elbow Method is used to help determine a suitable number of clusters.

## 📁 Project Structure

```text
getz-pharma-customer-segmentation/
│
├── pharma_revised.py
├── getz_pharma_customer_segmentation_1000.csv
└── README.md
```

## 🚀 How to Run the Project

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/getz-pharma-customer-segmentation.git
```

**2. Open the project folder**

```bash
cd getz-pharma-customer-segmentation
```

**3. Install the required libraries**

```bash
pip install streamlit pandas scikit-learn matplotlib
```

**4. Run the application**

```bash
streamlit run pharma_revised.py
```

The application will open in your browser.

## 💡 Learning Outcomes

Through this project, I practiced data exploration, feature scaling, K-Means clustering, and data visualization. I also learned how unsupervised learning can help discover customer behaviour patterns and support data-driven business decisions.

## 👨‍💻 Author

**Rafay Hasan**

Aspiring Python Developer | Machine Learning Learner

