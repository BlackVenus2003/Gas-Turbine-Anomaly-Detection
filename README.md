\## 🛠️ Engine Health Monitoring – Anomaly Detection on Gas Turbine Data



This project performs \*\*anomaly detection\*\* on gas turbine engine sensor data using a combination of statistical and machine learning techniques. The goal is to flag early signs of performance degradation, faults, or abnormal operation using:



\- Z-score thresholds

\- Isolation Forest (unsupervised anomaly detection)

\- Regression-based residual analysis



---



\## 📁 Project Structure



Engine\_Health\_Anomaly/

├── data/

│ └── gas\_turbine.csv # Input dataset (UCI or similar)

├── src/

│ └── detect\_anomalies.py # Main analysis script

├── output/

│ ├── anomaly\_report.csv # Output report with anomaly flags

│ ├── TIT\_anomalies.png # Visual of TIT over time

│ └── TEY\_residuals.png # Histogram of regression residuals

├── requirements.txt

└── README.md



---



\## 📊 Features Used



The script works on the following parameters if available:

\- TIT – Turbine Inlet Temperature

\- TAT – Turbine After Temperature

\- TEY – Turbine Energy Yield

\- CDP – Compressor Pressure Ratio

\- CO  – Carbon Monoxide

\- NOx – Nitric Oxides

\- Ambient parameters: AT (temp), AP (pressure), RH (humidity)



---



\## ⚙️ How It Works



1\. \*\*Z-Score Analysis\*\*  

&nbsp;  Flags outliers in any key sensor where value exceeds ±3 standard deviations.



2\. \*\*Isolation Forest\*\*  

&nbsp;  Uses unsupervised ML to detect rare combinations of sensor readings.



3\. \*\*Residual Analysis\*\*  

&nbsp;  Trains a regression model to predict `TEY` based on ambient and compressor inputs. High residuals = potential fault.



---



\## ▶️ Run Instructions



```bash

\# Install dependencies

pip install -r requirements.txt



\# Run the anomaly detection pipeline

python src/detect\_anomalies.py

Output files will be saved to the output/ directory.
```


📈 Output

anomaly\_report.csv: Combined anomaly flags (z\_flag, iso\_flag, res\_flag, final anomaly)



TIT\_anomalies.png: Shows flagged points on TIT trend



TEY\_residuals.png: Histogram of TEY prediction errors



🧠 Skills Demonstrated

Python \& Pandas data handling



Feature selection \& sensor correlation



Unsupervised ML (Isolation Forest)



Residual/Error analysis using regression



Data visualization with Seaborn \& Matplotlib



📚 Dataset Reference

Dataset: https://www.kaggle.com/datasets/muniryadi/gasturbine-co-and-nox-emission-data

You may also use synthetic or internal turbine datasets with similar sensor columns.



📌 Author

Paramjyot G. K. Tiwana

Feel free to fork, clone, or extend this project!

