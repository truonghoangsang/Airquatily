from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Đọc dữ liệu và chuẩn bị mô hình
data = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
data.replace(-200, np.nan, inplace=True)
data.dropna(subset=['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S2(NMHC)', 'T', 'RH', 'PT08.S1(CO)'], inplace=True)

# Chọn các biến độc lập và biến phụ thuộc
X = data[['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S2(NMHC)', 'T', 'RH']]
y = data['PT08.S1(CO)']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tạo mô hình hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Tạo mô hình rừng ngẫu nhiên
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Tạo mô hình Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Lưu mô hình vào file
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(linear_model, 'models/linear_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(gb_model, 'models/gb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    if request.method == 'POST':
        try:
            input_data = [
                float(request.form['CO']),
                float(request.form['NMHC']),
                float(request.form['C6H6']),
                float(request.form['NOx']),
                float(request.form['NO2']),
                float(request.form['PT08.S2']),
                float(request.form['T']),
                float(request.form['RH'])
            ]
            if any(value < 0 for value in input_data):
                error_message = "Giá trị đầu vào không được âm."
                return render_template('index.html', prediction=prediction, error_message=error_message)
            
            # Chuyển đổi danh sách đầu vào thành DataFrame
            input_data_df = pd.DataFrame([input_data], columns=['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S2(NMHC)', 'T', 'RH'])
            
            # Chuẩn hóa dữ liệu đầu vào
            input_data_scaled = scaler.transform(input_data_df)
            model_type = request.form['model_type']
            if model_type == 'linear':
                prediction = linear_model.predict(input_data_scaled)[0]
            elif model_type == 'rf':
                prediction = rf_model.predict(input_data_scaled)[0]
            else:  # gb
                prediction = gb_model.predict(input_data_scaled)[0]

        except ValueError:
            error_message = "Giá trị đầu vào không hợp lệ."
    
    return render_template('index.html', prediction=prediction, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)