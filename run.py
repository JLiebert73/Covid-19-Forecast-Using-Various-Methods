from data_preprocessing import preprocess_data
from XGB import train_model
from create_submission import create_submission
import matplotlib.pyplot as plt

train_path = r'Data\\train.csv'
test_path = r'Data\\test.csv'

x_train, x_test = preprocess_data(train_path, test_path)

country_list = x_train['Country_Region'].unique()

plots = train_model(x_train, x_test, country_list)

for plot_data in plots:
    country = plot_data['country']
    province = plot_data['province']

    # Plotting for Confirmed Cases
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data['X_train'], plot_data['Y_train_c'], label='Actual Confirmed Cases', marker='o')
    plt.plot(x_test.loc[(x_test['Country_Region'] == country) & (x_test['Province_State'] == province), ['Date']].astype('int'), plot_data['Y_pred_c'], label='Predicted Confirmed Cases', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.title(f'{country} - {province} - Confirmed Cases Prediction')
    plt.legend()
    plt.show()

    # Plotting for Fatalities
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data['X_train'], plot_data['Y_train_f'], label='Actual Fatalities', marker='o')
    plt.plot(x_test.loc[(x_test['Country_Region'] == country) & (x_test['Province_State'] == province), ['Date']].astype('int'), plot_data['Y_pred_f'], label='Predicted Fatalities', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Fatalities')
    plt.title(f'{country} - {province} - Fatalities Prediction')
    plt.legend()
    plt.show()

# Create submission
sub = create_submission(plots)
