import pandas as pd

def create_submission(sub):
    submission = pd.DataFrame(sub)
    submission[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv(path_or_buf='submission.csv', index=False)
