import pandas as pd
import datetime

class NJCleaner:

    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
    
    def order_by_scheduled_time(self) -> pd.core.frame.DataFrame:
        self.data.sort_values(['scheduled_time'], inplace=True)
        return self.data
    
    def drop_columns_and_nan(self) -> pd.core.frame.DataFrame:
        self.data.drop(['from', 'to'], axis=1, inplace=True)
        self.data.dropna(inplace=True)
        return self.data
    
    def convert_date_to_day(self) -> pd.core.frame.DataFrame:
        self.data['day'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d').dt.day_name()
        self.data.drop(['date'], axis=1, inplace=True)
        return self.data
    
    def convert_scheduled_time_to_part_of_the_day(self) -> pd.core.frame.DataFrame:
        self.data['part_of_the_day'] = pd.to_datetime(self.data['scheduled_time']).apply(lambda timestamp: 
                                                                                         'late_night' if timestamp.time() < datetime.time(4, 00) else 
                                                                                         'early_morning' if timestamp.time() < datetime.time(8, 00) else 
                                                                                         'morning' if timestamp.time() < datetime.time(12, 00) else 
                                                                                         'afternoon' if timestamp.time() < datetime.time(16, 00) else 
                                                                                         'evening' if timestamp.time() < datetime.time(20, 00) else 
                                                                                         'night')
        self.data.drop(['scheduled_time'], axis = 1, inplace=True)
        return self.data
    
    def convert_delay(self) -> pd.core.frame.DataFrame:
        self.data['delay'] = self.data['delay_minutes'].apply(lambda delay_minute: 0 if delay_minute >=0 and delay_minute < 5 else 1)
        return self.data
    
    def drop_unnecessary_columns(self) -> pd.core.frame.DataFrame:
        self.data.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1, inplace=True)
        return self.data
    
    def save_first_60k(self, path: str):
        self.data.head(60000).to_csv(path, index=False)

    def prep_df(self, path: str = 'data/NJ.csv'):
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        self.save_first_60k(path)


# cleaning = NJCleaner('data/2018_03.csv')
# cleaning.order_by_scheduled_time()
# cleaning.drop_columns_and_nan()
# cleaning.convert_date_to_day()
# print(cleaning.convert_scheduled_time_to_part_of_the_day())
#print(cleaning.convert_delay())
# cleaning.drop_unnecessary_columns()
# cleaning.save_first_60k('data/NJ.csv')

# cleaning.prep_df()

# print(cleaning.data)
