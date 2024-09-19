import pandas as pd
from rapidfuzz import process, fuzz

loc_delay_df = pd.read_excel('bus_delay.xlsx')
loc_df = pd.read_csv('gtfs/stops.txt')


loc_delay = loc_delay_df['Location'].unique()
loc = loc_df['stop_name'].unique()


threshold = 80
common_stops_loc_delay = []
common_stops_loc = []

for stop in loc_delay:
    best_match = process.extract(stop, loc, scorer=fuzz.ratio, limit=1)

    if best_match:
        match_name, score, match_idx = best_match[0]


        if score >= threshold:
            common_stops_loc_delay.append(stop)
            common_stops_loc.append(match_name)

common_stops_df = pd.DataFrame({
    'Location': common_stops_loc_delay,  # from bus_delay
    'stop_name': common_stops_loc  # from stops
})

filtered_loc_delay_df = loc_delay_df[loc_delay_df['Location'].isin(common_stops_df['Location'])]

filtered_loc_df = loc_df[loc_df['stop_name'].isin(common_stops_df['stop_name'])][['stop_name', 'stop_lat', 'stop_lon']]

merged_df = pd.merge(filtered_loc_delay_df, filtered_loc_df, left_on='Location', right_on='stop_name', how='inner')

merged_df['Original_Location'] = merged_df['Location']
merged_df['Matched_Stop_Name'] = merged_df['stop_name']


merged_df.drop(columns=['stop_name'], inplace=True)


merged_df.to_csv('common_stops_with_both_names.csv', index=False)


columns_to_check_for_duplicates = ['Date', 'Route', 'Time', 'Day', 'Location', 'Incident', 'Min Delay', 'Min Gap', 'Direction', 'Vehicle']
merged_df = merged_df.drop_duplicates(subset=columns_to_check_for_duplicates, keep='first')

merged_df.to_csv('common_stops_with_both_names_unique.csv', index=False)