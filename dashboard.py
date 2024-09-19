import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca.colormap as cm

df = pd.read_csv("common_stops_with_both_names_unique.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Weekday'] = df['Date'].dt.day_name()

df['Stop'] = df['Location']


pred_df = pd.read_csv('wwww.csv')
d = pd.merge( df,pred_df, on='Location', how='left')


st.set_page_config(page_title="Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .folium-map {
        width: 80% !important;  /* Force the map to take up full width */
        height: auto !important;  /* Auto height to maintain aspect ratio */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Bus Incident and Delay")


def create_map(df, selected_stop):
    stop_lat_lon = df[['stop_lat_x', 'stop_lon_x', 'Stop', 'predictions']].drop_duplicates().reset_index(drop=True)
    stop_lat_lon['predictions'] = stop_lat_lon['predictions'].interpolate(method='linear', limit_direction='both')

    norm = mcolors.Normalize(vmin=stop_lat_lon['predictions'].min(), vmax=stop_lat_lon['predictions'].max())
    cmap = plt.get_cmap('coolwarm')

    m = folium.Map(location=[stop_lat_lon['stop_lat_x'].mean(), stop_lat_lon['stop_lon_x'].mean()], zoom_start=11)


    for _, row in stop_lat_lon.iterrows():

        color = mcolors.to_hex(cmap(norm(row['predictions'])))

        folium.CircleMarker(
            location=[row['stop_lat_x'], row['stop_lon_x']],
            radius=10 if row['Stop'] == selected_stop else 4,
            color='green' if row['Stop'] == selected_stop else color,
            fill=True,
            fill_color=color,
            popup=f"{row['Stop']} - Delay: {row['predictions']}"
        ).add_to(m)


    colormap = cm.LinearColormap(colors=[(0, 0, 1, 0.5), (1, 0, 0, 0.5)],
                                 vmin=stop_lat_lon['predictions'].min(),
                                 vmax=stop_lat_lon['predictions'].max(),
                                 caption='Delay distribution (min)')


    colormap.add_to(m)
    return m



left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    stop_name = st.selectbox("Select Stop", df['Stop'].unique())
    map_object = create_map(d, stop_name)
    folium_static(map_object, width=800, height=300)

outer_col1, col1, outer_col2 = st.columns([0.1, 0.8, 0.1])
with outer_col1:
    st.write("")
with col1:
    inner_col1, inner_col2, inner_col3 = st.columns([1, 1, 1])
    with inner_col1:
        filtered_df = df[df['Stop'] == stop_name]
        incident_count = filtered_df['Incident'].value_counts()

        fig_incident = px.pie(
            incident_count,
            values=incident_count.values,
            names=incident_count.index,
            title="Incident breakdown",
            hole=0.4
        )
        fig_incident.update_layout(width=350, height=350)
        st.plotly_chart(fig_incident)

    with inner_col2:
        avg_delay_by_month = filtered_df.groupby('Month_Name')['Min Delay'].mean().reindex(
            ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December']
        )

        fig_month = go.Figure(go.Bar(
            x=avg_delay_by_month.index,
            y=avg_delay_by_month.values,
            marker=dict(color='royalblue'),
            name='Avg Delay'
        ))
        fig_month.update_layout(
            title="Average delay per month",
            xaxis_title="Month",
            yaxis_title="Average delay (min)",
            width=350, height=350
        )
        st.plotly_chart(fig_month)

    with inner_col3:
        avg_delay_by_weekday = df.groupby('Weekday')['Min Delay'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )

        fig_weekday = go.Figure(go.Bar(
            x=avg_delay_by_weekday.index,
            y=avg_delay_by_weekday.values,
            marker=dict(color='lightgreen'),
            name='Avg Delay'
        ))
        fig_weekday.update_layout(
            title="Average delay by weekday",
            xaxis_title="Days",
            yaxis_title="Average delay (min)",
            width=350, height=350
        )
        st.plotly_chart(fig_weekday)
with outer_col2:
    st.write("")
