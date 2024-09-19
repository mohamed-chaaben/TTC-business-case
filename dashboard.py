import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("common_stops_with_both_names_unique.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Weekday'] = df['Date'].dt.day_name()

df['Stop'] = df['Location']
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
    stop_lat_lon = df[['stop_lat', 'stop_lon', 'Stop']].drop_duplicates().reset_index(drop=True)
    m = folium.Map(location=[stop_lat_lon['stop_lat'].mean(), stop_lat_lon['stop_lon'].mean()], zoom_start=11)

    for _, row in stop_lat_lon.iterrows():
        if row['Stop'] == selected_stop:
            folium.CircleMarker(
                location=[row['stop_lat'], row['stop_lon']],
                radius=7,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"Stop: {row['Stop']}"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['stop_lat'], row['stop_lon']],
                radius=4,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=f"Stop: {row['Stop']}"
            ).add_to(m)

    return m

left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    stop_name = st.selectbox("Select Stop", df['Stop'].unique())
    map_object = create_map(df, stop_name)
    folium_static(map_object, width=800, height=300)

outer_col1, col1, outer_col2 = st.columns([0.1, 0.8, 0.1])  # Outer columns for margins
with outer_col1:
    st.write("")  # Left margin
with col1:
    inner_col1, inner_col2, inner_col3 = st.columns([1, 1, 1])
    with inner_col1:
        filtered_df = df[df['Stop'] == stop_name]
        incident_count = filtered_df['Incident'].value_counts()

        fig_incident = px.pie(
            incident_count,
            values=incident_count.values,
            names=incident_count.index,
            title="Incident Breakdown",
            hole=0.4
        )
        fig_incident.update_layout(width=350, height=350)  # Ensure each chart is not too wide
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
            yaxis_title="Average Delay (Minutes)",
            width=350, height=350
        )
        st.plotly_chart(fig_weekday)
with outer_col2:
    st.write("")
