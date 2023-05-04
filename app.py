import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
import os

#os.environ['OPENAI_API_KEY'] = 
st.set_page_config(layout='wide', page_title="Power Transformer DGA Monitoring", initial_sidebar_state='expanded')

with st.sidebar:
        st.header('About')
        st.markdown('<div style="text-align: justify;">A Dissolved Gas Analysis (DGA) monitoring application that uses machine learning to predict incipient faults from the transformer. Based on the fault probability, the following maintenance schedules are recommended:</div>', unsafe_allow_html=True)
        st.markdown(' ')
        st.markdown(':large_green_circle: Normal - Follow planned maintenance')
        st.markdown(':large_orange_circle: Caution - Schedule maintenance within 7 days')
        st.markdown(':red_circle: Hazardous - Schedule maintenance within 48 hours')
        st.markdown(' ')
        st.markdown("<div style='font-size : 14px; font-style: italic'>*click on the map markers to view each transformer's condition</div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size : 14px; font-style: italic">**the real-time automated diagnosis is powered by GPT3.5 model from OpenAI</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size : 14px; font-style: italic">***best viewed in desktop browser</div>', unsafe_allow_html=True)

# re-arrange data
df = pd.read_csv('all bank 1.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
address = pd.read_csv('address.csv')
model = pickle.load(open('model.pkl', 'rb'))
latest = df.groupby('SUBSTATION').last()
latest['SUBSTATION'] = latest.index
latest = latest.reset_index(drop=True)
X = latest.iloc[:,2:7]
df2 = pd.DataFrame(columns=['Arc discharge', 'High-temperature overheating', 'Low-temperature overheating', 'Middle-temperature overheating', 'Normal', 'Partial discharge', 'Spark discharge', 'max'])
for i in range(7):
    df2.iloc[:,i] = pd.DataFrame(model.predict_proba(X)[i])[1]
df2['max'] = df2.max(axis=1)

# set page rows and columns
container = st.container()
col_low, col_medium, col_high, col_pd, col_spark, col_arc = st.columns(6)
st.markdown(' ')
col1, col2 = st.columns((1,1))

# initialize chatGPT model
# @st.cache_resource
# def get_models():
#     chat = ChatOpenAI(temperature=0.1)
#     return chat

#plot gas concentrations
def plot(df, lat, lng):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.append_trace(go.Scatter(
        x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
        y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['HYDROGEN'],
        name='Hydrogen'
    ), row=1, col=1)
    fig.append_trace(go.Scatter(
        x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
        y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['METHANE'],
        name='Methane'
    ), row=2, col=1)
    fig.append_trace(go.Scatter(
        x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
        y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ETHANE'],
        name='Ethane'
    ), row=3, col=1)
    fig.append_trace(go.Scatter(
        x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
        y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ETHYLENE'],
        name='Ethylene'
    ), row=4, col=1)
    fig.append_trace(go.Scatter(
        x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
        y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ACETYLENE'],
        name='Acetylene'
    ), row=5, col=1)
    fig.update_layout(height=425, width=500,
            title_text='Dissolved Gases Concentrations',
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True)
    fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)', 'paper_bgcolor':'rgba(0,0,0,0)'})
    fig.update_xaxes(showticklabels=False)
    fig['layout']['yaxis3']['title']='Parts per million'
    st.plotly_chart(fig, use_container_width=True)

# calculate model confidence level
def fault_score(df, address, lat, lng):
    pxf = address[(address['LAT'] == lat) & (address['LNG'] == lng)]['SUBSTATION'].iloc[0]
    score = model.predict_proba(df[df['SUBSTATION'] == pxf].iloc[-1:None,3:8])
    score2 = model.predict_proba(df[df['SUBSTATION'] == pxf].iloc[-2:-1,3:8])
    delta = np.array(score)-np.array(score2)
    pxf_no = str(address[(address['LAT'] == lat) & (address['LNG'] == lng)].index[0]+1)
    container.subheader('Power Transformer #'+pxf_no)

    with col_low:
        if round(delta[2][0][1]*100) == 0:    
            st.metric(label='Low Overheating', value=str(round(score[2][0][1]*100))+" %")
        else:
            st.metric(label='Low Overheating', value=str(round(score[2][0][1]*100))+" %", delta=str(round(delta[2][0][1]*100))+" %", delta_color='inverse')
    with col_medium:
        if round(delta[3][0][1]*100) == 0:
            st.metric(label='Medium Overheating', value=str(round(score[3][0][1]*100))+" %")
        else:
            st.metric(label='Medium Overheating', value=str(round(score[3][0][1]*100))+" %", delta=str(round(delta[3][0][1]*100))+" %", delta_color='inverse')
    with col_high:
        if round(delta[1][0][1]*100) == 0:
            st.metric(label='High Overheating', value=str(round(score[1][0][1]*100))+" %")  
        else:
            st.metric(label='High Overheating', value=str(round(score[1][0][1]*100))+" %", delta=str(round(delta[1][0][1]*100))+" %", delta_color='inverse')
    with col_pd:
        if round(delta[5][0][1]*100) == 0:
            st.metric(label='Partial Discharge', value=str(round(score[5][0][1]*100))+" %")
        else:
            st.metric(label='Partial Discharge', value=str(round(score[5][0][1]*100))+" %", delta=str(round(delta[5][0][1]*100))+" %", delta_color='inverse')
    with col_spark:
        if round(delta[6][0][1]*100) == 0:
            st.metric(label='Spark Discharge', value=str(round(score[6][0][1]*100))+" %")
        else:
            st.metric(label='Spark Discharge', value=str(round(score[6][0][1]*100))+" %", delta=str(round(delta[6][0][1]*100))+" %", delta_color='inverse')
    with col_arc:
        if round(delta[0][0][1]*100) == 0:
            st.metric(label='Arc Discharge', value=str(round(score[0][0][1]*100))+" %")
        else:
            st.metric(label='Arc Discharge', value=str(round(score[0][0][1]*100))+" %", delta=str(round(delta[0][0][1]*100))+" %", delta_color='inverse')

# plot interactive map
def main():
    with col1:
        m = folium.Map(location=[14.517855931485975, 121.06789220281651], zoom_start=9)
        folium.TileLayer('cartodbdark_matter').add_to(m)
        for i in range(len(address)):
            if df2['max'][i] < 0.6:
                color = 'darkgreen'
            elif (df2['max'][i] >= 0.6) and (df2['max'][i] < 0.80):
                color = '#d45800'
            elif df2['max'][i] >= 0.80:
                color = 'darkred'
            folium.CircleMarker(location=[address['LAT'][i], address['LNG'][i]], 
                                radius=2.5,
                                color=color,
                                fill=True,
                                fill_color=color,
                                fill_opacity=1,
                                tooltip=address['SUBSTATION'][i]).add_to(m)

        st_data = st_folium(m, width=500, height=425, returned_objects=["last_object_clicked"])

    # show transformer 1 by default
    with col2:
        if st_data["last_object_clicked"] is None:
            lat = address[address['SUBSTATION'] == 'Substation1']['LAT'].iloc[0]
            lng = address[address['SUBSTATION'] == 'Substation1']['LNG'].iloc[0]
            plot(df, lat, lng)
            fault_score(df, address, lat, lng)
        
        elif st_data["last_object_clicked"] is not None:
            lat = st_data["last_object_clicked"]['lat']
            lng = st_data["last_object_clicked"]['lng']
            plot(df, lat, lng)
            fault_score(df, address, lat, lng)      

    # get insights from chatGPT
    # st.markdown('<div style="font-size : 20px; font-weight: bold">Automated Diagnosis</div>', unsafe_allow_html=True)
    # st.markdown(' ')
    # chat = get_models()
    # messages = [SystemMessage(content="""You are an expert in dissolved gas analysis who knows how to diagnose transformers.
    #                                     I will give you the ppm concentrations of the gases
    #                                     and you will provide your analysis and recommendations."
    #                           """)]
    # h_ppm = str(df[(df['LAT'] == lat) & (df['LNG'] == lng)]['HYDROGEN'].iloc[-1])
    # meth_ppm = str(df[(df['LAT'] == lat) & (df['LNG'] == lng)]['METHANE'].iloc[-1])
    # etha_ppm = str(df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ETHANE'].iloc[-1])
    # ethy_ppm = str(df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ETHYLENE'].iloc[-1])
    # acet_ppm = str(df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ACETYLENE'].iloc[-1])

    # user_message = "Hydrogen = "+h_ppm+", Methane = "+meth_ppm+", Ethane = "+etha_ppm+", Ethylene = "+ethy_ppm+", Acetylene = "+acet_ppm
    # messages.append(HumanMessage(content=user_message))
    # result = chat(messages).content
    # st.write(result)

if __name__ == '__main__':
    main()
