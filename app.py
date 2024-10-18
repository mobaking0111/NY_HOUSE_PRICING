import streamlit as st 
from tensorflow.keras.models import load_model 
import pickle
import pandas as pd
import numpy as np 

model = load_model("model.keras")

sc = pickle.load(open('scaler.pkl' , 'rb'))

encoder = pickle.load(open('encoder.pkl' , 'rb'))

st.title("NY House Pricing")

house_type = st.selectbox("House Type" ,('Condo for sale' ,'House for sale' ,'Townhouse for sale', 'Co-op for sale','Multi-family home for sale'))

house_sublocality = st.selectbox("Subocality" , ('Manhattan' ,'New York County', 'Richmond County' ,'Kings County' ,'New York',
 'East Bronx' ,'Brooklyn' ,'The Bronx' ,'Queens' ,'Staten Island',
 'Queens County' ,'Bronx County', 'Coney Island', 'Brooklyn Heights',
 'Jackson Heights', 'Riverdale', 'Rego Park' ,'Fort Hamilton', 'Flushing',
 'Dumbo' ,'Snyder Avenue'))
 
house_bath = st.number_input(label="Number of Baths" , min_value=1,step=1) 

house_beds = st.number_input(label="Number of beds" , min_value=1,step=1) 

house_size = st.number_input(label="size in square feet" , min_value=1 , step=1)

### TYPE_0	TYPE_1	TYPE_2	BEDS	BATH	PROPERTYSQFT	SUBLOCALITY_0	SUBLOCALITY_1	SUBLOCALITY_2	SUBLOCALITY_3	SUBLOCALITY_4
df = pd.DataFrame([[house_type , house_beds , house_bath , house_size , house_sublocality]] , columns=['TYPE' ,'BEDS' , 'BATH' , 'PROPERTYSQFT' , 'SUBLOCALITY',])

#df1 = df[['TYPE' , 'SUBLOCALITY']]
df_encoded = encoder.transform(df) 


# #df2 = pd.DataFrame([[house_type , house_beds , house_bath , house_size , house_sublocality]] , columns= ['TYPE_0','TYPE_1','TYPE_2','BEDS','BATH','PROPERTYSQFT','SUBLOCALITY_0','SUBLOCALITY_1','SUBLOCALITY_2','SUBLOCALITY_3','SUBLOCALITY_4'])

col = ['BEDS', 'BATH', 'PROPERTYSQFT']

# #df2 = df[['BEDS' , 'BATH' , 'PROPERTYSQFT']]
df_encoded[col] = sc.transform(df_encoded[col])
# st.text(df_encoded)

# #Final_df = pd.concat([df_encoded , df_scaled ])
predict = model.predict(df_encoded)[0][0]
 
predict_button = st.button(label="Submit")
if (predict_button) :
    st.text(f"prediction : {np.exp(predict)}") 