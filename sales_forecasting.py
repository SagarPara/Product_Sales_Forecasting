import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from datetime import date


sales_data = pd.read_csv('TRAIN.csv')


#Load the product model
#with open("Predict_Sales_Cost_MLmodel_stacked.pickle", "rb") as f:
#     model = pickle.load(f)


#Load the product model
with open("Predict_Sales_Cost_MLmodel_LR.pickle", "rb") as f:
     model = pickle.load(f)



st.title("Product Sales Prediction")
st.dataframe(sales_data)
st.subheader("Enter product details to predict sales")

df = pd.DataFrame(sales_data)


# Define mappings
mappings = {
    "region": {"R1": 0, "R2": 1, "R3": 2,  "R4": 3},
    "location": {"L1": 0, "L2": 1, "L3": 2,  "L4": 3},
    "store": {"S1": 0, "S2": 1, "S3": 2,  "S4": 3},
    }


user_inputs = {}
for key, mapping in mappings.items():
     selected_Str = st.selectbox(key, list(mapping.keys()))
     user_inputs[key] = mapping[selected_Str]



#selected_date = st.date_input("Date", date.today()).toordinal()
#selected_date = 0 if st.selectbox("Date", [1, 0]) == 1 else 0

#unique_holiday = df["Holiday"].unique()
holiday = 0 if st.selectbox("Holiday", [1, 0]) == 1 else 0

discount = 0 if st.selectbox("Discount", ["Yes", "No"]) == "Yes" else 0

order = st.number_input("Order", min_value=0, max_value=1000)



if st.button("Get Sales Price Prediction"):
     input_sales = [
          user_inputs["region"], user_inputs["location"], user_inputs["store"], holiday, discount, order
          ]
     
     #price = model.predict([input_sales])[0]
     #st.header(f"Predicted Sales Price {round(price, 2)}")

     
     
try:
          
          price = model.predict([input_sales])[0]
          st.header(f"Predicted Sales Price {round(price, 2)}")

          

except Exception as e:
        
        st.error("Error while predicting sales price: " + str(e))
        

