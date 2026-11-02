import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

with open('labelencoder.pkl','rb') as file:
  label=joblib.load(file)
with open('onehotencoder.pkl','rb') as file:
  onehot=joblib.load(file)
with open('scaler.pkl','rb') as file:
  scaler=joblib.load(file)
model=load_model('model.h5')

def survival(p,g,m,f,e):
  user={
    'Pclass':[p],
    'Sex':[g],
    'Members':[m],
    'Fare':[f],
    'Embarked':[e]
  }
  df=pd.DataFrame(user)
  df['Sex']=label.transform(df['Sex'])
  Embarked=onehot.transform(df[['Embarked']])
  Embarked=pd.DataFrame(Embarked,columns=onehot.get_feature_names_out()).astype('int32')
  df=pd.concat([df.drop(columns=['Embarked']),Embarked],axis=1)
  df[['Pclass','Sex','Fare','Members']]=scaler.transform(df[['Pclass','Sex','Fare','Members']])
  return model.predict(df)

st.title('Passenger Survival Chances in Titanic Journey')
Pclass=st.slider('Select the class of the Passenger: ',1,3,1,step=1)
Gender=st.selectbox('Select the Gender: ',['male','female'])
members=st.slider('Enter the size of the family :',1,11,4,1)
fare=st.number_input('Enter the amount paid for tickets :',0.0,value=10.89)
Embarked=st.selectbox('Select the Location from where Journey started :',['Queenstown','Cherbourg','Southampton'])

if st.button("Tell Chances"):
  chances=survival(Pclass,Gender,members,fare,Embarked)
  container=st.container(border=True)
  container.write(f'Probability for passenger to survie the journey :{chances[0][0]}')
  if chances[0][0]>=0.5:
    container.success("Passenger will survive the Titanic Journey")
  else:
    container.info("Passenger wont survive the Titanic Journey")