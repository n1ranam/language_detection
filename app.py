import streamlit as st
import pickle
import pandas as pd

with open('model1.pkl','rb') as file:
  model1=pickle.load(file)
with open('count_vectorizer.pkl','rb') as f:
    cv=pickle.load(f)
with open('label encoder.pkl','rb') as f:
    le=pickle.load(f)

def main():
    st.title("Language detection")
    input_text=st.text_input("Enter the sentence:")
    if st.button('Predict'):
        x=cv.transform([input_text]).toarray()
        Prediction=model1.predict(x)

        if Prediction ==10:
            st.write("The language is Malayalam")
        elif Prediction ==2:
            st.write("The language is Dutch")
        elif Prediction ==3:
            st.write("The language is English")
        elif Prediction ==4:
            st.write("The language is Portugese")
        else:
            st.write("Not Defined language")

if __name__=='__main__':
    main()