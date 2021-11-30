import pickle
import gradio as gr
gaussian = pickle.load(open('Admin.sav','rb'))
y_pred = gaussian.predict(xtest)
output = pd.DataFrame(y_pred)
print(output)

                              '2-GRADIO'
#1-REGRESSION

def prediction(OrganicSearches,Users):
    # Making predictions
    prediction = Orders.predict([[OrganicSearches,Users]])
    return prediction

OrganicSearch = gr.inputs.Slider(minimum=0, maximum=120, default=22, label="OS")
Users = gr.inputs.Slider(minimum=0, maximum=1000, default=100, label="Fare (british pounds)")

output = gr.outputs.Textbox(label='Weborders')
gr.Interface( fn=prediction,
              inputs =  [OrganicSearch,Users],outputs= output,live=True).launch(share=True)



#2-CLASSIFICATION
import gradio as gr
import pandas as pd
import numpy as np
import sklearn

def predict_survival(sex, age, fare):
    df = pd.DataFrame.from_dict({'Sex': [sex], 'Age': [age], 'Fare': [fare]})
    df = encode_sex(df)
    df = encode_fares(df)
    df = encode_ages(df)
    pred = clf.predict_proba(df)[0]
    return {'Perishes': pred[0], 'Survives': pred[1]}

sex = gr.inputs.Radio(['female', 'male'], label="Sex")
age = gr.inputs.Slider(minimum=0, maximum=120, default=22, label="Age")
fare = gr.inputs.Slider(minimum=0, maximum=1000, default=100, label="Fare (british pounds)")

gr.Interface(predict_survival, [sex, age, fare], "label", live=True).launch()