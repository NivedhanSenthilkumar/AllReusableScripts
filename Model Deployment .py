import pickle

gaussian = pickle.load(open('Admin.sav','rb'))
y_pred = gaussian.predict(xtest)
output = pd.DataFrame(y_pred)

print(output)

                              '2-GRADIO'
                            '1-REGRESSION'
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#load the dataset to pandas dataframe
URL = "http://bit.ly/w-data"
student_data = pd.read_csv(URL)
#Prepare data
X = student_data.copy()
y = student_data['Scores']
del X['Scores']
#create a machine learning model and train it
lineareg = LinearRegression()
lineareg.fit(X,y)
print('Accuracy score : ',lineareg.score(X,y),'\n')
#now the model has been trained well let test it
#function to predict the input hours
def predict_score(hours):
    hours = np.array(hours)
    pred_score = lineareg.predict(hours.reshape(-1,1))
    return np.round(pred_score[0], 2)
input = gr.inputs.Number(label=['Number of Hours studied'])
output = gr.outputs.Textbox(label='Predicted Score')
gr.Interface( fn=predict_score,
              inputs=input,
              outputs=output).launch(share=True)



                        "2-CLASSIFICATION"
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