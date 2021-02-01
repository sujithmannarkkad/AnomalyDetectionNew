from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import requests
import json
import math


app = Flask(__name__)
#model = pickle.load(open('Anomaly_Detection_model.pkl', 'rb'))
knn_model = pickle.load(open('Anomaly_Detection_model_knn_new.pkl', 'rb'))
df = pd.read_csv('Master Lookup AM Anomalies v2.0.csv',delimiter=',',header='infer')
scoring_uri='http://9d08086f-0622-4888-b345-5cad6ac1500b.eastus2.azurecontainer.io/score'


@app.route('/')
def home():
    return render_template('Metric Submission - MyWizard.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    lst=[]
    for x in request.form.values():
       lst.append(float(x))


    #input='(AO)Percentage SLAs Met={}, Ageing={}, Average Resolution Effort - Incidents={},      #Average Resolution Effort - Problems={}, Backlog Processing Efficiency - #Incidents={},Backlog Processing Efficiency - Problems={}, Delivered #Defects={}'.format(lst[0], lst[1],lst[2],lst[3],lst[4],lst[5],lst[6])

    #result=model.predict(csr_matrix([100,0,1144,1144,0.17,0.17,179]))
    result=model.predict(csr_matrix(lst))
    resultnew=pd.DataFrame(result.todense())
    resultnew1=pd.DataFrame(resultnew.iloc[0])
    predictions = resultnew1[resultnew1[0] > 0]
    print(lst)
    print(resultnew)
    anomaly=[]
    for i in list(predictions.index.values):
        anomaly.append(Metrics(i))


    finalResult=''
    for x in range(len(anomaly)):
        finalResult=finalResult+anomaly[x]+' ,'
    print('length',len(anomaly))
    if(len(anomaly)>1):
        text='{} seem to have suspicious data entry since their values are outside of normal range. Please validate once before submission.'.format(finalResult[:-1])
    elif (len(anomaly)==1):
        text='{} seem to have suspicious data entry since the value is outside of normal range. Please validate once before submission.'.format(finalResult[:-1])
    else:
        text='No Outliers detected.'
    return render_template('index.html',  prediction_text=text)

def Metrics(i):
    switcher={
                0:'SLA',
                1:'Ageing',
                2:'AREI',
                3:'AREP',
                4:'BPEI',
                5:'BPEP',
                6:'DD'

             }
    return switcher.get(i,"Invalid")

def Measures(i):
    switcher={
                'SLA': '%SLA met MTD Total SLAs Met, MTD Total SLA',
                'Ageing':'Number of open incidents exceeding resolution SLA',
                'AREI':'Total efforts spent on Incidents P1,Total efforts spent on Incidents P2',
                'AREP':'Total efforts spent on Problems P1,Total efforts spent on Problems P2',
                'BPEI':'Resolved Incidents P1,Resolved Incidents P2',
                'BPEP':'Resolved Problem Requests P1,Resolved Problem Requests P2',
                'DD':'Number of post delivery defects'

             }
    return switcher.get(i,"Invalid")
def outlier(lst):
    print('before call',lst)
    data = {"data":
        [
            lst
        ]
        }
    # Convert to JSON string
    input_data = json.dumps(data)
    print('input_data',input_data)
    # Set the content type
    headers = {'Content-Type': 'application/json'}
    # If authentication is enabled, set the authorization header
    #headers['Authorization'] = f'Bearer {key}'

    # Make the request and display the response
    resp = requests.post(scoring_uri, input_data, headers=headers)
    print('resp.text',resp.text)
    return resp.text

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    content = request.json
    lst=[]
    print(content)
    lst.append(float(content['SLA']))
    lst.append(float(content['Ageing']))
    lst.append(float(content['AREP']))
    lst.append(float(content['AREP']))
    lst.append(float(content['BPEI']))
    lst.append(float(content['BPEP']))
    lst.append(float(content['DD']))

    result1=outlier(lst)
    print('result1',result1=='[]')
    print('result1 length',len(result1))
    li=[]
    if(result1!='[]'):
        line = result1.replace('"', '')
        line = line.replace('[', '')
        line = line.replace(']', '')
        li = list(line.split(","))
        print(li)
    '''
    #commented for outlier detection load from api
    result=model.predict(csr_matrix(lst))
    resultnew=pd.DataFrame(result.todense())
    resultnew1=pd.DataFrame(resultnew.iloc[0])
    predictions = resultnew1[resultnew1[0] > 0]
    print(lst)
    print(resultnew)
    anomaly=[]
    for i in list(predictions.index.values):
        anomaly.append(Metrics(i))

    measures=[]
    for a in range(len(anomaly)):
        measures.append(Measures(anomaly[a]))


    finalResult=''
    for x in range(len(anomaly)):
        finalResult=finalResult+anomaly[x]+' ,'
    print('length',len(anomaly))
    if(len(anomaly)>1):
        text='{} seem to have suspicious data entry since their values are outside of normal range. Please validate once before submission.'.format(finalResult[:-1])
    elif (len(anomaly)==1):
        text='{} seem to have suspicious data entry since the value is outside of normal range. Please validate once before submission.'.format(finalResult[:-1])
    else:
        text='No Outliers detected.'

    finalResult1=''
    for x in range(len(measures)):
        finalResult1=finalResult1+measures[x]+' ,'
    print('measures',len(measures))
    if(len(measures)>1):
        text1='{} seem to have suspicious data entry since their values are outside of normal range. Please validate once before submission.'.format(finalResult1[:-1])
    elif (len(measures)==1):
        text1='{} seem to have suspicious data entry since the value is outside of normal range. Please validate once before submission.'.format(finalResult1[:-1])
    else:
        text1='No Outliers detected.'

    print('measures',text1)
    '''
    #testing
    print('knn starts')
    nearest_neighbor=knn_model.kneighbors([lst], return_distance=True)
    #print(nearest_neighbor)
    #df['warning_second_pass'] = df['warning_second_pass'].astype('str')
    #df['clusters_second_pass'] = df['clusters_second_pass'].astype('float')
    features=df.columns[0:7]
    df_array=np.array(df)

    result=np.unique(np.transpose(df_array[nearest_neighbor[1],7]))
    print("result length", len(result))
    print('result',result)


    rslt=[]
    for i in range(len(result)):
        print('type',type(result[i]))
        # if(math.isnan(result[i])):
        #     break
        rslt.append(result[i])
        print(rslt)
    if(rslt==[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]):
        rslt=[]

    # finalresult = {
    # "model1": text,
    # "model2":  rslt

    #  }

    # finalresultNew = {
    # "outlier": measures,
    # "anomaly":  rslt

    #  }
    finalresultNew = {
    "outlier": li,
    "anomaly":  rslt

     }

    return jsonify(finalresultNew)
    #return text

if __name__ == "__main__":
    app.run(debug=True)
