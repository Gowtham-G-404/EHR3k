import pandas as pd
from flask import Flask,render_template,request
import time
import requests
from base64 import b64encode, b64decode
import hashlib
from Cryptodome.Cipher import AES
import os
from Cryptodome.Random import get_random_bytes
import training
import predict
def call_train():
    training.main()
    
def call_predict():
    final_results = predict.main()
    return final_results
    
def encrypt(plain_text, password):
    # generate a random salt
    salt = get_random_bytes(AES.block_size)
    
    # use the Scrypt KDF to get a private key from the password
    private_key = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    
    # create cipher config
    cipher_config = AES.new(private_key, AES.MODE_GCM)

    # return a dictionary with the encrypted text
    cipher_text, tag = cipher_config.encrypt_and_digest(bytes(plain_text, 'utf-8'))
    return {'Cipher Text': b64encode(cipher_text).decode('utf-8')}

        

def main(data):
    password = "abcdefgh"
    # encrypt secret message
    encrypted = encrypt(data, password)

    return encrypted

def connect2cerner():

    head = {'Accept': 'application/fhir+json',

            'Authorization': 'OAuth2 Bearer Token'}

    cern = "https://fhir-open.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d/Patient?_id=12724067"

    cern_patient = requests.get(cern, headers=head)

    return cern_patient.text

app = Flask(__name__)
  
@app.route('/')
def fun1():
    return render_template('data.html')

@app.route('/data',methods =['POST'])
def fun2():
    global ehr_data
    ehr_format = request.form.get("ehr_foramt")
    if ehr_format == "epic":
        df = pd.read_json("Abdul218_Upton904_766e68a7-d5b2-639a-5b96-419c68c65063.json")
        ehr_data = df.sample(1).to_string()
        return '<p>System Connected with EPIC</p>'
    else:
        ehr_data = connect2cerner()
        return '<p>System Connected with CERNER</p>'
    
   
    

@app.route('/encrypt',methods =['POST'])
def fun3():
    
    select = request.form.get("encryption")
    if select == "e1":
        cipher = main(ehr_data)
        return cipher
    elif select == "e2":
        cipher = main(ehr_data)
        return cipher
    else:
        cipher = main(ehr_data)
        return cipher
       
    
@app.route('/training')
def fun4():
    call_train()
    
    return render_template('train.html')
 

@app.route('/tt')
def fun5():
    time.sleep(3)
    return '<h3 style="color:black;text-align:left;">Model Training Completed Successfully</h3>'


@app.route('/pp')
def fun6():   
    
    df = pd.read_excel("Hekma_Prediction_Results.xlsx")
    table = df.to_html(index=False,justify="left")
   
    return table.replace('<tr>', '<tr align="center">')



if __name__=='__main__':
    
    app.run(debug=True)
    