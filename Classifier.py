import pandas as pd
import hashID
import numpy as np
import pickle
import joblib
from chepy_fix import Chepy
from sklearn.model_selection import train_test_split  #type: ignore
from sklearn.feature_extraction.text import CountVectorizer #type: ignore
from sklearn.naive_bayes import MultinomialNB  #type: ignore
import argparse



#CONSTANTS
vectorizer = joblib.load('vectorizer.pkl')
model = pickle.load(open("model.txt","rb"))
classes = model.classes_

def isHash(decoded):
    """Check if a string is a hash and return matches"""
    hashes = hashID.main(decoded)
    return hashes


def classifier(encoded):
    """Using Naive Bayes check encoding type of string and return list of weights and 
    classes"""
    user_input = encoded
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_weights = model.predict_proba(user_input_vectorized)
    predicted_label = model.predict(user_input_vectorized)
    inSet = True
    weights = []
    for i in range(len(predicted_weights[0])):   
        pWeights = predicted_weights[0]
        k = abs(float(pWeights[i]))
        cl = str(classes[i])
        weights.append([k,cl])
        weights.sort(reverse=True)
    wNums = []
    for i in weights:
        wNums.append(i[0])
    return weights


def decode(data,encType): #data decode 
    """decode string given encoding type and return decoded string"""
    data = Chepy(data)

    #dictionary of encoding methods as well as stringified code for decoding
    encDict =  {
        'base64' : 'data.from_base64()',
        'base16' : 'data.from_base16()',
        'base32' : 'data.from_base32()',
        'base45' : 'data.from_base45()',
        'base58' : 'data.form_base58()',
        'base62' : 'data.from_base62()',
        'base85' : 'data.from_base85()',
        'base92' : 'data.from_base92()',
        'url' : 'data.from_url_encoding()',
        'binary': 'data.from_binary()',
        'html entity': 'data.from_html_entity()',
        'charcode': 'data.from_charcode()',
        'octal' : 'data.from_octal()',
        'hex' : 'data.from_hex()',
        'morse' : 'data.from_morse_code()',
        'rot_13' : 'data.rot_13()',
        'rot_47' : 'data.rot_47()',
        'plain' : 'data.to_string()'
    }
    decoded =  eval(encDict[encType])
    return decoded


def main(usrString,verbosity,returnOnly):
    """main program loop"""
    isHashFlag = False
    hashMap = [usrString]
    decString_2 = None
    decString_1 = None
    initial = usrString
    while True:
        test = usrString
        usrString = str(usrString)
        
        encoding = classifier(usrString)
        if encoding[0][1] == 'plain':
            for enc in encoding:
                if enc[1] == "rot_13" and ((enc[0]*100) > 0.01):
                    rot_dec = str(decode(usrString,"rot_13"))
                    rot_class = classifier(rot_dec)
                    for rot in rot_class:
                        if rot[1] == "plain" and rot[0] > encoding[0][0]:
                            usrString = rot_dec
                        else:
                            break
                            


            if not returnOnly:
                print(f'\nfinal output is: \n{usrString}')
            else:
                print(f'200,{isHashFlag},{usrString}')
            return 200, isHashFlag, usrString
        if encoding[0][1] == "morse":
            usrString = usrString.split('\n')
            usrString = ' '.join(usrString)
        
        hashes = isHash(str(usrString))
        if len(hashes) > 1:
            isHashFlag = True
            initial = usrString
            if isHashFlag and not returnOnly:
                
                if verbosity:                     
                    print(f"\n\n [!!ATTENTION!!] string may be HASH of following types.\n Identified as {encoding[0][1]} encoding method")
                    for i in hashes:
                        print(f'[+] {i[0]}')
                    print("\n\n\n")
                else:
                    print(f"\n{str(usrString)}\n[!!ATTENTION!!] string may be HASH type.")
                    for i in hashes[:3]:
                        print(f'[+] {i[0]}')
                    print(f" \nIdentified as encoding {encoding[0][1]} - {round(encoding[0][0]*100)}%\n\n\n")


        
#legitimacy tests
        iterCount = 0
                    
            
        for k in range(len(encoding)):
            iterCount+= 1
            
            i = encoding[k]
            if k>0 and (encoding[k-1][1] == 'base32'):
                i = [0,'base64']
                k = k-1
                
           
           

            
            
            
            
            

# first level encoding legitimacy test
            try:
                decString_1 = (decode(usrString,i[1]))
                decString_1_bytes = decString_1.o
                decString_1 = str(decString_1)
            except:
                decString_1 = "Could not convert to str"
                


            if decString_1 == "":
                continue

            if "Could not convert to str" in (decString_1):
                continue

            if decString_1 in hashMap:
                continue

            else: #second level encoding legitimacy test 
                encoding_1 = classifier((decString_1))
                
                if encoding_1[0][1] == 'plain':
                    if decString_1[:4] not in str(decString_1_bytes):
                        hashMap.append(decString_1)
                        continue
                    if verbosity and not returnOnly:
                        print(f'{usrString}\n{round((i[0]*100),4)}% {i[1]}')
                    usrString = decString_1
                    break
                
                for enc in encoding_1:
                        
                    try:
                        decString_2 = str(decode((decString_1),enc[1]))
                    except:
                        decString_2 = "Could not convert to str"

                    if (enc[1] == "rot_47" and i[1] == enc[1]) or (enc[1] == "rot_13" and i[1] == enc[1]):
                        break

                    isCheck2Hash = isHash(decString_1)
                    if len(isCheck2Hash) > 0:
                        usrString = decString_1
                        
                        if verbosity and not returnOnly:
                            tried = []
                            for j in encoding[:k+1]:
                                tried.append(j[1])
                            print(f'\ntried: {', '.join(tried)}\n{test}\n{round((i[0]*100),10)}% {i[1]}\n')
                            break

                    if "Could not convert to str" in str(decString_2):
                        if enc == encoding_1[-1]:
                            break
                        continue

                    if str(decString_2) in hashMap:
                        continue
                    
                    

                    elif decString_1 != '':
                        usrString = decString_1
                        if verbosity and not returnOnly:
                            tried = []
                            for j in encoding[:k+1]:
                                tried.append(j[1])
                            print(f'\ntried: {', '.join(tried)}\n{test}\n{round((i[0]*100),4)}% {i[1]}\n')
                        hashMap.append(str(decString_1))
                        break
                if usrString != test:
                    hashMap.append(str(decString_1))
                    break


        if iterCount > (len(encoding))-1:
            
            if not returnOnly:
                print(f"no encoding match found for string \n{initial}")
                if isHashFlag:
                    print("likely string is HASH")
            else:
                print(f"400,{isHashFlag},{initial}")
                pass
            return 400, isHashFlag, initial


#command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("userString")
parser.add_argument('-v', '--verbose',action='store_true',help="returns detailed decoding information")
parser.add_argument('-r', '--returnOnly',action='store_true',help="super minimal, just the essentials")
args = parser.parse_args()
userString = str(args.userString)
verbosity = args.verbose
returnOnly = args.returnOnly


main(userString,verbosity,returnOnly)