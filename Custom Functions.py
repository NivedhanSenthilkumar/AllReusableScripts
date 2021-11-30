#Using loops to iterate through nested dictionary/list python
def nested_loop_dict(obj):
    # Iterate over all key-value pairs of dict argument
    for key, value in obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in  nested_loop_dict(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            yield (key, value)
print('Iterating over Nested Dict:')
for pair in nested_loop_dict(details):
    print('\n',pair)


#Unique values in a dictionary
dict = {'511':'Vishnu','512':'Vishnu','513':'Ram','514':'Ram','515':'sita'}
list =[] # create empty list
for val in dict.values():
  if val in list:
    continue
  else:
    list.append(val)
print(list)


#Numerical if
def pricesplit(x):
    if x>=0 and x<= 500:
        return 1
    if x>=501 and x<= 1000:
        return 2
    if x>=1001 and x<= 1500:
        return 3
    if x>=1501 and x<= 2000:
        return 4
    if x>=2001 and x<= 2500:
        return 5
    if x>=2501 and x<= 3000:
        return 6
    if x>=3001 and x<= 3500:
        return 7
    if x>=3501 and x<= 4000:
        return 8
    if x>=4001 and x<= 4500:
        return 9
    if x>=4501 and x<= 5000:
        return 10
    if x>=5001 and x<= 5500:
        return 11
    if x>=5501 and x<= 6000:
        return 12
    else:
        return 13


## Iterating Dictionary





#Writing to excel
import pandas as pd
df1 = pd.DataFrame({'Data': ['a', 'b', 'c', 'd']})
df2 = pd.DataFrame({'Data': [1, 2, 3, 4]})
df3 = pd.DataFrame({'Data': [1.1, 1.2, 1.3, 1.4]})

writer = pd.ExcelWriter('multiple.xlsx', engine='xlsxwriter')

df1.to_excel(writer, sheet_name='Sheeta')
df2.to_excel(writer, sheet_name='Sheetb')
df3.to_excel(writer, sheet_name='Sheetc')

writer.save()

##--ALITER
import csv
# my data rows as dictionary objects
mydict = [{'branch': 'COE', 'cgpa': '9.0', 'name': 'Nikhil', 'year': '2'},
          {'branch': 'COE', 'cgpa': '9.1', 'name': 'Sanchit', 'year': '2'},
          {'branch': 'IT', 'cgpa': '9.3', 'name': 'Aditya', 'year': '2'},
          {'branch': 'SE', 'cgpa': '9.5', 'name': 'Sagar', 'year': '1'},
          {'branch': 'MCE', 'cgpa': '7.8', 'name': 'Prateek', 'year': '3'},
          {'branch': 'EP', 'cgpa': '9.1', 'name': 'Sahil', 'year': '2'}]

# field names
fields = ['name', 'branch', 'year', 'cgpa']
# name of csv file

#String Split
df[['code', 'name_of_code']] = df["code"].str.split(" ", 1, expand=True)


#Separating Number from String
def find_number(text):
    num = re.findall(r'[0-9]+', text)
    return " ".join(num)

#Removing whitespace
def removewhitespace(x):
    """
    Helper function to remove any blank space from a string
    x: a string
    """
    try:
        # Remove spaces inside of the string
        x = "".join(x.split())

    except:
        pass
    return x


#Add values in a row separated by comma
def sum_of_number(listx):
                        a = []
                        n = 0
                        if len(listx) == 1:
                            a = listx
                            n = a
                               else:
                                   a = str(listx).split(',')
                                   for i in range(0, len(a)):
                                       n = n + int(a[i])
                return n

df['Sum'] = df['Scores'].apply(lambda x: sum(map(float, x.split(','))))


#String if
def regions(x):
    if x in ['WA', 'MT', 'OR', 'ID', 'WY', 'CA', 'NV', 'UT', 'CO', 'AK']:
        return ('West')
    if x in ['AZ', 'NM', 'TX', 'OK', 'HI']:
        return ('Southwest')
    if x in ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH']:
        return ('Midwest')
    if x in ['LA', 'AR', 'MS', 'AL', 'GA', 'FL', 'KY', 'TN', 'SC', 'NC', 'VA', 'DC', 'WV',
             'DE', 'MD', 'PR', 'VI']:
        return ('Southeast')
    if x in ['PA', 'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'VT', 'ME', 'GU']:
        return ('Northeast')
    if x in ['AE', 'AP', 'AA']:
        return ('Armedforces')


#ENCRYPTION and DECRYPTION
from cryptography.fernet import Fernet
    # we will be encryting the below string.
message = "hello geeks"

                           # generate a key for encryptio and decryption
                           # You can use fernet to generate
                           # the key or use random key generator
                           # here I'm using fernet to generate key

key = Fernet.generate_key()

                           # Instance the Fernet class with the key

fernet = Fernet(key)

                           # then use the Fernet class instance
                           # to encrypt the string string must must
                           # be encoded to byte string before encryption
encMessage = fernet.encrypt(message.encode())

print("original string: ", message)
print("encrypted string: ", encMessage)

                           # decrypt the encrypted string with the
                           # Fernet instance of the key,
                           # that was used for encrypting the string
                           # encoded byte string is returned by decrypt method,
                           # so decode it to string with decode methos
decMessage = fernet.decrypt(encMessage).decode()
print("decrypted string: ", decMessage)


#Replace
for i in numdata:
    numdata[i] = numdata[i].replace('-', None)



