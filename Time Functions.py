'TIME FUNCTIONS'
## Splitting date into day,month,year
concat["day"] = concat['Orderdate'].map(lambda x: x.day)
concat["month"] = concat['Orderdate'].map(lambda x: x.month)
concat["year"] = concat['Orderdate'].map(lambda x: x.year)
concat['dayofweek'] = pd.to_datetime(concat['Orderdate']).dt.dayofweek

##  Split Weeks
def Week(x):
    if x >= 1 and x<= 8:
        return 1
    elif x >= 9 and x<= 16:
        return 2
    elif x >= 17 and x<= 24:
        return 3
    else:
        return 4

## Split into Weekday/Weekend
def Weekdayend(x):
    if x >= 0 and x<= 4:
        return 1
    else:
        return 2


## Segregate a month into 4 parts
def Monthsplitup(x):
    if x >= 1 and x<= 10:
        return 1
    elif x >= 11 and x<= 20:
        return 2
    elif x >= 21 and x<= 25:
        return 3
    else:
        return 4


## Segregate into Quarters
def Quarter(x):
    if x >= 1 and x<= 3:
        return 1
    elif x >= 4 and x<= 6:
        return 2
    elif x >= 7 and x<= 9:
        return 3
    else:
        return 4

## Appying the Functions
concat['Week'] = concat['day'].apply(Week)
concat['Weekday/Weekend'] = concat['dayofweek'].apply(Weekdayend)
concat['Monthsplitup'] = concat['day'].apply(Monthsplitup)
concat['Quarter'] = concat['month'].apply(Quarter)


#Python program to convert time from 12 hour to 24 hour format
def convert24(str1):# Function to convert the date format
    # Checking if last two elements of time
    # is AM and first two elements are 12
    if str1[-2:] == "AM" and str1[:2] == "12":
        return "00" + str1[2:-2]

    # remove the AM
    elif str1[-2:] == "AM":
        return str1[:-2]

    # Checking if last two elements of time
    # is PM and first two elements are 12
    elif str1[-2:] == "PM" and str1[:2] == "12":
        return str1[:-2]

    else:
        return str(int(str1[:2]) + 12) + str1[2:8]# add 12 to hours and remove PM

# Applying Function
print(convert24("08:05:45 PM"))

-----------------------------------ALITER------------------------------------------------------------------
#import datetime
from datetime import datetime
#sample input time to be converted
inputTime = "08:05:45 PM"
#Create datetime object from string
in_time = datetime.strptime(inputTime, "%I:%M:%S %p")
#convert to 24 hour format
out_time = datetime.strftime(in_time, "%H:%M:%S")
#print result
print(out_time)


## Time to seconds
def time2seconds(time):
  if type(time) != str:
    return time
  parts = [float(p) for p in time.split(':')]
  parts = [p * (60 ** i) for i, p in enumerate(reversed(parts))]
  return sum(parts)

