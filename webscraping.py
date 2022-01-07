"LIBRARIES"
import requests
from bs4 import BeautifulSoup



#1-Extract Facebook Likes
for i in user['USERNAME']:
    url = 'https://www.facebook.com/'+ i
    response = requests.get(url)
    soup = BeautifulSoup(response.content,'lxml')
    f = soup.find('div', attrs={'class': '_4-u3 _5sqi _5sqk'})
    if f != None:
        likes=f.find('span',attrs={'class':'_52id _50f5 _50f7'}) #finding span tag inside class
        print(i,likes.text)

