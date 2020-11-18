import requests
import json
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def fetchEvents():
	apiUrl = "https://o136z8hk40.execute-api.us-east-1.amazonaws.com/dev/get-list-of-conferences"

	response = requests.get(apiUrl)

	jsonFile = response.json()
	return jsonFile['free']+jsonFile['paid']


def displayEvents(eventList):
	print("="*20)
	print("UPCOMING EVENTS ARE ->")
	print("="*20)
	for event in eventList:
		print(f"{event['confName']},  {event['confStartDate']},  {event['city']}, {event['state']}.  {event['entryType']}, {event['confUrl']}\n")


def findExactDuplicates(eventList):
	alreadyPresent = {}

	for event in eventList:
		eventInfo = event['confName']+" "+event['confStartDate']+" "+event['venue']
		if eventInfo in alreadyPresent:
			alreadyPresent[eventInfo] += 1
		else:
			alreadyPresent[eventInfo] = 1
	print("="*20)
	print("EXACT DUPLICATES ARE ->")
	print("="*20)
	for event in alreadyPresent.keys():
		if alreadyPresent[event] >1:
			print(event," is present ",alreadyPresent[event], "times.\n")

#preprocessing for finding similarity
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def findSemanticDuplicates(eventList):
	print("="*20)
	print("Semantic Duplicates")
	print("="*20)
	n = len(eventList)
	for i in range(n-1):
		for j in range(i+1,n):
			event1 = eventList[i]['confName']+eventList[i]['confStartDate']
			event2 = eventList[j]['confName']+eventList[j]['confStartDate']
			similarity = cosine_sim(event1, event2)
			if similarity > .7:
				print(event1,f" looks {similarity*100}% similar to ", event2, "\n")

#main function
def main():
	eventList = fetchEvents()
	displayEvents(eventList)
	findExactDuplicates(eventList)
	findSemanticDuplicates(eventList)


main()


