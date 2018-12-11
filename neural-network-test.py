
# coding: utf-8

# In[1]:


# use natural language toolkit
    import nltk
    from nltk.stem.lancaster import LancasterStemmer
    import os
    import json
    import datetime
    import pymongo
    stemmer = LancasterStemmer()


# In[2]:


# Establish MongoDB connection and execute a query
    mongoInstance = pymongo.MongoClient("mongodb://localhost:27017/") # conect to local MongoDB instance
    mongoDB = mongoInstance["gis-chatbot"] # connect to mongo database called 'gis-chatbot'
    mongoCollection = mongoDB["training-data"] # connect to mongo collection called 'training-data'
    mongoQuery = {} # empty query = all 
    mongoResult = mongoCollection.find(mongoQuery) # execute query to return all records in the the 'training-data' collection    


# In[3]:


# iterate through the mongoDB query result to populate a list
    trainingData = [] # create empty list that we will populate with the query results
    for row in mongoResult:
        inputString = row['input']
        conversationTypeString = row['conversationType']
        conversationSubjectString = row['conversationSubject']
        trainingRow = {"input" : inputString , "conversationType" : conversationTypeString , "conversationSubject" : conversationSubjectString}
        trainingData.append(trainingRow)
    print(trainingData[0]) # smoke-test to ensure the output is formatted correctly


# In[4]:


# Create empty lists that will contain the user input and the complete output component responses
    userInput = []
    conversations = []

# Create empty lists that contain the separated component responses (type and subject)
    conversationTypes = []
    conversationSubjects = []
    
# Create an empty list that will contain the input _and_ the type and subject from the training data
    documents = []

# Specify any characters that will be ignored when the strings are coneverted
    IgnoreWords = ['?']

# Iterate through each row (pattern) in the training-data (MongoDB query results)
    for pattern in trainingData:
        # Tokenize each word within each pattern (row)
        tokenizedInput = nltk.word_tokenize(pattern['input'])
        # Add the tokenized words to the userInput list
        userInput.extend(tokenizedInput)
        # Add tokenized words and types and patterns to our documents corpus
        documents.append((tokenizedInput, pattern['conversationType'], pattern['conversationSubject']))
        # Check to see if the pattern's (row's) type is in the conversations list,
        #  if not, add it. If it is, continue on.
        if pattern['conversationType'] not in conversations:
            conversations.append(pattern['conversationType'])
        # Check to see if the pattern's (row's) subject is in the conversations list,
        #  if not, add it. If it is, continue on.
        if pattern['conversationSubject'] not in conversations:
            conversations.append(pattern['conversationSubject'])
    
        # Check to see if the pattern's (row's) type is in the conversations type list,
        #  if not, add it. If it is, continue on.
        if pattern['conversationType'] not in conversationTypes:
            conversationTypes.append(pattern['conversationType'])
        # Check to see if the pattern's (row's) subject is in the conversations subject list,
        #  if not, add it. If it is, continue on.
        if pattern['conversationSubject'] not in conversationSubjects:
            conversationSubjects.append(pattern['conversationSubject'])

# Stem and lower each input in the userInput list
    userInput = [stemmer.stem(tokenizedInput.lower()) for tokenizedInput in userInput if tokenizedInput not in IgnoreWords]
    userInput = list(set(userInput))

# Smoketest for various objects
    print (len(documents), "documents")
    print (len(userInput), "Unique Word Stems")
    print (len(conversationTypes), "Conversation Types")
    print (len(conversationSubjects), "Conversation Subjects")


# In[5]:


# Create training set list
    trainingSet = []

# Create conversation list 
    completeConversations = []

# Create list of conversation types
    completeTypes = []
    completeSubjects = []

# Create a list [] for conversations that is equal in length to the conversations list
#  and contains only 0s --> [[0],[0],[0]...]
    emptyConversations = [0] * len(conversations)

# Create lists [] for conversation types and subjects that is equal in length to the 
# conversationsTypes and subjects (respectively) and contains only 0s --> [[0],[0],[0]...]
    emptyTypes = [0] * len(conversationTypes)
    emptySubjects = [0] * len(conversationSubjects)

# Create a bag of words that contains all the words within the userInput training data 
    for doc in documents:
    
        # Initialize our bag of words
        wordBag = []
    
        # Get list of tokenized words from the pattern
        wordPattern = doc[0]
    
        # Stem each word in the word pattern
        wordPattern = [stemmer.stem(word.lower()) for word in wordPattern]
    
        # Add content to our bag of words
        for inputWords in userInput:
            wordBag.append(1) if inputWords in wordPattern else wordBag.append(0)

        # Append the content from out wordBag (tokenized, stemmed words) to our traininSet
        #  lists.
        trainingSet.append(wordBag)
        
        # Create conversationRow item that is equal to the [[0],[0].[0]] empty conversation
        #  object that was created earlier.
        conversationRow = list(emptyConversations)
        
        # Checks if the conversation Type is 1 for a particular object and, if so, set that
        #  objects value within the list to one. I.e. [[0],[0],[0]] --> [[0],[0],[1]]
        conversationRow[conversations.index(doc[1])] = 1
        
        # Checks if the conversation Subject is 1 for a particular object and, if so, set 
        #  that objects value within the list to one. I.e. [[0],[0],[0]] --> [[0],[0],[1]]
        conversationRow[conversations.index(doc[2])] = 1
        
        # Append the updated index to the completed conversation list
        completeConversations.append(conversationRow)
    
## ---------------Same as above but separated by components-----------------------##    
    
        typeRow = list(emptyTypes)
        typeRow[conversationTypes.index(doc[1])] = 1
        completeTypes.append(typeRow)
    
        subjectRow = list(emptySubjects)
        subjectRow[conversationSubjects.index(doc[2])] = 1
        completeSubjects.append(subjectRow)


# In[6]:


# Import Necessary Modules for Function
    import numpy as np
    import time

# Computer a non-linear Sigmoid curve (__--)
    def sigmoid(x):
        sigmoidOutput = 1/(1+np.exp(-x))
        return sigmoidOutput

# Convert the output of the signmoid function to its derivative
    def sigmoidDerivative(sigmoidOutput):
        return sigmoidOutput*(1-sigmoidOutput)
 
    def cleanSentence(sentence):
        # Tokenize the words within the input sentence
        sentenceWords = nltk.word_tokenize(sentence)
        # Stem the words within the tokenized setence
        sentenceWords = [stemmer.stem(userInput.lower()) for userInput in sentenceWords]
        return sentenceWords

# Return a binary bag of words [0 or 1] to evaluate whether or not a word exists within
#  a sentence.
    def bagWordCheck(sentence, userInput, show_details=False):
        # Tokenize the sentence
        sentenceWords = cleanSentence(sentence)
        # Create a word bag using the training-data from the user-input
        wordBag = [0]*len(userInput)  
        for sWord in sentenceWords:
            for i,w in enumerate(userInput):
                if w == sWord: 
                    wordBag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(wordBag))

# Evaluate the user's input 
    def think(sentence, showDetails=False):
        x = bagWordCheck(sentence.lower(), userInput, showDetails)
        if showDetails:
            print ("sentence:", sentence, "\n bagWordCheck:", x)
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = sigmoid(np.dot(l0, synapse0))
        # output layer
        l2 = sigmoid(np.dot(l1, synapse2))
        return l2, l2


# In[7]:


# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
    def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(conversations)) )
        np.random.seed(1)

        lastMeanError = 1
        
        # randomly initialize our weights with mean 0
        synapse0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
        synapse2 = 2*np.random.random((hidden_neurons, len(conversations))) - 1
        #synapse2 = 2*np.random.random((hidden_neurons, len(conversationTypes))) - 1
        #synapse2 = 2*np.random.random((hidden_neurons, len(conversationSubjects))) - 1

        prev_synapse0_weight_update = np.zeros_like(synapse0)
        prev_synapse2_weight_update = np.zeros_like(synapse2)

        synapse0_direction_count = np.zeros_like(synapse0)
        synapse2_direction_count = np.zeros_like(synapse2)
        
        for j in iter(range(epochs+1)):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0, synapse0))
                
            if(dropout):
                layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

            layer_2 = sigmoid(np.dot(layer_1, synapse2))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j% 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < lastMeanError:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    lastMeanError = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", lastMeanError )
                    break
                
            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * sigmoidDerivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse2.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * sigmoidDerivative(layer_1)
        
            synapse2_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse0_weight_update = (layer_0.T.dot(layer_1_delta))
        
            if(j > 0):
                synapse0_direction_count += np.abs(((synapse0_weight_update > 0)+0) - ((prev_synapse0_weight_update > 0) + 0))
                synapse2_direction_count += np.abs(((synapse2_weight_update > 0)+0) - ((prev_synapse2_weight_update > 0) + 0))        
        
            synapse2 += alpha * synapse2_weight_update
            synapse0 += alpha * synapse0_weight_update
        
            prev_synapse0_weight_update = synapse0_weight_update
            prev_synapse2_weight_update = synapse2_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse0.tolist(), 'synapse2': synapse2.tolist(),
                   'datetime': now.strftime("%Y-%m-%d %H:%M"),
                   'userInput': userInput,
                   'conversations' : conversations,
                   'conversationTypes': conversationTypes,
                   'conversationSubjects' : conversationSubjects
                  }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)
        print ("saved synapses to:", synapse_file)


# In[8]:


X = np.array(trainingSet)
y = np.array(completeConversations)
y1 = np.array(completeTypes)
y2 = np.array(completeSubjects)

start_time = time.time()

train(X, y, hidden_neurons=5, alpha=0.1, epochs=30000, dropout=False, dropout_percent=0.2)
#train(X, y1, hidden_neurons=5, alpha=0.1, epochs=30000, dropout=False, dropout_percent=0.2)
#train(X, y2, hidden_neurons=5, alpha=0.1, epochs=30000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


# In[9]:


# probability threshold
ERROR_THRESHOLD = 0.75
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse0 = np.asarray(synapse['synapse0']) 
    synapse2 = np.asarray(synapse['synapse2'])

def classify(sentence, showDetails=False):
    resultsT = think(sentence, showDetails)
    for result in range(len(resultsT)):
        results = [[i,r] for i,r in enumerate(resultsT[result]) if r>ERROR_THRESHOLD ] 
        results.sort(key=lambda x: x[1], reverse=True) 
        return_results =[[conversations[r[0]],r[1]] for r in results]
        #return_results =[[conversationSubjects[r[0]],r[1]] for r in results]
        #return_results =[[conversationTypes[r[0]],r[1]] for r in results]
        print(return_results)
        #return return_results


# In[10]:


classify("cltex is broken")

