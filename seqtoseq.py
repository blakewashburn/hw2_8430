# Blake Washburn
# Homework 2 - Deep Learning CPSC 8430
# Due March 26 2021
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import torch
import torch.nn as nn
import torch.optim as optim
import re 
import random

# Read in training and test video features
# Training and Testing paths need to be changed for local execution
trainingPath = "/Users/blakewashburn/Main_Files/Clemson/Spring2021/DeepLearning8430/hw2/MLDS_hw2_1_data/training_data/feat/"
testingPath = "/Users/blakewashburn/Main_Files/Clemson/Spring2021/DeepLearning8430/hw2/MLDS_hw2_1_data/testing_data/feat/"
trainingDataDict = {}
testingDataDict = {}
for filename in os.listdir(trainingPath):
    if filename.endswith('.npy'):
        trainingDataDict[filename] = np.load(trainingPath + filename)
for filename in os.listdir(testingPath):
    if filename.endswith('.npy'):
        testingDataDict[filename] = np.load(testingPath + filename)


# Read in json files for training and testing
# json path need to be changed for local execution
jsonPath = "/Users/blakewashburn/Main_Files/Clemson/Spring2021/DeepLearning8430/hw2/MLDS_hw2_1_data"
with open(jsonPath + "/training_label.json") as trainingLabelsPath:
    trainingLabels = json.load(trainingLabelsPath)
with open(jsonPath + "/testing_label.json") as testingLabelsPath:
    testingLabels = json.load(testingLabelsPath)

# Create language for decoder to use from captions provided in json documents
word2index = {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3}
word2count = {}
index2word = {0: "PAD", 1: "BOS", 2: "EOS", 4: "UNK"}
numOfWords = 4

# Grab each sentence from each set of captions in training set
for index in trainingLabels:
    captionSet = index['caption']
    for sentence in captionSet:
        # Clean each word in the sentence
        for word in sentence.split(' '):
            word = word.lower().strip()
            word = re.sub(r'[^\w\s]', '', word)
            # Add each unique word to dictionary and increment count for repeat words
            if word not in word2index:
                word2index[word] = numOfWords
                word2count[word] = 1
                index2word[numOfWords] = word
                numOfWords += 1
            else:
                word2count[word] += 1

# Grab each sentence from each set of captions in testing set
for index in testingLabels:
    captionSet = index['caption']
    for sentence in captionSet:
        # Clean each word in the sentence
        for word in sentence.split(' '):
            word = word.lower().strip()
            word = re.sub(r'[^\w\s]', '', word)
            # Add each unique word to dictionary and increment count for repeat words
            if word not in word2index:
                word2index[word] = numOfWords
                word2count[word] = 1
                index2word[numOfWords] = word
                numOfWords += 1
            else:
                word2count[word] += 1

# change sentence from list of words to list of indexes
def encodeSentence(inputSentence, word2index):
    outputEncoding = []
    for word in inputSentence.split(' '):
        word = word.lower().strip()
        word = re.sub(r'[^\w\s]', '', word)
        outputEncoding.append(word2index[word])
    return outputEncoding

# encoder-decoder model
class SeqToSeq(nn.Module):
    def __init__(self, vocabSize, batchSize, frameDim, hidden, numOfFrames):
        super().__init__()
        # Parameters for model
        self.batchSize = batchSize
        self.frameDim = frameDim
        self.hidden = hidden
        self.numOfFrames = numOfFrames

        # architecture of model
        self.encodeLSTM = nn.LSTM(hidden, hidden, batch_first=True)
        self.decodeLSTM = nn.LSTM(2*hidden, hidden, batch_first=True)
        self.embedding = nn.Embedding(vocabSize, hidden)
        self.encodeLinear = nn.Linear(frameDim, hidden)
        self.decodeLinear = nn.Linear(hidden, vocabSize)

    def forward(self, videoInput, caption, status):
        # encode video 
        videoInput = videoInput.view(-1, self.frameDim)
        videoInput = self.encodeLinear(videoInput.float())
        videoInput = videoInput.view(-1, self.numOfFrames, self.hidden)
        padding = torch.zeros([self.batchSize, self.numOfFrames-1, self.hidden])
        videoInput = torch.cat((videoInput, padding), 1)   
        videoOutput, state_vid = self.encodeLSTM(videoInput)
        
        # Determine training vs testing status
        if status == "training":        # training
            # set up and embed caption
            caption = self.embedding(caption[:, 0:self.numOfFrames-1])
            padding = torch.zeros([self.batchSize, self.numOfFrames, self.hidden])
            caption = torch.cat((padding, caption), 1)  
            caption = torch.cat((caption, videoOutput), 2)

            # decode caption
            outputCaption, captionState = self.decodeLSTM(caption)
            outputCaption = outputCaption[:, self.numOfFrames:, :]
            outputCaption = outputCaption.view(-1, self.hidden)
            outputCaption = self.decodeLinear(outputCaption)
            return outputCaption

        else:       # Testing
            # decode the input caption  
            padding = torch.zeros([self.batchSize, self.numOfFrames, self.hidden])
            inputCaption = torch.cat((padding, videoOutput[:, 0:self.numOfFrames, :]), 2)
            outputCaption, captionState = self.decodeLSTM(inputCaption)
            
            # create caption to fill with correct caption
            startOfCaption = word2index['<BOS>'] * torch.ones(self.batchSize, dtype=torch.long)
            inputCaption = self.embedding(startOfCaption)
            inputCaption = torch.cat((inputCaption, videoOutput[:, self.numOfFrames, :]), 1)
            inputCaption = inputCaption.view(self.batchSize, 1, 2 * self.hidden)
            outputCaption, captionState = self.decodeLSTM(inputCaption, captionState)
            outputCaption = outputCaption.view(-1, self.hidden)
            outputCaption = self.decodeLinear(outputCaption)
            outputCaption = torch.argmax(outputCaption, 1)
            
            # generate one word at a time and add to caption list, reference vocabulary
            finalCaption = []
            finalCaption.append(outputCaption)
            for i in range(self.numOfFrames-2):
                inputCaption = self.embedding(outputCaption)
                inputCaption = torch.cat((inputCaption, vid_out[:, self.numOfFrames + 1 + i, :]), 1)
                inputCaption = inputCaption.view(self.batchSize, 1, 2 * self.hidden)
                outputCaption, startOfCaption = self.decodeLSTM(inputCaption, startOfCaption)
                outputCaption = cap_out.view(-1, self.hidden)
                outputCaption = self.decodeLinear(outputCaption)
                outputCaption = torch.argmax(outputCaption, 1)
                finalCaption.append(outputCaption)
            return caption


# hyperparameters for model
batchSize = 1
frameDim = 4096
hidden = 200
numOfFrames = 80
model = SeqToSeq(numOfWords, batchSize, frameDim, hidden, numOfFrames)


# Parameters for training
EPOCHS = 1      # would be changed to 200 or so if program properly trained
learningRate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learningRate)
criterion = nn.CrossEntropyLoss()


# =================== Training process =================== #
for i in range(0, EPOCHS): 
    # Get each video to use in training
    for videoID in trainingDataDict:
        videoData = trainingDataDict[videoID]
        
        # Get groundtruth caption using videoID
        for index in trainingLabels:
            if index['id'] + '.npy' == videoID:
                break
        
        # Select caption from set of groundtruth captions corresponding to videoID
        # Manipulate input caption to be proper size
        correctCaption = index['caption'][0]
        videoData = torch.from_numpy(videoData).view(1, 80, 4096)
        correctCaption = encodeSentence(correctCaption, word2index)
        correctCaption = torch.Tensor(correctCaption).long()
        correctCaption = correctCaption.view(1, -1)

        # pass caption and video features to model
        # captionOutput = model(videoData, correctCaption, "training")
        # loss = criterion(captionOutput, correctCaption)
        # loss.backwards()
        # optimizer.step()


# Open output file for testing 
outputFile = open("output.txt", "w")


# =================== Testing process =================== #
# Get each video to use in training
for videoID in testingDataDict:
    videoData = testingDataDict[videoID]
    
    # Get groundtruth caption using videoID
    for index in testingLabels:
        if index['id'] + '.npy' == videoID:
            break

    # Select caption from set of groundtruth captions corresponding to videoID
    # Manipulate input caption to be proper size
    correctCaption = index['caption'][0]
    videoData = torch.from_numpy(videoData).view(1, 80, 4096)
    correctCaption = encodeSentence(correctCaption, word2index)
    correctCaption = torch.Tensor(correctCaption).long()
    correctCaption = correctCaption.view(1, -1) 

    # pass caption and video features to model
    # captionOutput = model(videoData, correctCaption, "testing")
    # TODO: translate captionOutput to english
    outputString = index['id'] + " " + index['caption'][0] + "\n"
    outputFile.write(outputString)

# Evaluate accuracy of captionOutput using Bleu score

        