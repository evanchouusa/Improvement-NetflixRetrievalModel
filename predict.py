#Evan Chou
#COEN169 Project 2
#Due-date: 3/12/2021

import numpy as np #contains functions for linear algebra and matrix that's needed for math formulas
import pandas as pd #for rows and columns

file_path = "/Users/evanchou/Desktop/Coen169 Project 2/" #change to your file path accordingly on your computer 

def train_data():
    with open(file_path+"train.txt", "r") as train_file:
        data = [list(map(int,line.split())) for line in train_file]
    return list(data)

def write_data(data, file_name):
	export_list = []
	i = 0
	for user, movie, rating in data:
		export_string = f"{user} {movie} {rating}\n"
		export_list.append(export_string)

	fp = open(file_path+file_name.replace("test","result"), 'w')
	fp.writelines(export_list)

def test_data(file_name):
    with open(file_path+file_name, "r") as train_file:
        data = [list(map(int,line.split())) for line in train_file]
    return list(data)  

def removingZeros(a, b):
    removea = np.array([])
    removeb = np.array([])
    for first1, second2 in zip(a,b):
        if first1 and second2:
            removea = np.append(removea, first1)
            removeb = np.append(removeb, second2)
    return removea, removeb

def cosineSimilarity(a, b):
    testa, testb = removingZeros(a, b) #call removing zeroes function to remove 0
    #return 0 if either test a or b is all 0
    if len(testa) == 0 or len(testb) == 0:
        return 0.0
    #cosine similarity formula
    numerator = np.dot(a, b) #dot product
    denominator = np.linalg.norm(testa)*np.linalg.norm(testb) #denominator
    return numerator/denominator 

def pearsonCorrelation(a, b):
    testa, testb = removingZeros(a, b) #call removing zeroes function to remove 0
    #return 0 if either test a or b is all 0
    if len(testa) == 0 or len(testb) == 0:
        return 0.0
    #subtract average rating from original rating
    x = np.mean(testa)
    y = np.mean(testb)
    
    testa = testa - x
    testb = testb - y
    
    #cosine similarity formula
    numerator = np.dot(a, b) #dot product
    denominator = np.linalg.norm(testa)*np.linalg.norm(testb) #denominator
    return 0.0 if denominator==0 else numerator/denominator #denominator could be 0 if we subtract weight_b_avg
    
def weightedAverage(sw, r, absValue=False): #sw is similar weights and r is rating
    if np.sum(sw) == 0:
        return 0
    if absValue:
        return np.sum(sw*r)/np.sum(np.absolute(sw)) #Pearson: use absolute value to take into consideration users that are really different
    return np.sum(np.array(sw)*np.array(r))/np.sum(sw) #regular weighted average for basic cosine similarity

def user_row(data, size):
    temp = [0] * size
    for d in data:
        if d[1] >0:
            temp[d[0]-1] = d[1]
    return temp

def rounding(val): #how I am rounding the ratings
    #if the rating is 0, put rating of 3 
    if val == 0:
        return 3
    #if the rating is less than 1, put rating of 1 
    elif val < 1:
        return 1
    #if the rating is greater than 5, put rating of 5
    elif val > 5:
        return 5
    else:
        return round(val)

#user-based cosine similarity
def ub_cosine_similarity(filename):
    trainData = train_data() #this is our training data
    testData = test_data(filename) #this is our testing data
    cols = len(trainData[0]) #cols = columns
    userSimilarity = [] #user similarity
    user_ids = list(set(j[0] for j in testData)) #specific user ID
    result=[]
    for u in user_ids:
        a = user_row([ [j[1],j[2]] for i, j in enumerate(testData) if j[0]==u], cols)
        userSimilarity = []
        for b in trainData:
            userSimilarity.append(cosineSimilarity(a,b))

        for m, index in [[j[1], i] for i, j in enumerate(testData) if j[0]==u and j[2]==0]: # movie with rate 0 in index row
            weight_a, weight_b = removingZeros(userSimilarity, [y[m-1] for x, y in enumerate(trainData)])
            result.append([u, m, rounding(weightedAverage(weight_a, weight_b))])
    write_data(result, filename)   
    
ub_cosine_similarity("test5.txt")
ub_cosine_similarity("test10.txt")
ub_cosine_similarity("test20.txt")

#user-based pearson correlation
def ub_pearson_correlation(filename):
    trainData = train_data() #this is our training data
    testData = test_data(filename) #this is our testing data
    cols = len(trainData[0]) #cols = columns
    userSimilarity = [] #user similarity
    user_ids = list(set(j[0] for j in testData)) #specific user ID
    result=[]
    for u in user_ids:
        a = user_row([ [j[1],j[2]] for i, j in enumerate(testData) if j[0]==u], cols)
        averageRating = np.mean([j[2] for i, j in enumerate(testData) if j[0]==u and j[2]>0])
        userSimilarity = []
        for b in trainData:
            userSimilarity.append(pearsonCorrelation(a,b))
            
        for m, index in [[j[1], i] for i, j in enumerate(testData) if j[0]==u and j[2]==0]: # movie with rate 0 in index row
            weight_a, weight_b = removingZeros(userSimilarity, [y[m-1] for x, y in enumerate(trainData)])
            weight_b_avg = np.mean(weight_b) if len(weight_b)>0 else 0
            weight_b = [x - weight_b_avg for x in weight_b]
            result.append([u, m, rounding(weightedAverage(weight_a, weight_b, True) + averageRating)])
    write_data(result, filename)   
    
ub_pearson_correlation("test5.txt")
ub_pearson_correlation("test10.txt")
ub_pearson_correlation("test20.txt")

#pearson correlation with IUF
def pearson_correlation_IUF(filename):
    trainData = train_data() #this is our training data
    testData = test_data(filename) #this is our testing data
    cols = len(trainData[0]) #cols = columns
    userSimilarity = [] #user similarity
    iuf = [] #IUF empty array in order to implement IUF later
    user_ids = list(set(j[0] for j in testData)) #specific user ID
    result=[]
    
    #IUF log(m/mj)
    m = len(trainData)
    train_t=pd.DataFrame(trainData).T.values.tolist()
    for x in train_t:
        mj=len([r for r in x if r>0])
        iuf.append(np.log(m/mj) if mj else 0.0)
    trainIUF = trainData * np.array(iuf) #multiply original ratings by IUF
    
    for u in user_ids:
        a = user_row([ [j[1],j[2]] for i, j in enumerate(testData) if j[0]==u], cols)
        averageRating = np.mean([j[2] for i, j in enumerate(testData) if j[0]==u and j[2]>0])
        userSimilarity = []
        for b in trainData:
            userSimilarity.append(pearsonCorrelation(a,b))

        for m, index in [[j[1], i] for i, j in enumerate(testData) if j[0]==u and j[2]==0]: # movie with rate 0 in index row
            weight_a, weight_b = removingZeros(userSimilarity, [y[m-1] for x, y in enumerate(trainIUF)])
            weight_b_avg = np.mean(weight_b) if len(weight_b)>0 else 0
            weight_b = [x - weight_b_avg for x in weight_b]
            result.append([u, m, rounding(weightedAverage(weight_a, weight_b, True) + averageRating)])
    write_data(result, filename)   
    
pearson_correlation_IUF("test5.txt")
pearson_correlation_IUF("test10.txt")
pearson_correlation_IUF("test20.txt")

#pearson correlation with case modification
def pearson_correlation_caseModification(filename):
    trainData = train_data() #this is our training data
    testData = test_data(filename) #this is our testing data
    cols = len(trainData[0]) #cols = columns
    userSimilarity = [] #user similarity
    user_ids = list(set(j[0] for j in testData)) #specific user ID
    result=[]
    p = 2.5 #choosing 2.5 for value p
    
    for u in user_ids:
        a = user_row([ [j[1],j[2]] for i, j in enumerate(testData) if j[0]==u], cols)
        averageRating = np.mean([j[2] for i, j in enumerate(testData) if j[0]==u and j[2]>0])
        userSimilarity = []
        for b in trainData:
            userSimilarity.append(pearsonCorrelation(a,b))
        
        #pearson case modification formula
        userSimilarity = userSimilarity * pow(np.array(userSimilarity), p-1)
            
        for m, index in [[j[1], i] for i, j in enumerate(testData) if j[0]==u and j[2]==0]: #movie not rated in index row
            weight_a, weight_b = removingZeros(userSimilarity, [y[m-1] for x, y in enumerate(trainData)])
            weight_b_avg = np.mean(weight_b) if len(weight_b)>0 else 0
            weight_b = [x - weight_b_avg for x in weight_b]
            result.append([u, m, rounding(weightedAverage(weight_a, weight_b, True) + averageRating)])
    write_data(result, filename)   
    
pearson_correlation_caseModification("test5.txt")
pearson_correlation_caseModification("test10.txt")
pearson_correlation_caseModification("test20.txt")

def itemBased_adjustedCosineSimilarity(filename):
    trainData = train_data() #this is our training data
    testData = test_data(filename) #this is our testing data
    cols = len(trainData[0]) #cols = columns
    userSimilarity = [] #user similarity
    user_ids = list(set(j[0] for j in testData)) #specific user ID
    result=[]
    train_avg = [] 
    
    #user average to subtract
    for x in trainData:
        x_sum = sum(x)
        x_count = len([r for r in x if r>0])
        train_avg.append(x_sum/x_count if x_count else 0.0)

    for u in user_ids:
        for m0, u0 in [[j[1], i] for i, j in enumerate(testData) if j[0]==u and j[2]==0]: #unknown movie (rate 0)
            item_sim = []
            for m1, u1 in [[l[1], k] for k, l in enumerate(testData) if l[0]==u and l[2]>0]: #known movie (rate>0)
                a = [y[m0-1]-train_avg[x] for x, y in enumerate(trainData)]
                b = [y[m1-1]-train_avg[x] for x, y, in enumerate(trainData)]
                item_sim.append(cosineSimilarity(a,b))
            weight_a, weight_b = removingZeros(item_sim, [y[2] for x, y in enumerate(testData) if y[0]==u and y[2]>0])
            result.append([u, m0, rounding(weightedAverage(weight_a, weight_b))])
    write_data(result, filename)   
    
itemBased_adjustedCosineSimilarity("test5.txt")
itemBased_adjustedCosineSimilarity("test10.txt")
itemBased_adjustedCosineSimilarity("test20.txt")

def ownAlgorithm(filename):
    trainData = train_data() #this is our training data
    testData = test_data(filename) #this is our testing data
    userSimilarity = []
    data = []
    user = ""
    result=[]
    for tt in testData:
        if user != tt[0]:
            userSimilarity = []
        for tn in trainData:
            if tt[2]>0 and tn[tt[1]-1]>0 and tt[2]>=tn[tt[1]-1]-1 and tt[2]<=tn[tt[1]-1]+1:
                userSimilarity.append(tn)

        if tt[2]==0:
            count=0  
            rating=3.0
            sum=0.0
            for d in userSimilarity:
                if d[tt[1]-1]>0:
                    sum+=d[tt[1]-1]
                    count+=1
            result.append([tt[0], tt[1], round(float(sum)/float(count) if count>0 else 3)]) #put rating 3 if there are 0s
        user=tt[0]
    write_data(result, filename)

ownAlgorithm("test5.txt")
ownAlgorithm("test10.txt")
ownAlgorithm("test20.txt")