# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:57:39 2018

@author: lapid
"""
from enum import Enum
from typing import NamedTuple, Dict, Tuple, Optional, Sequence, List,Set,FrozenSet
import random
import time
import numpy 
import matplotlib.pyplot as plt
from numpy.linalg import inv
import torch
import csv
import json
from numpy import genfromtxt

class Color(Enum):
    RED = 0
    GREEN = 1
    ORANGE = 2
    YELLOW = 3 
    BLUE = 4
    VIOLET =  5
    
Code  = List[Color]
Slots:int = 4 
      
def GenAllCombinations():
    combinations = []
    for col1  in Color:
        for col2  in Color:
            for col3  in Color:
                for col4  in Color:                 
                    combinations.append([col1,col2,col3,col4])
            
    return combinations



class Response(NamedTuple):
    ExactMatches: int
    NonExactMatches: int

      

def Getresponses():
    responses = []
    for i in range(Slots+1) :
        for j in range(Slots+1) :
            if((i + j) <= Slots):
                responses.append(Response(i,j))                
        
            
    return responses

allcombinations = GenAllCombinations()
allresponses = Getresponses()

    
class StrategyNode :
    guess: Code
    IsWinning : bool
    IsLossing : bool
    next_guess: List[None]
    next_guess_lens: List[int]

    def __init__(self, guess:Code,IsWinning:bool,IsLossing:bool):
        self.guess = guess
        self.IsWinning = IsWinning
        self.IsLossing = IsLossing
        self.next_guess = [None] * (len(allresponses) )
        self.next_guess_lens = [int] * (len(allresponses) -1)



class StrategyTreeBuilder():
    def Build(Strategy,candidates,combinations,level=0,pguess=None,pcandidates=None) -> StrategyNode:        
        
        if(len(candidates) == 0):
           return StrategyNode(None,False,True)
     
        guess = Strategy.ChooseGuess(candidates,combinations)
        Node = StrategyNode(guess,False,False)
        for i in range(len(allresponses)) :           
            if( allresponses[i] == Response(Slots,0)):
                Node.next_guess[i] = StrategyNode(guess,True,False)
            else :   
                candidates_for_resp =  [candidate for candidate in candidates if  Util.IsConsistent(guess, allresponses[i],candidate)]  
                Node.next_guess_lens[i] = len(candidates_for_resp)
                Node.next_guess[i] = StrategyTreeBuilder.Build(Strategy,candidates_for_resp,combinations,level+1,guess,candidates)
               
                
        return   Node
                

class Responder : 
    
    def GetExactMatches(secret, guess):
        return len([speg for speg, gpeg in zip(secret, guess) if speg == gpeg])
     
        
    def GetResponse(secret, guess):
        first = Responder.GetExactMatches(secret, guess)
        return first, sum([min(secret.count(j), guess.count(j)) for j in Color]) - first    
            
  
class Util :    
     def IsConsistent(guesscode:Code, response:Response,  code:Code)  -> bool :   
         if(response.ExactMatches != Responder.GetExactMatches(code, guesscode)):
             return False;
         
         return Responder.GetResponse(code,guesscode) == response
    
            
    
        
class KnuthStrategy :
    def __guessScoreKnuth (guess:Code, candidates):        
        maxlen = None     
        
        for resp in allresponses:                     
            newcandidates = [candidate for candidate in candidates if Util.IsConsistent(guess, resp,candidate)]    
           
            if( maxlen == None or  maxlen < len(newcandidates)):
                maxlen = len(newcandidates)               
               
        return maxlen
    
    def __BestKnuth(candidates,combinations) :
        minlen = None
        best = None
        for guess  in combinations:
            score = KnuthStrategy.__guessScoreKnuth(guess,candidates)
            if(minlen == None or minlen > score):
                minlen = score
                best = guess
            
        return (best, minlen)   
    
    def ChooseGuess(candidates,combinations) -> Code :
        guessscore = KnuthStrategy.__BestKnuth(candidates ,candidates);
        guess = guessscore[0]
        guessscore_alt = KnuthStrategy.__BestKnuth(candidates ,combinations);
        if( guessscore_alt[1] <  guessscore[1]):
            guess = guessscore_alt[0]

        return guess
    
class RamdomStrategy :
    def ChooseGuess(candidates , combinations) -> Code :
        epsilon = 0.0000000001
        if(random.random() < epsilon):
            return combinations[ random.randint(0,len(combinations)-1)]
        
        slot = random.randint(0,len(candidates)-1)
        return candidates[slot]        
         

class NNStrategy:
     TheModel:torch.nn.modules.container.Sequential
     
     def __init__(self):
         
        input_dimension = 14
        
        hidden_dimension = 50
        
        # to one output variable.
        output_dimension = 1 
        
        self.TheModel = torch.nn.Sequential(
        torch.nn.Linear(input_dimension, hidden_dimension),
         # A ReLU layer turns inputs into activations nonlinearly   
        torch.nn.Sigmoid(), 
        torch.nn.Linear(hidden_dimension, output_dimension)            
        )                       
             
     def Getfeature (guess:Code, candidates):
         features = [int] * (len(allresponses) -1)  
        
         for i  in range(len(allresponses)):    
            if(allresponses[i] ==  Response(Slots,0)) :
                continue
            
            newcandidates = [candidate for candidate in candidates if Util.IsConsistent(guess, allresponses[i],candidate)] 
            features[i] = len(newcandidates)      
         
         return features
    
     def guessScore (self,guess:Code, candidates):        
        features = NNStrategy.Getfeature(guess,candidates)      
                                  
        features = torch.Tensor(features)
        score = float(self.TheModel(features).detach())
        return score
     
     def BestNNStrategy(self,candidates,combinations) :
        minlen = None
        best = None
        for guess  in candidates:
            score = self.guessScore(guess,candidates)
            if(minlen == None or minlen > score):
                minlen = score
                best = guess
            
        return (best, minlen)       
    
     def ChooseGuess(self,candidates,combinations) -> Code:
               
        guess = self.BestNNStrategy(candidates,candidates)
        return guess[0]
    
    
     def Train(self, filename:str) :
            data = genfromtxt(filename, delimiter=',')
            
           
            #print(data)
            #random.shuffle(data)
            #print(data.shape)             
            print(data.shape)            
            validation_split = 0.3 # Take 30% for validation
            samples = data.shape[0] # Get the number of rows
            validation_samples = validation_split * samples  
            Y = data[int(validation_samples):,0]
            X = data[int(validation_samples):,1:]
            Y_validation = data[:int(validation_samples),0]
            X_validation = data[:int(validation_samples),1:] 
            
            print("length of validation set is ",len(X_validation))

            Xt = torch.Tensor(X)
            # To make Yt match the shape of Yhat, we'll need it to be a slightly different shape
            Yt = torch.Tensor(Y.reshape((len(Y), 1)))
            
            # Convert our numpy arrays to torch tensors
            Xt_validation = torch.Tensor(X_validation)
            # To make Yt match the shape of Yhat, we'll need it to be a slightly different shape
            Yt_validation = torch.Tensor(Y_validation.reshape((len(Y_validation), 1)))                              

          
            
            learning_rate = 0.0000001
            
            loss_fn = torch.nn.MSELoss(size_average=False)
            for t in range(200):
                # Make a prediction
                Yhatt = self.TheModel(Xt)
                
                   
                # Calculate loss (the error of the residual)
                loss = loss_fn(Yhatt, Yt)
                
                if ((t % 10) == 0):     
                     print(loss.item()/len(Yt))
                
              
                # Clear out the "gradient", i.e. the old update amounts
                self.TheModel.zero_grad()
                # Fill out the new update amounts
                loss.backward()
                # Go through and actually update model weights
                with torch.no_grad():
                    for param in self.TheModel.parameters():
                        param -= learning_rate * param.grad
            
         
            
            plt.clf()
            plt.xlabel("Y")
            plt.ylabel("residual")
            res = Yt_validation-self.TheModel(Xt_validation)
            #plt.plot(Yt_validation.cpu().detach().numpy(),res.cpu().detach().numpy(),'ro')
            #plt.show()
            print(loss_fn(Yt_validation,self.TheModel(Xt_validation)))                 
                    
    

class MasterMindSolver :    
    __candidates:list
    __Strategy = None
    __training = list((List[int], int))
    
    def __init__(self,candidates:list,Strategytree):
         self.__candidates = candidates
         self.__Strategytree = Strategytree
                  
    
    def GetTraining(self) :
        return self.__training
    
    
    
    def Play(self,secret:Code,file=None) -> Dict[Code, Response]:
        guesshistory:Dict[str, Response] = {}
        Node = self.__Strategytree
        self.__training = []
        Move:int = 0
        while(True):
            #guess = self.ChooseGuess();     
            if(Node.IsWinning ):
                break;
                
            if(Node.IsLossing):
                raise ValueError('We should not be lossing');     
             
            Move = Move + 1    
            guess = Node.guess    
            
          
            self.__training.append(  (Node.next_guess_lens,Move))           
            resp:Response =  Responder.GetResponse(guess,secret)            
            guesshistory[str(guess)] = resp    
            
                      
            if(resp == Response(Slots,0)):
                break;
                
                                
            Node = Node.next_guess[allresponses.index(resp)]       
        
        for i in range(len(self.__training) ):
            self.__training[i] = (self.__training[i][0], Move - self.__training[i][1])
              
       
        return guesshistory


class MasterMindSolverSimulator :   
    def PersistTraining(trainings,file) :      
        if(file == None):           
            return
                    
        for example in trainings :            
             training = str(example[1]) + "," + ','.join(map(str, example[0]))
             file.write(training+"\n")    
             file.flush()
                 
    def Simulate(Strategytree,iterations:int,filename:str) :     
        themax:int = None 
        thesum:int = 0 
        count=0
        file  = None
        if(filename != None) :
            file = open(filename, "a")
            
       
        allcodes = GenAllCombinations()          
        for  i in range(iterations) :
             for code  in allcodes :             
                 solver=MasterMindSolver(allcodes,Strategytree)
                 #print("code=",code)
                 
                 #if(file != None):                       
                 #    file.write("start game code = " + str(code) +"\n")
                 solution = solver.Play(code,file);
                 
                 trainings = solver.GetTraining()                
                 MasterMindSolverSimulator.PersistTraining(trainings,file)
                 #print(solution)
                 
                 #print(solution)
                 count = count+1
                 thesum = thesum + len(solution);
                 if( themax == None or len(solution) > themax   ) :
                     themax = len(solution)      
        
        if(file != None):                
            file.close()
            
        return  (themax, thesum/count)  

    
            

#KnuthTree = StrategyTreeBuilder.Build(RamdomStrategy) 
#Tree = StrategyTreeBuilder.Build(RamdomStrategy,allcombinations,allcombinations) 
#print("finish building the tree")
#solver=MasterMindSolver(allcombinations,Tree)
#path=solver.Play([Color.RED,Color.RED,Color.RED,Color.BLUE])
#Tree = StrategyTreeBuilder.Build(KnuthStrategy,allcombinations,allcombinations)  
minsteps = None
avgsteps = None    

bestmin = None 
bestavg = None  
  
nn = NNStrategy()    
fname = "tr0.txt";
nn.Train(fname)  
count = 0
while(True):
   
    #Tree = StrategyTreeBuilder.Build(KnuthStrategy,allcombinations,allcombinations)  
   
    nn.Train(fname)  
    random.shuffle(allcombinations)
    Tree = StrategyTreeBuilder.Build(nn,allcombinations,allcombinations) 
    #Tree = StrategyTreeBuilder.Build(RamdomStrategy,allcombinations,allcombinations) 
    count = count +1
    fname = "trcurrent.txt"
    stat= MasterMindSolverSimulator.Simulate(Tree,1,fname)
    print (stat)
    if(minsteps == None or minsteps > stat[0]):
        minsteps =  stat[0]
        bestmin = stat
        
    if(avgsteps == None or avgsteps > stat[1]):
        avgsteps =  stat[1]    
        bestavg = stat
    
    print("progress so far ",(bestmin,bestavg))
    #solver=MasterMindSolver(allcombinations,Tree)
    #solver.Play([Color.RED,Color.RED,Color.RED,Color.RED])






