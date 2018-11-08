# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:57:39 2018

@author: lapid
"""
from enum import Enum
from typing import NamedTuple, Dict, Tuple, Optional, Sequence, List,Set,FrozenSet
import random
import time

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

class Responder :        
    
    def ExactMatches(guesscode:Code,secret:Code)  -> int :
        ret:int = 0
        for i in range(Slots):
            if (guesscode[i] == secret[i]):
                ret = ret +1
        
        return ret        
    
    def TotalMatches(guesscode:Code,secret:Code)  -> int :
        ret:int = 0
        for i in range(Slots):
            if (guesscode[i] in secret):
                ret = ret +1
        
        return ret   
    
    def NonExactMatches(guesscode:Code,secret:Code)  -> int :        
        return Responder.TotalMatches(guesscode,secret) -  Responder.ExactMatches(guesscode,secret)    
    
    
    def GetResponse(guesscode:Code,secret:Code)  -> int :              
        exactmatches = Responder.ExactMatches(guesscode,secret)
        nonexactmatches =Responder.NonExactMatches(guesscode,secret)
        return Response(exactmatches,nonexactmatches)
        

class MasterMindSolver :    
    __candidates:list
    
    def __init__(self,candidates:list):
         self.__candidates = candidates
    
    def ChooseGuess(self)  -> Code :
        slot = random.randint(0,len(self.__candidates)-1)
        return self.__candidates[slot]             
        
    def IsConsistent(guesscode:Code, response:Response,  code:Code)  -> bool :   
        em = Responder.ExactMatches(guesscode,code)
        return response.ExactMatches == em and response.NonExactMatches == (Responder.TotalMatches(guesscode,code) - em)
    
    
    def guessScoreKnuth (guess:Code, candidates):        
        maxlen = None     
        
        for resp in allresponses:                     
            newcandidates = [candidate for candidate in candidates if MasterMindSolver.IsConsistent(guess, resp,candidate)]          
            if( maxlen == None or  maxlen < len(newcandidates)):
                maxlen = len(newcandidates)               
               
        return maxlen
    
    def BestKnuth(candidates,combinations) :
        minlen = None
        best = None
        for guess  in combinations:
            score = MasterMindSolver.guessScoreKnuth(guess,candidates)
            if(minlen == None or minlen > score):
                minlen = score
                best = guess
            
        return (best, minlen)   
    
    def Play(self,secret:Code) -> Dict[Code, Response]:
        guesshistory:Dict[str, Response] = {}
        while(True):
            #guess = self.ChooseGuess();           
            guessscore = MasterMindSolver.BestKnuth(self.__candidates ,self.__candidates);
            guess = guessscore[0]
            guessscore_alt = MasterMindSolver.BestKnuth(self.__candidates ,allcombinations);
            if( guessscore_alt[1] <  guessscore[1]):
                guess = guessscore_alt[0]
            
            
            resp:Response =  Responder.GetResponse(guess,secret)
            guesshistory[str(guess)] = resp
            if(resp  == Response(Slots,0)) :
                break;
        
            newcandidates = [candidate for candidate in self.__candidates if  MasterMindSolver.IsConsistent(guess, resp,candidate)]         
            #print("candidate len after filtering is ",type(newcandidates))
            self.__candidates = newcandidates
            
        return guesshistory


class MasterMindSolverSimulator :                     
    def Simulate(iterations:int) :     
        themax:int = None 
        thesum:int = 0 
        count=0
        allcodes = GenAllCombinations()          
        for  i in range(iterations) :
             for code  in allcodes :             
                 solver=MasterMindSolver(allcodes)
                 print(code)
                 solution = solver.Play(code);
                 print(solution)
                 count = count+1
                 thesum = thesum + len(solution);
                 if( themax == None or len(solution) > themax   ) :
                     themax = len(solution)                 
        
        return  (themax, thesum/count)  

start = time.time()        
solver = MasterMindSolver(allcombinations)                                         
S=solver.Play([Color.GREEN, Color.RED, Color.RED, Color.BLUE])
MasterMindSolver.BestKnuth(allcombinations,  allcombinations)
end = time.time()
print(end-start)
print(S)
print(MasterMindSolverSimulator.Simulate(1))





