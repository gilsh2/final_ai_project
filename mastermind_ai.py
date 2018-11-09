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


class StrategyNode(NamedTuple):
    guess: Code
    IsWinning : bool
    IsLossing : bool
    next_guess: List[None]
    next_guess_lens: List[int]

class StrategyTreeBuilder():
    def Build(Strategy,candidates,combinations) -> StrategyNode:
        
        empty = [None for i in range(len(allresponses))]
        if(len(candidates) == 0):
            return StrategyNode(None,False,True,None,None)
        
        guess = Strategy.ChooseGuess(candidates,combinations)
        Node = StrategyNode(guess,False,False,empty,empty)
        for i in range(len(allresponses)) :           
            if( allresponses[i] == Response(Slots,0)):
                Node.next_guess[i] = StrategyNode(guess,True,False,empty,empty)
            else :   
                candidates_for_resp =  [candidate for candidate in candidates if  Util.IsConsistent(guess, allresponses[i],candidate)]  
                Node.next_guess_lens[i] = len(candidates_for_resp)
                Node.next_guess[i] = StrategyTreeBuilder.Build(Strategy,candidates_for_resp,combinations)
               
                
        return   Node
                

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

class Util :    
     def IsConsistent(guesscode:Code, response:Response,  code:Code)  -> bool :   
        em = Responder.ExactMatches(guesscode,code)
        return response.ExactMatches == em and response.NonExactMatches == (Responder.TotalMatches(guesscode,code) - em)
    
    
         
    
        
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
        slot = random.randint(0,len(candidates)-1)
        return candidates[slot]        
        
 


class MasterMindSolver :    
    __candidates:list
    __Strategy = None
    
    def __init__(self,candidates:list,Strategytree):
         self.__candidates = candidates
         self.__Strategytree = Strategytree
                  
        
    def Play(self,secret:Code) -> Dict[Code, Response]:
        guesshistory:Dict[str, Response] = {}
        Node = self.__Strategytree
        while(True):
            #guess = self.ChooseGuess();     
            if(Node.IsWinning ):
                break;
                
            if(Node.IsLossing):
                raise ValueError('We should not be lossing');     
                       
            guess = Node.guess            
            resp:Response =  Responder.GetResponse(guess,secret)
            guesshistory[str(guess)] = resp             
            Node = Node.next_guess[allresponses.index(resp)]       
                       
        return guesshistory


class MasterMindSolverSimulator :                     
    def Simulate(Strategytree,iterations:int) :     
        themax:int = None 
        thesum:int = 0 
        count=0
        allcodes = GenAllCombinations()          
        for  i in range(iterations) :
             for code  in allcodes :             
                 solver=MasterMindSolver(allcodes,Strategytree)
                 #print(code)
                 solution = solver.Play(code);
                 #print(solution)
                 count = count+1
                 thesum = thesum + len(solution);
                 if( themax == None or len(solution) > themax   ) :
                     themax = len(solution)                 
        
        return  (themax, thesum/count)  


#KnuthTree = StrategyTreeBuilder.Build(RamdomStrategy) 
Tree = StrategyTreeBuilder.Build(RamdomStrategy,allcombinations,allcombinations) 
print("finish building the tree")
solver=MasterMindSolver(allcombinations,Tree)
path=solver.Play([Color.RED,Color.RED,Color.RED,Color.BLUE])
for  i in range(1,100):
    random.shuffle(allcombinations)
    Tree = StrategyTreeBuilder.Build(KnuthStrategy,allcombinations,allcombinations) 
    print(MasterMindSolverSimulator.Simulate(Tree,10))
    





