import mastermind_ai_core 

avgsteps =None
bestavg =None

nn = NNStrategy()    
nn.TheModel = torch.load("bestavg_sofar_43865.model")
nn.TheModel.eval()
fname = "tr.txt"
f=open(fname,"w") 
count = 0
while(True):    
    Tree = StrategyTreeBuilder.Build(nn,allcombinations,allcombinations) 
    count = count +1  
    stat= MasterMindSolverSimulator.Simulate(Tree,1,fname)
    print (stat)
   
    random.shuffle(allcombinations)
  
    if(avgsteps == None or avgsteps > stat[1]) :
        avgsteps =  stat[1]    
        bestavg = stat
        torch.save(nn.TheModel,"bestavg_sofar.model")
                  
    print("progress so far ",(bestavg))
    nn.Train(fname)