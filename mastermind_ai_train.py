import mastermind_ai_core 


nn = NNStrategy()    
nn.TheModel = torch.load("bestavg_sofar_4394.model")
nn.TheModel.eval()
fname = "rrr2.txt"
count = 0
while(True):   
    #Tree = StrategyTreeBuilder.Build(KnuthStrategy,allcombinations,allcombinations)  
       
    random.shuffle(allcombinations)
    #Tree = StrategyTreeBuilder.Build(KnuthStrategy(),allcombinations,allcombinations) 
    Tree = StrategyTreeBuilder.Build(KnuthStrategy(),allcombinations,allcombinations) 
    count = count +1
  
    stat= MasterMindSolverSimulator.Simulate(Tree,1,fname)
    print (stat)
   
    
  
    if( (avgsteps == None or avgsteps > stat[1]) and  stat[0] == 5  ):
        avgsteps =  stat[1]    
        bestavg = stat
        torch.save(nn.TheModel,"bestavg_sofar.model")
        StrategyTreeBuilder.Save(Tree,"knuth_tree_optimized.txt")
    
   
    
    print("progress so far ",(bestavg))
    #nn.Train(fname)  
    #solver=MasterMindSolver(allcombinations,Tree)
    #solver.Play([Color.RED,Color.RED,Color.RED,Color.RED])