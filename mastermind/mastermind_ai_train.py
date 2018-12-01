from mastermind.mastermind_ai_core import *

avgsteps =None
bestavg =None

#open and clean training file 
fname = "training.txt"
file=open(fname,"w") 

#get some random tarining examples
for i in range(1,10):
    Tree = StrategyTreeBuilder.Build(RamdomStrategy(),allcombinations,allcombinations) 
    stat= MasterMindSolverSimulator.Simulate(Tree,1,fname)
    print (stat)

#start nn strategy 
nn = NNStrategy()    

#to start  from existing model - uncomment those 2 lines
#nn.TheModel = torch.load("bestavg_sofar.model")
#nn.TheModel.eval()

#train in a loop 
count = 0
statfname = "stat.txt"
statfile=open(statfname,"w") 

while(True):
    count = count+1
    nn.Train(fname)
    Tree = StrategyTreeBuilder.Build(nn,allcombinations,allcombinations) 
    stat= MasterMindSolverSimulator.Simulate(Tree,1,fname)
    print (stat)
    random.shuffle(allcombinations)
  
    #save the best so fat model
    if(avgsteps == None or avgsteps > stat[1]) :
        avgsteps =  stat[1]    
        bestavg = stat
        torch.save(nn.TheModel,"bestavg_sofar.model")

    #every 10 cycles start fresh with the best model so far    
    print("interation = ",count)
    if(count % 100 == 0):
       file=open(fname,"w") 
       nn.TheModel = torch.load("bestavg_sofar.model")
       nn.TheModel.eval()
       Tree = StrategyTreeBuilder.Build(nn,allcombinations,allcombinations) 
       stat= MasterMindSolverSimulator.Simulate(Tree,1,fname)
                    
    statfile.write(str(count) + "," + str(avgsteps) + "\n" )      
    statfile.flush()    
    print("progress so far ",(bestavg))
    