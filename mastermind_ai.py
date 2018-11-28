import mastermind_ai_core 

Tree=StrategyTreeBuilder.Load("knuth_tree_optimized.txt")
s=MasterMindSolverSimulator.Simulate(Tree,1,None)
print(s)