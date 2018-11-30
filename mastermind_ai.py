from mastermind.mastermind_ai_core import *

print("original knuth tree")
Tree=StrategyTreeBuilder.Load("knuth_tree.txt")
s=MasterMindSolverSimulator.Simulate(Tree,1,None)
print(s)

print("optimized knuth tree")
Tree=StrategyTreeBuilder.Load("knuth_tree_optimized.txt")
s=MasterMindSolverSimulator.Simulate(Tree,1,None)
print(s)


print("nn tree")
Tree=StrategyTreeBuilder.Load("nn_best_tree.txt")
s=MasterMindSolverSimulator.Simulate(Tree,1,None)
print(s)
