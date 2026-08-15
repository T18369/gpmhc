Algorithm 1 Forward Loop of Graph-pMHC
1: procedure FORWARDLOOP(peptide, n flank, cflank, mhc, mhe-gene, graph lookup table) #these are the lengths accepted by the model
2: Ensure 9 ≤ Len(peptide) ≤ 30
3: Ensure Len(nflank) ≤ 5
4: Ensure Len(cflank) ≤ 5
5: Ensure Len(mhc) ≤ 43

#Get the lengths to keep track of the number of nodes in the graphs
6: peptide length + Len(peptide)
7: nflank length + Len(nflank)
8: cflank length + Len(nflank)
9: mhc length + Len(mhc)
10: lengths + Concatenate(peptide_length, nflank length, cflank length, mhc length)

#Block 1: Pad the sequences so that positional encoding can be performed
11: peptide3o + Pad(peptide, 30)
12: nflanks + Pad(nflank, 5)
13: cflanks + Pad(cflank, 5)
14: mhcs + Pad(mhc, 43)
15: featuresgs + concatenate(peptide, n flank, cflank, mhc)

#Block 2: Convert tokens to learned vectors
16: features, model-dim + Embedding(features) 
Add a learned vector (one for each position) to the sequence
17: featuresg3, modeldim + PositionalEmbedding(features)

Block 4: Using the lengths and allele gene, get all the possible graphs unpad the feature vector so that it corresponds to the graph nodes
18: graph #graphs, node_features #graphs, #nodes, model.dim, edge-features #graphs #cdges,3 +
LookupGraphs(lengths, features, mhe gene, graph lookup table)

#Block 5: Update the node features with message passing (GAT, GIN, etc.)
19:node-features #graphs, #nodes, model.dim
个 MessagePassing (graph #graphs, node-features #graphs, #nodes, model dim, edge-features #graphs, #edges,3)

#Block 6: Summarize node features into one graph vector (weighted node, attentive GRU, etc.)
20: graph-features#graphs,model-dim + GraphReadout(graph #graphs, node-features #graphs, #nodes, model-dim)

#Block 7: Obtain the logit score for the graphs 
21: Logits#graphs,1 + Linear(graph-features#graphs,model.dim) #Block 8: Determine which graph had the highest logit score, this is the
score for the pMHC
22:
23: Logit + Max (Logits #graphs,1)
return Logit
24: end procedure
