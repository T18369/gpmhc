### Algorithm 3 Extract possible graphs for a particular pMHC ###
1: procedure LookUPGRAPHs(lengths, features, mhe-gene, graph lookup table)
Create a bool tensor in the shape of the tensors that result from GeneratePossibilitiesTensor, where values are True if they have the
lengths/mhe_gene of the pMHC of interest

2: bool_tensor - GeneratePossibilitiesBoolTensor)
Flatten tensor so that the appearences are in the same order as lookup-df

3: bool_list + Flatten(bool tensor)
Grab graphs (and their corresponding edges) that fulfill the condition

4: graphs, edge features + graph_lookup _table bool _tensor]
Get rid of features that don't have nodes in the graph

5:node-features + UnPad(features)

6:return graph, node features, edge features

7: end procedure
