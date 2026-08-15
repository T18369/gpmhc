### Algorithm 2 Create the lookup table with all possible graphs ###
1: procedure GENERATELOOKUP TABLE
Make tensors of shape (# possible peptide lengths (21), 
possible nflank lengths (5), 
possible cflank lengths (5), 
possible binding core start positions (30), 
allele genes (3)) one for all possible combinations of peptide
lengths, flank lengths, binding core starting positions, and allele genes

2: peptide lengths, nflank lengths, cflank lengths, positions, mhc_genes + GeneratePossibilitiesTensor)
Generate all the graphs

3: for peptide_length, nflank length, cflank length, position, gene in
zip(peptide_lengths, nflank_lengths, cflank_lengths, positions, genes) do
4: graph, edges-features
5: - GenerateGraph(peptide length,
6: nflank_length, cflank length, position, gene)
7: Append graph to graphs
8: Append edges to alleges
9: end for
10: lookup_table < [graphs, edges]
11: return graph_lookup_table
12: end procedure
