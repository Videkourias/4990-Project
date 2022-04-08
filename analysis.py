import gensim
from gensim.models import Word2Vec


# Prints a number of possible replacements for a given original sentence with the word at position pos
# replaced with its numReplacements most similar vectors
# original - Array of words
# pos - integer
# numReplacements - integer, 10 by default
def replacement(original, pos, numReplacements=10):
    # Determine replacement words
    replace = original[pos]
    replacements = model.wv.similar_by_key(replace, topn=numReplacements)

    # Print output
    print("%s" % " ".join(str(x) for x in original))
    for i in range(len(replacements)):
        temp = original.copy()
        temp[pos] = '*' + replacements[i][0] + '*'
        print("%s (%.2f%%)" % (" ".join(str(x) for x in temp), replacements[i][1] * 100))


model = Word2Vec.load("26m.model")



# Analyze model

# Word Replace (Sentences taken from processed sentences of data set)
replacement(['formate', 'assay', 'in', 'body', 'fluids', 'application', 'in', 'methanol', 'poisoning'], 8)
replacement(['radiochemical', 'assay', 'of', 'glutathione', 'epoxide', 'transferase', 'and', 'its', 'enhancement', 'by', 'phenobarbital', 'in', 'rat', 'liver', 'in', 'vivo'], 12)
replacement(['effects', 'of', 'on', 'tyrosine', 'hydroxylase', 'activity', 'in', 'central', 'neurons', 'of', 'the', 'rat'], 5)
replacement(['pharmacological', 'properties', 'of', 'new', 'neuroleptic', 'compounds'], 1)
replacement(['effect', 'of', 'human', 'erythrocyte', 'stromata', 'on', 'complement', 'activation'], 0)
replacement(['stabilization', 'of', 'the', 'globular', 'structure', 'of', 'ferricytochrome', 'by', 'chloride', 'in', 'acidic', 'solvents'], 4)

print("\n\n\n")

# Difference in the relation between two pairs of words
pair1 = [['rna', 'dna'], ['glipizide', 'diabetes'], ['ribavirin', 'hepatitis'], ['cycloserin', 'tuberculosis'], ['fentanyl', 'pain'], ['interferon', 'purpura'], ['fluocinolone', 'facial']]
pair2 = [['uracil', 'thymine'], ['carvedilol', 'hypertension'], ['cefotaxime', 'pneumonia'], ['praziquantel', 'schistosomiasis'], ['polymyxin', 'urinary'], ['rifampin', 'leprosy'], ['topiramate', 'spasms']]

for x, y in zip(pair1, pair2):
    print("{}-{} ?= {}-{} (Difference of {}) ".format(x[0], x[1], y[0], y[1], round(abs(model.wv.distance(x[0], x[1])-model.wv.distance(y[0], y[1])), 4)))


