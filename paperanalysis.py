import gensim
import pandas as pd

# Read data set (~26m titles) from local file
titles = pd.read_csv("id_title.txt", sep="\n", header=None, names=['Titles'], nrows=122000000)

# Preprocess the read data
processed_titles = titles.Titles.apply(gensim.utils.simple_preprocess)
# print(processed_titles)


# Build model
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)

# Build vocab of model from processed text
model.build_vocab(processed_titles, progress_per=1000)

# Train the model
model.train(processed_titles, total_examples=model.corpus_count, epochs=model.epochs)

model.save('122m.model')
