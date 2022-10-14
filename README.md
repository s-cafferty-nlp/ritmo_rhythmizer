## Ritmo: A Rhythmic Search Engine

### Ritmo is a tool that allows users to create a custom rhythmic search engine from a list of words. The current version is optimized for Spanish, but it can be adapted for any language.

## How it works:

### Ritmo takes a list of words and creates a collection of embeddings for subwords in either xsampa format or IPA. The input words are weighted and encoded via the subword embeddings to create a rhythmic representation of the word. A LSH-based search engine is then able to extract the most similar words from the vocabulary based on a given input word.

```
import ritmo

word_list = ['some', 'long', 'list', 'of', 'words']
R = ritmo.Rhythmizer()
R.add_word_list(word_list)
```
