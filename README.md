# Ritmo: A Rhythmic Search Engine

## Ritmo is a tool that allows users to create a custom rhythmic search engine from a list of words. The current version is optimized for Spanish, but it can be adapted for any language.

# How it works:

## Ritmo takes a list of words and creates a collection of embeddings for subwords in either xsampa format or IPA. The input words are weighted and encoded via the subword embeddings to create a rhythmic representation of the word. A LSH-based search engine is then able to extract the most similar words from the vocabulary based on a given input word.

## Check out this [Jupyter notebook](https://github.com/s-cafferty-nlp/ritmo_rhythmizer/blob/main/ritmo_demonstration.ipynb) for a more thorough explanation of this project.

```
>>> import ritmo

>>> word_list = ['some', 'long', 'list', 'of', 'words']

>>> R = ritmo.Rhythmizer()

>>> R.add_word_list(word_list)

>>> word_ = 'polvorosa'

>>> sent_ = """Aura sutil su clámide olorosa de verde enredadera en los festones desgarra Melancólicas canciones flébiles surgen de la selva hojosa Tiñe el cielo su bóveda azulosa con lácteas tenuidades de jarrones, y en el ocaso humeantes bermellones del sol la veste esparce esplendorosa"""

>>> R.rhythmize_word_to_sentence(word_,sent_)

[OUTPUT]
0	olorosa	9.000
1	hojosa	7.785
2	esplendorosa	7.425
3	azulosa	6.336
4	ocaso	5.814
...

>>> R.query_all_vocab('Italia')

[OUTPUT]
0	Italia	0.996669
1	Somalia	0.979055
2	Thelia	0.978236
3	Ucrania	0.976562
4	Liberia	0.974564
...
```
