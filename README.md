# Ritmo: A Rhythmic Search Engine

## Ritmo is a tool that allows users to create a custom rhythmic search engine from a list of words. The current version is optimized for Spanish, but it can be adapted for any language.

# How it works:

## Ritmo takes a list of words and creates a collection of embeddings for subwords in either xsampa format or IPA. The input words are weighted and encoded via the subword embeddings to create a rhythmic representation of the word. A LSH-based search engine is then able to extract the most similar words from the vocabulary based on a given input word.

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
1	esplendorosa	7.560
2	hojosa	7.254
3	ocaso	5.841
4	azulosa	5.742
5	bóveda	5.625
...

>>> R.query_all_vocab('Italia')

[OUTPUT]
0	Italia	0.996669
1	Somalia	0.975529
2	Liberia	0.971664
3	Ucrania	0.971496
4	Malasia	0.970687
...
```

## Check out this [Jupyter notebook](https://github.com/s-cafferty-nlp/ritmo_rhythmizer/blob/main/ritmo_demonstration.ipynb) for a more thorough explanation of this product.
