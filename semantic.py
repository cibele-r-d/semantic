import spacy

nlp = spacy.load("en_core_web_sm")
nlp_md = spacy.load("en_core_web_md")

# +++++++++++++++++++++ Practical Task 1 ++++++++++++++++++++++++++++

# sm model
# Using small model comes with an UserWarning,
# It says the model has no vectors loaded, as a result the similarity method is based on tagger parser and NER.

print("Word individually separated using nlp small")
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.text, word2.text, word1.similarity(word2))
print(word3.text, word2.text, word3.similarity(word2))
print(word3.text, word1.text, word3.similarity(word1))

# md model
print("")
print("Word individually separated using nlp medium")
word1_md = nlp_md("cat")
word2_md = nlp_md("monkey")
word3_md = nlp_md("banana")
print(word1_md.text, word2_md.text, word1_md.similarity(word2_md))
print(word3_md.text, word2_md.text, word3_md.similarity(word2_md))
print(word3_md.text, word1_md.text, word3_md.similarity(word1_md))



print("")
# Comparing words in a string, separated by a space

tokens = nlp("cat apple monkey banana")

print("Word separated by space using nlp medium")
tokens_md = nlp_md("cat apple monkey banana")

for token1_md in tokens_md:
    for token2_md in tokens_md:
        print(token1_md.text, token2_md.text, token1_md.similarity(token2_md))


print("")
# Comparing sentences

sentence_to_compare = "Why my cat is on the car"
sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana",
]

model_sentence_md = nlp_md(sentence_to_compare)


print("Sentences using nlp medium")
for sentence_md in sentences:
    similarity = nlp_md(sentence_md).similarity(model_sentence_md)
    print(sentence_md, "-", similarity)


print("")

words_nlp = nlp_md("science art religion humanity earth 42 galaxy")

for new_word1 in words_nlp:
    for new_word2 in words_nlp:
        print(new_word1.text, new_word2.text, new_word1.similarity(new_word2))


print("")


# +++++++++++++++++++++ Practical Task 2 ++++++++++++++++++++++++++++

# The function movie recommendation has a dictionary with all the movies in the list included the movie already watched.
# Where dictionary key is a string with the movie title and value is a string with the sinopses.


def movie_recommendation(movie_watched):

    movies_dict = {
        "Planet Hulk": "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator.",
        "How to Train Your Dragon: The Hidden World": "When Hiccup discovers Toothless isn't the only Night Fury, he must seek 'The Hidden World', a secret Dragon Utopia before a hired tyrant named Grimmel finds it first.",
        "Reign of the Supermen": "After the death of Superman, several new people present themselves as possible successors.",
        "Suspiria": "A darkness swirls at the center of a world-renowned dance company, one that will engulf the artistic director, an ambitious young dancer, and a grieving psychotherapist. Some will succumb to the nightmare. Others will finally wake up.",
        "Holmes & Watson": "A humorous take on Sir Arthur Conan Doyle's classic mysteries featuring Sherlock Holmes and Doctor Watson.",
        "Summer '03": "A 16-year-old girl and her extended family are left reeling after her calculating grandmother unveils an array of secrets on her deathbed.",
        "The Captain": "In the last moments of World War II, a young German soldier fighting for survival finds a Nazi captain's uniform. Impersonating an officer, the man quickly takes on the monstrous identity of the perpetrators he is trying to escape from.",
        "The Last Boy": "The world at an end, a dying mother sends her young son on a quest to find the place that grants wishes.",
        "A Star Is Born": "A musician helps a young singer and actress find fame, even as age and alcoholism send his own career into a downward spiral.",
        "The Most Wonderful Time of the Year": "Corporate analyst and single mom, Jen, tackles Christmas with a business-like approach until her uncle arrives with a handsome stranger in tow.",
        "Ladies in Black": "Adapted from the bestselling novel by Madeleine St John, Ladies in Black is an alluring and tender-hearted comedy drama about the lives of a group of department store employees in 1959 Sydney.",
    }

    max_similarity = 0
    test = {}
    # Loop through the values(sinopses) to compare with movie_watched
    for title, sinopse in movies_dict.items():
        similarity = nlp_md(movies_dict[movie_watched]).similarity(nlp_md(sinopse))
        # To understand the similarity I decided to get all the similarity values.
        test.update({title: similarity})
        # The conditional excludes the movie with absolute similarity (the movie being compared with itself) and find the highest similarity.
        if similarity != 1 and similarity > max_similarity:
            max_similarity = similarity
            recommendation_title = title

    print("Watch next:", recommendation_title, "\n")
    print(test, "\n")


movie_recommendation("Planet Hulk")

movie_recommendation("Reign of the Supermen")

movie_recommendation("The Most Wonderful Time of the Year")

