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

"""Output:
small model:
cat monkey 0.6770565401305904
banana monkey 0.7276309426913784
banana cat 0.6806928240713817

medium model:
cat monkey 0.5929929675536907
banana monkey 0.4041501317354622
banana cat 0.22358827466989753

Observation: the small model makes little difference in the comparison between nouns that have less or more similarity than others 
"""

print("")
# Comparing words in a string, separated by a space

tokens = nlp("cat apple monkey banana")

print("Word separated by space using nlp medium")
tokens_md = nlp_md("cat apple monkey banana")

for token1_md in tokens_md:
    for token2_md in tokens_md:
        print(token1_md.text, token2_md.text, token1_md.similarity(token2_md))

""" Output:
cat cat 1.0
cat apple 0.20368055999279022
cat monkey 0.5929929614067078
cat banana 0.2235882729291916
apple cat 0.20368055999279022
apple apple 1.0
apple monkey 0.2342509925365448
apple banana 0.6646700501441956
monkey cat 0.5929929614067078
monkey apple 0.2342509925365448
monkey monkey 1.0
monkey banana 0.404150128364563
banana cat 0.2235882729291916
banana apple 0.6646700501441956
banana monkey 0.404150128364563
banana banana 1.0

Observations (other than observations made in the lesson):
Cat and apple has a small but higher similarity than banana, 
The order of the factors doesn't alter the results,
It's interesting the fact that monkey has a higher similarity with banana then with cat since biologically speaking the truth is the opposite. 
"""

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


""" Output:
where did my dog go - 0.6300651636725305
Hello, there is my car - 0.8033180053170995
I've lost my car in my car - 0.6787540967120793
I'd like my boat back - 0.5624940476146296
I will name my dog Diana - 0.6491444691953406

Observations:
- Interesting to notice tha "Hello, there is my car" has the highest similarity, much higher than "I've lost my car in my car". 

"""
print("")

words_nlp = nlp_md("science art religion humanity earth 42 galaxy")

for new_word1 in words_nlp:
    for new_word2 in words_nlp:
        print(new_word1.text, new_word2.text, new_word1.similarity(new_word2))

""" Output:
science science 1.0
science art 0.4173954427242279
science religion 0.5334969758987427
science humanity 0.5662701725959778
science earth 0.2616058886051178
science 42 0.07566720247268677
science galaxy 0.1383415162563324
art science 0.4173954427242279
art art 1.0
art religion 0.3020717203617096
art humanity 0.2846473157405853
art earth 0.15678200125694275
art 42 0.019458509981632233
art galaxy 0.031233523041009903
religion science 0.5334969758987427
religion art 0.3020717203617096
religion religion 1.0
religion humanity 0.6512510776519775
religion earth 0.3251609206199646
religion 42 0.019499247893691063
religion galaxy 0.09234324842691422
humanity science 0.5662701725959778
humanity art 0.2846473157405853
humanity religion 0.6512510776519775
humanity humanity 1.0
humanity earth 0.5690114498138428
humanity 42 -0.021569598466157913
humanity galaxy 0.34430110454559326
earth science 0.2616058886051178
earth art 0.15678200125694275
earth religion 0.3251609206199646
earth humanity 0.5690114498138428
earth earth 1.0
earth 42 -0.05975884199142456
earth galaxy 0.46446898579597473
42 science 0.07566720247268677
42 art 0.019458509981632233
42 religion 0.019499247893691063
42 humanity -0.021569598466157913
42 earth -0.05975884199142456
42 42 1.0
42 galaxy 0.007126497104763985
galaxy science 0.1383415162563324
galaxy art 0.031233523041009903
galaxy religion 0.09234324842691422
galaxy humanity 0.34430110454559326
galaxy earth 0.46446898579597473
galaxy 42 0.007126497104763985
galaxy galaxy 1.0

"""
print("")


# +++++++++++++++++++++ Practical Task 2 ++++++++++++++++++++++++++++

# Used Google or Bing! to find the movie titles.
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


# The movie with max similarity was Suspiria, that's interesting because I don't believe Suspiria would be a good suggestion for someone who just watched a Marvel movie.
# Non technical observation: My human intelligence would recommend "Reign of Superman" because super-hero movie, or "Holmes & Watson" or "The Captain".
# test outputs a dictionary with similarity scores:

"""Output:
{'Planet Hulk': 1.0, 
'How to Train Your Dragon: The Hidden World': 0.8520757356824906, 
'Reign of the Supermen': 0.8401837623151783, 
'Suspiria': 0.9089252220278423, 
'Holmes & Watson': 0.5444309494097908, 
"Summer '03": 0.7267662134484585, 
'The Captain': 0.8930368281975917, 
'The Last Boy': 0.9006874124338198, 
'A Star Is Born': 0.833305858292919, 
'The Most Wonderful Time of the Year': 0.8403134504950217, 
'Ladies in Black': 0.7492708832635129}
"""

# Tested with other titles:

movie_recommendation("Reign of the Supermen")


"""Output:
Watch next: The Captain
{'Planet Hulk': 0.8401837623151783, 
'How to Train Your Dragon: The Hidden World': 0.7282553131921449, 
'Reign of the Supermen': 1.0, 
'Suspiria': 0.8638184160251822, 
'Holmes & Watson': 0.5374814137829727, 
"Summer '03": 0.7455480194980266, 
'The Captain': 0.8879236461105849, 
'The Last Boy': 0.7834666875915071, 
'A Star Is Born': 0.7723176751989722, 
'The Most Wonderful Time of the Year': 0.7765269398763539, 
'Ladies in Black': 0.8459861089206402}
"""


movie_recommendation("The Most Wonderful Time of the Year")

""" Output:
Watch next: Suspiria 

{'Planet Hulk': 0.8403134504950217, 
'How to Train Your Dragon: The Hidden World': 0.7608831491891471, 
'Reign of the Supermen': 0.7765269398763539, 
'Suspiria': 0.8780795282624921, 
'Holmes & Watson': 0.5950793588508154, 
"Summer '03": 0.8222191031185758, 
'The Captain': 0.8238221943924654, 
'The Last Boy': 0.8464576558171767, 
'A Star Is Born': 0.875972301798728, 
'The Most Wonderful Time of the Year': 1.0, 
'Ladies in Black': 0.7838370537609176} 
"""
# Conclusion: With training to fine tune the comparison spaCy could give a more precise recommendation.