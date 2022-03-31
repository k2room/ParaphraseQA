import torch
import json
# for POS tagging
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from sentence_transformers import SentenceTransformer, util

print("GPU :", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available() == True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("-----model loading-----")
tagger = SequenceTagger.load('ner').to(device)   #   load tagger
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

print("-----data loading-----")
# text = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\")"
# text = 'The cat sits outside. Brian is playing guitar. I love pasta. My favorite movie Lalaland is awesome. The cat plays in the garden. Brian watches TV. The movie Notebook is so great. Do you like pizza or pasta?'
with open('datasets/squad20/train-v1.1.json', 'r') as json_f:
    json_data = json.load(json_f)
    data = json_data['data']
total = 0 
similar = 0
similar2 = 0
res = []
print("-----start-----")
for article in data:
    for p in article['paragraphs']:
        text = p['context']
        total += 1
        sentence = Sentence(text)
        tagger.predict(sentence)
        # print(sentence.to_tagged_string())

        sens = sentence.to_tagged_string().split(" . ")
        for i in range(len(sens)):
            sens[i] = sens[i] +"."

        #Compute embeddings
        embeddings = model.encode(sens, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.cos_sim(embeddings, embeddings).to(device)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(len(cosine_scores)-1):
            for j in range(i+1, len(cosine_scores)):
                if cosine_scores[i][j] >= 0.75:
                    similar += 1
                    pairs.append({'index': [i, j], 'score': cosine_scores[i][j], 'id':similar})
                if cosine_scores[i][j] >= 0.9:
                    similar2 += 1

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        for pair in pairs[0:-1]:
            i, j = pair['index']
            res.append({'s1':sens[i], 's2':sens[j], 'score':round(pair['score'].item(), 3), 'id':pair['id']})
            # print(res[-1])

    print("total:",total, "| similar(>=0.75):", similar,"| similar2(>=0.9):",similar2)

with open('datasets/res.json', 'w') as result:
    json.dump(res, result, indent=4)