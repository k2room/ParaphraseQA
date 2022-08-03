import torch
import json
# for POS tagging
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from sentence_transformers import SentenceTransformer, util

def test(text, tagger, model):
    similar = 0
    similar2 = 0
    res = []
    sentence = Sentence(text)
    tagger.predict(sentence)
    # print(tagger.tagsave)
    # print(sentence.get_labels('ner'))
    sens = sentence.to_tagged_string().split(" . ")

    cnt = 0
    ss_tag = []
    for i in range(len(sens)):
        sens[i] = sens[i] +"."
        s_tag=[]
        for p, c in enumerate(sens[i]):
            if c == '<':
                s_tag.append(tagger.tagsave[cnt])
                cnt += 1
        ss_tag.append(s_tag)
    print(sens, ss_tag)

    embeddings = model.encode(sens, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(embeddings, embeddings).to(device)

    pairs = []
    res = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            if cosine_scores[i][j] >= 0.5:
                similar += 1
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j], 'id':similar})
            if cosine_scores[i][j] >= 0.9:
                similar2 += 1
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    for pair in pairs[0:-1]:
        i, j = pair['index']
        res.append({'s1':sens[i], 's2':sens[j], 's1_tag':ss_tag[i], 's2_tag':ss_tag[j], 'score':round(pair['score'].item(), 3), 'id':pair['id']})
    return res

if __name__=="__main__":
    print("GPU :", torch.cuda.is_available(), torch.cuda.device_count())
    if torch.cuda.is_available() == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("-----model loading-----")
    tagger = SequenceTagger.load('ner').to(device)   #   load tagger
    # tagger = SequenceTagger.load('ner-ontonotes').to(device)   #   load tagger
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    print("-----data loading-----")
    text1 = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\")"
    text2 = 'The cat sits outside. Brian is playing guitar. I love pasta. My favorite movie Lalaland is awesome. The cat plays in the garden. Brian watches TV. The movie Notebook is so great. Do you like pizza or pasta?'
    with open('squad1.1/train-v1.1.json', 'r') as json_f:
        json_data = json.load(json_f)
        data = json_data['data']

    print("-----test-----")
    print(test(text1, tagger, model))
    
    total = 0 
    similar = 0
    similar2 = 0
    res = []

    min_len = 15
    max_len = 180

    print("-----start-----")
    for article in data:
        for p in article['paragraphs']:
            text = p['context']
            total += 1
            sentence = Sentence(text)
            tagger.predict(sentence)
            # print(sentence.to_tagged_string())

            sens = sentence.to_tagged_string().split(" . ")

            cnt = 0
            ss_tag = []
            for i in range(len(sens)):
                sens[i] = sens[i] +"."
                s_tag=[]
                for p, c in enumerate(sens[i]):
                    if c == '<':
                        # if c == '<': change tagger form to /<~~>
                        try:
                            s_tag.append(tagger.tagsave[cnt])
                            cnt += 1
                        except IndexError as e: # when there is '<' (not for tag sign)
                            print(e)
                            print(sens[i], tagger.tagsave, cnt)
                            continue
                ss_tag.append(s_tag)
            # for i in range(len(sens)):
            #     if len(sens[i]) > 180 or len(sens[i]) < 15:
            #         sens[i] = ''
            #         ss_tag[i] = ''
            # sens = [v for v in sens if v]
            # ss_tag = [v for v in ss_tag if v]
            # if sens == []:
            #     continue

            #Compute embeddings
            embeddings = model.encode(sens, convert_to_tensor=True)

            #Compute cosine-similarities for each sentence with each other sentence
            cosine_scores = util.cos_sim(embeddings, embeddings).to(device)

            #Find the pairs with the highest cosine similarity scores
            pairs = []
            for i in range(len(cosine_scores)-1):
                for j in range(i+1, len(cosine_scores)):
                    if cosine_scores[i][j] >= 0.75 and cosine_scores[i][j] < 1.00 and (len(sens[i]) < max_len or len(sens[i]) > min_len) and (len(sens[j]) < max_len or len(sens[j]) > min_len):
                        similar += 1
                        pairs.append({'index': [i, j], 'score': cosine_scores[i][j], 'id':similar})
                    if cosine_scores[i][j] >= 0.9 and cosine_scores[i][j] < 1.00:
                        similar2 += 1

            #Sort scores in decreasing order
            pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

            for pair in pairs[0:-1]:
                i, j = pair['index']
                res.append({'s1':sens[i], 's2':sens[j], 's1_tag':ss_tag[i], 's2_tag':ss_tag[j], 'score':round(pair['score'].item(), 3), 'id':pair['id']})
                # print(res[-1])

        print("total:",total, "| similar(>=0.75):", similar,"| similar2(>=0.9):",similar2)

    with open('squad1.1/res2.json', 'w') as result:
        json.dump(res, result, indent=4)
    