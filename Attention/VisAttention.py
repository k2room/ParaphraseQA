# from transformers import DebertaTokenizer, DebertaForQuestionAnswering
# import torch
# from transformers import logging
# logging.set_verbosity_error()

# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
# model = DebertaForQuestionAnswering.from_pretrained("microsoft/deberta-base")

# question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

# inputs = tokenizer(question, text, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# answer_start_index = outputs.start_logits.argmax()
# answer_end_index = outputs.end_logits.argmax()

# predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
# print(predict_answer_tokens)
# output = tokenizer.decode(predict_answer_tokens)
# print(output)

"""get_attention - Compute representation of attention to pass to the d3 visualization
    Args:
        model: pytorch-transformers model
        model_type: type of model. Valid values 'bert', 'gpt2', 'xlnet', 'roberta'
        tokenizer: pytorch-transformers tokenizer
        sentence_a: Sentence A string
        sentence_b: Sentence B string
        include_queries_and_keys: Indicates whether to include queries/keys in results
    Returns:
      Dictionary of attn representations with the structure:
      {
        'all': All attention (source = AB, target = AB)
        'aa': Sentence A self-attention (source = A, target = A) (if sentence_b is not None)
        'bb': Sentence B self-attention (source = B, target = B) (if sentence_b is not None)
        'ab': Sentence A -> Sentence B attention (source = A, target = B) (if sentence_b is not None)
        'ba': Sentence B -> Sentence A attention (source = B, target = A) (if sentence_b is not None)
      }
      where each value is a dictionary:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape [num_heads, source_seq_len, target_seq_len]
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """

from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show, get_attention
from transformers import BertForQuestionAnswering
import numpy as np
# from transformers import BertTokenizer

model_type = 'bert'
model_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
# model = BertForQuestionAnswering.from_pretrained(model_version)
# tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
sentence_a = "The cat sat on the mat"
sentence_b = "Where is cat?"

att_data = get_attention(model, model_type, tokenizer, sentence_a, sentence_b, include_queries_and_keys=False)['ba']
# print(att_data['attn'])
arr = np.array(att_data['attn'])
print(arr.shape)

# show(model, model_type, tokenizer, sentence_a, sentence_b, layer=2, head=0)


