# -*- coding: utf-8 -*-
"""GLTRipynb

Automatically generated by Colaboratory.

Original file is located at
	https://colab.research.google.com/drive/1Yu-h2xEqTa8NztABEGtNv-BNSx8YlroF
"""

import numpy as np
import torch
import time
import jsonlines
from sklearn.linear_model import LogisticRegression
from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
							BertTokenizer, BertForMaskedLM)
from class_register import register_api
from random import random
import argparse
import transformers
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
from sklearn import metrics
from sklearn.metrics import roc_auc_score

class AbstractLanguageChecker():
	"""
	Abstract Class that defines the Backend API of GLTR.

	To extend the GLTR interface, you need to inherit this and
	fill in the defined functions.
	"""

	def __init__(self):
		'''
		In the subclass, you need to load all necessary components
		for the other functions.
		Typically, this will comprise a tokenizer and a model.
		'''
		self.device = torch.device(
			"cuda" if torch.cuda.is_available() else "cpu")

	def check_probabilities(self, in_text, topk=40):
		'''
		Function that GLTR interacts with to check the probabilities of words

		Params:
		- in_text: str -- The text that you want to check
		- topk: int -- Your desired truncation of the head of the distribution

		Output:
		- payload: dict -- The wrapper for results in this function, described below

		Payload values
		==============
		bpe_strings: list of str -- Each individual token in the text
		real_topk: list of tuples -- (ranking, prob) of each token
		pred_topk: list of list of tuple -- (word, prob) for all topk
		'''
		raise NotImplementedError

	def postprocess(self, token):
		"""
		clean up the tokens from any special chars and encode
		leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
		:param token:  str -- raw token text
		:return: str -- cleaned and re-encoded token text
		"""
		raise NotImplementedError


def top_k_logits(logits, k):
	'''
	Filters logits to only the top k choices
	from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
	'''
	if k == 0:
		return logits
	values, _ = torch.topk(logits, k)
	min_values = values[:, -1]
	return torch.where(logits < min_values,
						 torch.ones_like(logits, dtype=logits.dtype) * -1e10,
						 logits)


@register_api(name='gpt2-l')
class LM(AbstractLanguageChecker):
	def __init__(self, model_name_or_path="gpt2-l"):
		super(LM, self).__init__()
		self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path,cache_dir='/rdata/zainsarwar865/models')
		self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path,cache_dir='/rdata/zainsarwar865/models')
		self.model.to(self.device)
		self.model.eval()
		self.start_token = '<|endoftext|>'
		print("Loaded GPT-2 model!")

	def check_probabilities(self, in_text, topk=40):
		# Process input
		start_t = torch.full((1, 1),
							 self.enc.encoder[self.start_token],
							 device=self.device,
							 dtype=torch.long)
		context = self.enc.encode(
							in_text,
							truncation=True,                      # Sentence to encode.
							max_length = 512,           # Pad & truncate all sentences.
							)
		context = torch.tensor(context,
								 device=self.device,
								 dtype=torch.long).unsqueeze(0)
		
		context = torch.cat([start_t, context], dim=1)
		# Forward through the model
	 
		with torch.no_grad():

			logits = self.model(context)
			# construct target and pred
			yhat = torch.softmax(logits[0][0, :-1], dim=-1)
			y = context[0, 1:]
			# Sort the predictions for each timestep
			sorted_preds = np.argsort(-yhat.data.cpu().numpy())
			# [(pos, prob), ...]
			real_topk_pos = list(
				[int(np.where(sorted_preds[i] == y[i].item())[0][0])
				 for i in range(y.shape[0])])
			real_topk_probs = yhat[np.arange(
				0, y.shape[0], 1), y].data.cpu().numpy().tolist()
			real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

			real_topk = list(zip(real_topk_pos, real_topk_probs))
			# [str, str, ...]
			bpe_strings = [self.enc.decoder[s.item()] for s in context[0]]

			bpe_strings = [self.postprocess(s) for s in bpe_strings]

			# [[(pos, prob), ...], [(pos, prob), ..], ...]
			pred_topk = [
				list(zip([self.enc.decoder[p] for p in sorted_preds[i][:topk]],
						 list(map(lambda x: round(x, 5),
									yhat[i][sorted_preds[i][
											:topk]].data.cpu().numpy().tolist()))))
				for i in range(y.shape[0])]

		pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]
		payload = {'bpe_strings': bpe_strings,
					 'real_topk': real_topk,
					 'pred_topk': pred_topk}
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		return payload

	def postprocess(self, token):
		with_space = False
		with_break = False
		if token.startswith('Ġ'):
			with_space = True
			token = token[1:]
			# print(token)
		elif token.startswith('â'):
			token = ' '
		elif token.startswith('Ċ'):
			token = ' '
			with_break = True

		token = '-' if token.startswith('â') else token
		token = '“' if token.startswith('ľ') else token
		token = '”' if token.startswith('Ŀ') else token
		token = "'" if token.startswith('Ļ') else token

		if with_space:
			token = '\u0120' + token
		if with_break:
			token = '\u010A' + token

		return token



def compute_metrics(preds,labels,y_true,y_scores, model):

	precision, recall, fscore, support = score(labels, preds)
	#print(y_true)
	#print(y_scores)
	auc_test = roc_auc_score(y_true,y_scores)
	return {"model": model,
			"AUC": auc_test,
			"acc": np.mean(preds == labels),
			"precision_machine": precision[1],
			"recall_machine": recall[1],
			"fscore_machine":fscore[1],
			"support_machine":float(support[1]),
			"precision_human":precision[0],
			"recall_human": recall[0],
			"fscore_human":fscore[0],
			"support_human":float(support[0]),
			"y_labels": y_true.tolist(),
			"y_scores": y_scores.tolist(),
			}






def main():

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--test_dataset', # Testing dataset
		default=None,
		type=str,
		required=True,
		 )

	parser.add_argument(
		'--gpt2_xl_gltr_ckpt', # The GLTR-GPT2 model checkpoint for evaluation
		default=None,
		type=str,
		required=True,
		 )

	parser.add_argument( 
		'--gpt2_model', # backend model used
		default=None,
		type=str,
		required=True,
		 )

	parser.add_argument(
		'--return_stat_file', # The file that saves statistical features, you may load this file later to avoid extracting feature again
		default=None,
		type=str,
		required=True,
	)
	
	parser.add_argument(
		'--output_metrics', # The file that saves evaluation results
		default=None,
		type=str,
		required=True,
		 )
	
	args = parser.parse_args()


	histogram_of_likelihoods_test =  []
	labels_test = []
	index_count = 1
	lm = LM(model_name_or_path=args.gpt2_model)
	with jsonlines.open(args.test_dataset, 'r') as src_file:
		for article in src_file:
			# if index_count > 200:
			# 	break
			print("Processing article number: {}".format(index_count))
			index_count += 1
			#print("Processing article number: {}".format(article['id']))
			raw_text = article['text']
			labels_test.append(article['label'])
			payload = lm.check_probabilities(raw_text, topk=5)
			count_10 = 0
			count_100 = 0
			count_1000 = 0
			count_beyond_1000 = 0
			payload = np.asarray(payload['real_topk'])

			for i in range(payload.shape[0]):
				rank = payload[i][0].astype(np.int64)
				if rank <=9:
					count_10 +=1
				elif rank <= 99:
					count_100+=1
				elif rank <= 999:
					count_1000+=1
				else:
					count_beyond_1000+=1   

			histogram_of_likelihoods_test.append([count_10,count_100,count_1000,count_beyond_1000])

	labels_test = np.asarray(labels_test)
	labels_test = np.where((labels_test == 'machine'),1,0)

	# ############## save stat ###################
	return_stat = {
		'histograms': histogram_of_likelihoods_test,
		'labels': labels_test.tolist(),
		'token_len': 512,
	}

	with jsonlines.open(args.return_stat_file, 'w') as return_stat_file:
		return_stat_file.write(return_stat)
		
	# ############## save stat ###################

	loaded_model_test = pickle.load(open(args.gpt2_xl_gltr_ckpt, 'rb'))
	y_preds_test = loaded_model_test.predict(histogram_of_likelihoods_test)
	y_test_probs = loaded_model_test.predict_proba(histogram_of_likelihoods_test)


	acc = np.mean(y_preds_test == labels_test)
	print("Accuracy on gpt2-xl is : {}".format(acc))
	gpt2_xl_metrics = compute_metrics(y_preds_test,labels_test,labels_test, y_test_probs[:,1],args.gpt2_model)

	print(gpt2_xl_metrics)

	

	with jsonlines.open(args.output_metrics, 'w') as output_metrics:
		output_metrics.write(gpt2_xl_metrics)
	




if __name__ == '__main__':
	main()

