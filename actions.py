# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"
import json
from typing import Any, Text, Dict, List

from bert_serving.client import BertClient
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
import os
import boto3


bertHost = 'bert'
BUCKET = 'BUCKET'
FAQ = 'FAQ'


class ActionGetFAQAnswer(Action):

    def __init__(self):
        super(ActionGetFAQAnswer, self).__init__()
        self.bc = BertClient(bertHost)
        self.faq, self.encoder, self.encoder_len = encode_faq(self.bc)
        print(self.encoder.shape)

    def find_question(self, query_question):
        query_vector = self.bc.encode([query_question])[0]
        score = np.sum((self.encoder * query_vector), axis=1) / (
                self.encoder_len * (np.sum(query_vector * query_vector) ** 0.5))
        top_id = np.argsort(score)[::-1][0]
        return top_id, score[top_id]

    def name(self) -> Text:
        return "action_get_answer"

    def run(
                self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:
        query = tracker.latest_message['text']
        most_similar_id, score = self.find_question(query)
        if float(score) > 0.93:
            response = self.faq[most_similar_id]['a']
            dispatcher.utter_message(response)
        else:
            response = "Sorry, this question is beyond my ability..."
            dispatcher.utter_message(response)
        return []


def get_faq():
    bucket = None
    faq = 'faq.json'
    if BUCKET in os.environ:
        bucket = os.environ[BUCKET]
    if FAQ in os.environ:
        faq = os.environ[FAQ]
    if bucket and faq:
        s3client = boto3.client('s3')
        try:
            r = s3client.get_object(Bucket=bucket, Key=faq)
            data = json.load(r['Body'])
            with open(os.path.basename(faq), 'w') as f:
                json.dump(data, f)
            return data
        except Exception as ex:
            print(f'Cannot get FAQ from s3://{bucket}/{faq}: {ex}')
    if not os.path.exists(faq):
        faq = os.path.basename(faq)
    if os.path.exists(faq):
        with open(faq, "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def encode_faq(bc):
    faq = get_faq()
    questions = [each['q'] for each in faq]
    print(f'FAQ size {len(questions)}')
    print('Calculating encoder')
    encoder = bc.encode(questions)
    np.save("./questions", encoder)
    encoder_len = np.sqrt(np.sum(encoder * encoder, axis=1))
    np.save("./questions_len", encoder_len)
    print('FAQ encoded')
    return faq, encoder, encoder_len


if __name__ == '__main__':
    bc = BertClient(bertHost)
    encode_faq(bc)
