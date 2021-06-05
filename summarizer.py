import math
import json
import numpy as np

import jieba
jieba.load_userdict('data/dict.txt')
jieba.add_word('个管师')
doc = "个管师"
doc2 = "医师"
pat = "民众"

import summarizer_utils

class Summarizer(object):
    def __init__(self):
        self.pg_turns_idx_map = {}
        self.model = None
        self.top_query = 30
        pass
    
    def load_model(self, path):
        all_turns = []
        last_turn_idx = 0
        
        with open(path, 'r') as f1:
            all_med_qa = json.load(f1)
            
        for med_qa in all_med_qa:
            passage, question, q_opts = self._get_instance(med_qa)
            turns = self._get_turns(passage)
            # record pg idx -> turns idxs mapping
            self.pg_turns_idx_map[med_qa["id"]] = [last_turn_idx, last_turn_idx + len(turns)]
            last_turn_idx = last_turn_idx + len(turns)
            all_turns.extend(turns)
        
        # calcualte tf-tdf
        doc = []
        for turn in all_turns:
            words = jieba.lcut(turn)
            words = summarizer_utils.filter_stop(words)
            doc.append(words)
        self.model = summarizer_utils.BM25(doc)
        
    def _get_instance(self, data):
        passage = data['text']
        question = data['question']['stem']
        opts = data['question']['choices']
        q_opts = []
        for opt in opts:
            q_opts.append(opt['text'])
        return passage, question, q_opts
    
    def _get_turns(self, passage):
        seg_list = jieba.lcut(passage)
        turns = []
        tmp_str = ""
        for idx, token in enumerate(seg_list):
            if (token == doc or token == doc2)and idx != 0:
                turns.append(tmp_str)
                tmp_str = token
            else:
                tmp_str += token
        
        if len(turns) == 0:
            print("ENCOUNTER UNKNOWN DOCOTOR TOKEN!")
        
        return turns
    
    def get_summary_local(self, passage, qa_instancce_idx, queries, max_seq_len):
        doc = []
        
        # calcualte tf-tdf
        turns = self._get_turns(passage)
        for turn in turns:
            words = jieba.lcut(turn)
            words = summarizer_utils.filter_stop(words)
            doc.append(words)
        s = summarizer_utils.BM25(doc)

        # calculate similarity using BM25 
        all_top_turn_idxs = []
        for q in queries:
            sim_res = np.array(s.simall(q))
            top_turns_idxs = sim_res.argsort()[-self.top_query:][::-1]
            all_top_turn_idxs.append(top_turns_idxs)
        
        # select top queries to join a summary that fits max_seq_len
        seleted_turns_idx = self._select_top_turns(turns, all_top_turn_idxs, max_seq_len)
        joined_rel_turns = "".join([turns[idx] for idx in sorted(seleted_turns_idx.keys())])
        return joined_rel_turns

    def get_summary(self, passage, qa_instancce_idx, queries, max_seq_len):
        turns = self._get_turns(passage)
        try:
            start, end = self.pg_turns_idx_map[qa_instancce_idx]  # [start_i, end_i]
        except:
            print("CAN NOT FIND TURNS MAPPING!")
            return None
        
        # calculate similarity using BM25 
        search_turn_idxs = [i for i in range(start, end)]
        all_top_turn_idxs = []
        for q in queries:
            sim_res = np.array(self.model.sim_some(q, search_turn_idxs))
            top_turns_idxs = sim_res.argsort()[-self.top_query:][::-1]
            all_top_turn_idxs.append(top_turns_idxs)
        
        # select top queries to join a summary that fits max_seq_len
        seleted_turns_idx = self._select_top_turns(turns, all_top_turn_idxs, max_seq_len)
        summary = "".join([turns[idx] for idx in sorted(seleted_turns_idx.keys())])
        
        return summary
    
    def _select_top_turns(self, turns, all_top_turn_idxs, max_seq_len):
        remain_len = max_seq_len
        seleted_turns_idx = {}  # key: turn_idx
        opt_i = 0
        top_query_num = min(self.top_query, len(all_top_turn_idxs[0]))
        for top_i in range(top_query_num):  # loop top turns...
            for opt_turns in all_top_turn_idxs:  # in each option's top turns
                if seleted_turns_idx.get(opt_turns[top_i], -1) != -1:  # turn already selected
                    continue
                remain_len -= len(turns[opt_turns[top_i]])
                if remain_len < 0:
                    break
                seleted_turns_idx[opt_turns[top_i]] = 1
            if remain_len < 0:
                break
        return seleted_turns_idx

if __name__ == '__main__':
    s = Summarizer()
    dialogue_path = "data/sample_cn.json"
    s.load_model(dialogue_path)
    with open(dialogue_path, 'r') as f1:
        all_med_qa = json.load(f1)

    qa_instancce_idx = 5
    top_idx= 20  # select top k results from each option query
    max_seq_len = 1024 - 3 - 50
    passage, question, q_opts = s._get_instance(all_med_qa[qa_instancce_idx])
    
    print(passage)
    print(question)
    print(q_opts)
    
    summary = s.get_summary(passage, qa_instancce_idx, q_opts, max_seq_len)
    print(summary)