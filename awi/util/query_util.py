#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import csv

import jieba

import cn_util
import re

class QueryUtils:
    static_tokenizer_url = "http://localhost:11415/pos?q="
    remove_tags = ["PN", "VA", "AD", "PU", "SP", "DT"]
    def __init__(self):
        self.remove_tags = ["PN", "VA", "AD"]
        jieba.load_userdict("../data/char_table/ext1.dic")
        self.tokenizer_url = "http://localhost:11415/pos?q="

    def jieba_cut(self, query, smart=True):
        if smart:
            seg_list = jieba.cut(query)
        else:
            seg_list = jieba.cut_for_search(query)
        tokens = []
        for t in seg_list:
            if t:
                tokens.append(t)
        return tokens

    @staticmethod
    def static_jieba_cut(query, smart=True, remove_single=False):
        if smart:
            seg_list = jieba.cut(query)
        else:
            seg_list = jieba.cut(query, cut_all=True)
        tokens = []
        for t in seg_list:
            t = t.replace(' ', '').replace('\t','')
            if t:
                if remove_single:
                    if len(t) == 1:
                        continue
                tokens.append(t)
        return tokens

    def corenlp_cut(self, query, remove_tags=[]):
        try:
            q = query
            r = requests.get(url=self.tokenizer_url + q)
            # purify
            text = []
            for t in r.text.encode("utf-8").split(" "):
                tag = t.split("/")[1]
                word = t.split("/")[0]
                if not tag in remove_tags:
                    text.append(word)
            return text
        except:
            return [query]

    @staticmethod
    def static_corenlp_cut(query, remove_tags=[]):
        q = query
        r = requests.get(url=QueryUtils.static_tokenizer_url + q)
        # purify
        text = []
        for t in r.text.encode("utf-8").split(" "):
            tag = t.split("/")[1]
            word = t.split("/")[0]
            if not tag in remove_tags:
                text.append(word)
        return text

    def pos(self, query, remove_tags=[]):
        try:
            q = query
            r = requests.get(url=self.tokenizer_url + q)
            # purify
            text = []
            for t in r.text.encode("utf-8").split(" "):
                tag = t.split("/")[1]
                if not tag in remove_tags:
                    text.append(t)
            return text
        except:
            return [query]

    @staticmethod
    def static_pos(query, remove_tags=[]):
        try:
            q = query
            r = requests.get(url=QueryUtils.static_tokenizer_url + q)
            # purify
            text = []
            for t in r.text.encode("utf-8").split(" "):
                tag = t.split("/")[1]
                if not tag in remove_tags:
                    text.append(t)
            return text
        except:
            return [query]


    skip_CD = ['一些', '一点', '一些些', '一点点', '一点零', '不少','很多']


    def quant_fix(self, query):
        return QueryUtils.static_quant_fix(query)

    def quant_bucket_fix(self, query):
        return QueryUtils.static_quant_bucket_fix(query)

    @staticmethod
    def static_quant_fix(query):
        query = query.replace('两','二')
        pos = QueryUtils.static_pos(query)
        fixed = False
        new_query = []
        for t in pos:
            word, tag = t.split("/")
            if tag == 'CD' and word not in QueryUtils.skip_CD:
                word = str(cn2arab.cn2arab(word)[1])
                word = re.sub("[^0-9.]", "", word)
                fixed = True
            new_query.append(word)
        return fixed, new_query

    @staticmethod
    def static_quant_bucket_fix(query):
        b, q = QueryUtils.static_quant_fix(query)
        if b:
            new_q = []
            for token in q:
                if token.isdigit():
                    token = QueryUtils.static_fix_money(token)[0]
                new_q.append(token)
            return new_q
        return [query]

    def remove_cn_punct(self, q):
        try:
            return ''.join(self.corenlp_cut(q, remove_tags=['PU']))
        except:
            return q

    @staticmethod
    def static_remove_pu(q):
        # q = re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", q)
        pu = re.compile(ur'[啊|呢|哦|哈|呀|捏|撒|哟|呐|吧|我要|我想|我来|我想要]')
        try:
            return re.sub(pu, '', q.decode('utf-8'))
        except:
            return q

    @staticmethod
    def static_remove_stop_words(q):
        # q = re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", q)
        pu = re.compile(ur'[啊|呢|哦|哈|呀|捏|撒|哟|呐|吧|我要|我想|我来|我想要]')
        try:
            return re.sub(pu, '', q.decode('utf-8'))
        except:
            return q

    @staticmethod
    def static_remove_cn_punct(q):
        try:
            return ''.join(QueryUtils.static_corenlp_cut(q, remove_tags=['PU']))
        except:
            return re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", q.decode("utf8"))

    @staticmethod
    def static_simple_remove_punct(q):
        return re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", q.decode("utf8"))

if __name__ == '__main__':
    qu = QueryUtils()
    jieba.load_userdict("../data/char_table/ext1.dic")
    # qu.process_data('../data/business/intention_pair_q', '../data/business/business_train_v7')
    # # print(QueryUtils.static_remove_cn_punct(u'我在电视上见过你，听说你很聪明啊?'))
    # cn_util.print_cn(qu.quant_bucket_fix('一点钱'))
    # cn_util.print_cn(qu.quant_bucket_fix('我要取1千零1百'))
    # cn_util.print_cn(QueryUtils.static_jieba_cut('紫桂焖大排', smart=True, remove_single=True))
    cn_util.print_cn(QueryUtils.static_remove_stop_words('我来高兴哈'))
    # cn_util.print_cn(','.join(jieba.cut_for_search('南京精菜馆'.decode('utf-8'))))