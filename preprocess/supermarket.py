#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json

reload(sys)
sys.setdefaultencoding("utf-8")


dialogue_path = '../data/supermarket/dialogue'
greetings_path = '../data/supermarket/greetings'
qa_path = '../data/supermarket/qa'

path_list = [dialogue_path, greetings_path, qa_path]

supermarket_sessions_path = '../data/supermarket/sm_sessions.txt'


def sm_process(path_list, output):
    with open(output, 'w+') as out:
        for path in path_list:
            with open(path, 'r') as inp:
                for l in inp:
                    line = json.loads(l.strip().decode('utf-8'))
                    question = line['question']
                    answer = line['answer']
                    print(type(answer))
                    if isinstance(answer, list):
                        for a in answer:
                            out.write(question)
                            out.write('\n')
                            out.write(a)
                            out.write('\n')
                            out.write('\n')
                    else:
                        out.write(question)
                        out.write('\n')
                        out.write(answer)
                        out.write('\n')
                        out.write('\n')


if __name__ == '__main__':
    sm_process(path_list, supermarket_sessions_path)
