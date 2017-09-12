#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

path = '../../data/classified/qa/qa2.txt'
outpath = '../../data/classified/qa/qa_small.txt'


def f(path, outpath):
    with open(outpath, 'w+') as out:
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                answer = ''.join(line[1:])
                question = line[0]
                out.write(question)
                out.write('\n')
                out.write(answer)
                out.write('\n')
                out.write('\n')


if __name__ == '__main__':
    f(path, outpath)
