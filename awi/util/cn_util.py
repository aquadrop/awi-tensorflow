#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
import _uniout


reload(sys)
sys.setdefaultencoding("utf-8")


def print_cn(*q):
    print(_uniout.unescape(','.join(q), 'utf8'))


def cn(q):
    return _uniout.unescape(str(q), 'utf8')

def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
    if new_line:
        f.write(b"\n")

    # stdout
    print(s.encode("utf-8"), end="", file=sys.stdout)
    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()

if __name__ == '__main__':
    # with open('../logs/a.log', 'a') as f:
    #     print_out('你好', f)
    #     print_out('你好b', f)
    print_cn('我',u'他','python')
