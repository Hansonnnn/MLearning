# coding=utf-8
import os


def readEmail():
    files = os.listdir("/Users/hanzhao/Downloads/spam")
    with open("/Users/hanzhao/Downloads/SubObjectSpam.out", "a") as out:
        category = "spam"
        for fname in files:
            with open('/Users/hanzhao/Downloads/spam/' + fname, errors='ignore') as f:
                data = f.readlines()
                for line in data:
                    if line.startswith("Subject:"):
                        line.replace(",", "")
                        print(line)
                        out.write("{0},{1} \n".format(line[8:-1], category))
