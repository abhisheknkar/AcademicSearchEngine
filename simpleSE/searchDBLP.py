__author__ = 'abhishek'

import MySQLdb as mdb
import sys
import cPickle as pickle
import time
import MySQLdb as mdb
import sys
import operator

if __name__ == "__main__":
    time_start = time.time()
    print "Loading Data..."
    con = mdb.connect('localhost', 'abhishek', 'Pass@1234', 'aminerV7');
    cur = con.cursor()
    with open('data/PageRank.pickle', 'rb') as handle:
        pr = pickle.load(handle)
        print "Data loaded..." + str(time.time() - time_start) + " seconds."

    while(1):
        print "Enter a query: "
        query = raw_input()

        query_split = query.split()
        outputPapers = {}
        for word in query_split:
            # cur.execute('SELECT * FROM allPapers WHERE title LIKE "%' + word + '%"')
            cur.execute('SELECT * FROM allPapers WHERE MATCH (title) AGAINST("' + word + '")')
            currPapers = cur.fetchall()
            for paper in currPapers:
                outputPapers[str(paper[0])] = paper[1:]

        pr_query = {}
        for paperID in outputPapers:
            if paperID in pr:
                pr_query[paperID] = pr[paperID]
            else:
                pr_query[paperID] = 0

        pr_query_sorted = sorted(pr_query.items(), key=operator.itemgetter(1), reverse=True)

        count = 1
        size = 10
        output_len = len(pr_query_sorted)
        for paper in pr_query_sorted:
            if count % size == 0:
                print "Displaying results " + str(count+1-size) + " to " + str(count) + " out of " + str(output_len) + ". Press q to stop, any other key to continue."
                response = raw_input()
                if response == 'q':
                    break
            print outputPapers[paper[0]]
            count += 1

        print "Do you wish to enter more queries? (y/n)"
        more = raw_input()
        if more == 'n':
            break


rows = cur.fetchall()
# for row in rows:
#     print row

# print "Finished in " + str(time.time() - time_start) + " seconds."

con.close()