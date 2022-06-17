import pickle
import numpy as np
projects=["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene-solr"]
for project in projects:
    f1 = open("/home/wanghao/Work/TOSEM/data/raw_data/{}/{}.pkl".format(project,project), 'rb')
    dbn_features = []
    raw_parse_res = pickle.load(f1)
    lst_words = raw_parse_res
    train_data = np.load('/home/wanghao/Work/TOSEM/data/commit_guru/within-project/{}_train.npy'.format(project), allow_pickle=True)
    label_dict = {}
    for rec in train_data:
        label_dict[rec[0]] = rec[1]
        # ace feature
    commits = []
    dbn_features = []
    labels = []
    count = 0
    with open('./{}.txt'.format(project), 'w') as f:
        for commit in lst_words:
            if len(commit[1]) != [] and commit[0] in label_dict:
                f.write(" ".join(commit[1]))
                f.write(" ")
        
