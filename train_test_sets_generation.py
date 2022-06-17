import numpy as np
import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm
import os

def save_glove_transform(project_name,condition, ace, length,dimension,changed_para):
    w2v_model = KeyedVectors.load_word2vec_format(
        '../../../GloVe/models/within-project/{}/{}.txt'.format(dimension,project_name))
    place_holder = np.zeros([len(ace), length, dimension])
    for i in tqdm(range(len(ace))):
        sentence = ace[i]
        if sentence:
            for j in range(min(len(sentence), length)):
                word = sentence[j]
                if word in w2v_model:
                    place_holder[i][j] = w2v_model[word]
                else:
                    place_holder[i][j] = w2v_model['<unk>']
    np.save('/data01/zwy/ACE/data/within-project/{}/{}_{}_X.npy'.format(changed_para,project_name,condition), place_holder)

def generate(length,dimension,parameter):
    if parameter=="dimension":
        changed_para=dimension
    else:
        changed_para = length
    projects = ["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins",
                "lucene-solr"]

    for project in projects:
        print('{} start!\n'.format(project))
        # generate label dict
        label_dict = {}
        data = np.load('../../../../../commit_guru/within-project/{}_train.npy'.format(project), allow_pickle=True)
        for rec in data:
            label_dict[rec[0]] = rec[1]
        # ace feature
        commits = []
        ace = []
        labels = []
        count = 0
        with open('/data01/zwy/ACE/data/diff/{}.pkl'.format(project), 'rb') as f:
            raw_parse_res = pickle.load(f)
            for item in raw_parse_res:
                commit_id = item[0]
                if commit_id in label_dict:
                    count += 1
                    ast_change = item[1]
                    label = label_dict[commit_id]
                    ace.append(ast_change)
                    commits.append(commit_id)
                    labels.append(label)
            np.save('/data01/zwy/ACE/data/within-project/{}/{}_train_Y.npy'.format(changed_para,project),
                            np.array(labels))
            np.save('/data01/zwy/ACE/data/within-project/{}/{}_train_commits.npy'.format(changed_para,project),
                            np.array(commits))
        # Deep Learning

        save_glove_transform(project,  "train", ace, length,dimension,changed_para)

        label_dict = {}
        data = np.load('../../../../../commit_guru/within-project/{}_test.npy'.format(project), allow_pickle=True)
        for rec in data:
            label_dict[rec[0]] = rec[1]
        # ace feature
        commits = []
        ace = []
        labels = []
        count = 0
        with open('/data01/zwy/ACE/data/diff/{}.pkl'.format(project), 'rb') as f:
            raw_parse_res = pickle.load(f)
            for item in raw_parse_res:
                commit_id = item[0]
                if commit_id in label_dict:
                    count += 1
                    ast_change = item[1]
                    label = label_dict[commit_id]
                    ace.append(ast_change)
                    commits.append(commit_id)
                    labels.append(label)
            np.save('/data01/zwy/ACE/data/within-project/{}/{}_test_Y.npy'.format(changed_para,project),
                            np.array(labels))
            np.save('/data01/zwy/ACE/data/within-project/{}/{}_test_commits.npy'.format(changed_para,project),
                            np.array(commits))
        # Deep Learning

        save_glove_transform(project, "test", ace, length,dimension,changed_para)
        print('{} end!\n'.format(project))

if __name__ == '__main__':
    for para in [50]:
        dirs="../data/ACE_feature/within-project/{}".format(para)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        length=150
        dimension=para
        parameter="dimension"
        generate(length, dimension, parameter)




