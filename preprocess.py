import pickle
import re
from tqdm import tqdm


if __name__ == '__main__':
    projects =["aptoide", "camel", "cassandra", "egeria"]
    cpu = 21
    res = []
    for project in projects:
        print('{} start!\n'.format(project))
        for i in tqdm(range(cpu)):
            with open('./{}/{}{}.pkl'.format(project, project, str(i)), 'rb') as f:
                temp = pickle.load(f)
                for rec in temp:
                    commit_id = rec[0]
                    if rec[1]:
                        actions = []
                        lines = [record.split('\n') for record in rec[1]]
                        for line in lines:
                            for action in line:
                                if not action.startswith('Match') and action:
                                    action_name = re.sub(u"\\(.*?\)", "", action.split(' at ')[0].replace(':', ''))
                                    simple_act = action_name.split(' ')[0] + action_name.split(' ')[1]
                                    actions.append(simple_act)
                        res.append([commit_id, actions])
                    else:
                        res.append([commit_id, []])
        with open('/home/wanghao/Work/TOSEM/data/features/ace/diff/{}.pkl'.format(project,project), 'wb') as f2:
            pickle.dump(res, f2)
        print('{} end!\n'.format(project))