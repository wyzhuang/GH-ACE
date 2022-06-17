from pygit2 import Repository
from pygit2 import GIT_SORT_TIME, GIT_SORT_REVERSE
from tqdm import tqdm
import pickle
import time
from multiprocessing import Process, cpu_count
import jpype.imports
from jpype.types import *
import os

def handle_commit_batch(i, start, end,project):
    jpype.startJVM(classpath=['../jpypetest/jars/*'])
    MyClass = JClass('test')
    MyClass.main([])

    repo = Repository('./{}'.format(project))
    commits = []
    try:
        head = repo.references.get('refs/heads/trunk')
        print(head.target)
    except:
        head = repo.references.get('refs/heads/master')
        print(head.target)
    for commit in repo.walk(head.target, GIT_SORT_TIME | GIT_SORT_REVERSE):
        commits.append(commit)
    parse_res = []
    batch_commits = commits[start:end]
    for index, job in tqdm(enumerate(batch_commits[1:])):
        patches = []
        old_commit = batch_commits[index]
        diff = repo.diff(old_commit, job)
        if not (diff.stats.insertions > 5000 or diff.stats.deletions > 5000):
            for patch in diff.deltas:
                if patch.is_binary:
                    continue
                new_file = patch.new_file
                old_file = patch.old_file
                if not (new_file.path.endswith('.java')):
                    continue
                if new_file.id.hex == '0000000000000000000000000000000000000000' or old_file.id.hex == '0000000000000000000000000000000000000000':
                    continue
                # 构建提交前后版本的文件
                try:
                    old_content = repo.read(old_file.id)[1]
                    new_content = repo.read(new_file.id)[1]
                    old_f = open('./old{}.java'.format(str(i)), 'w', encoding='utf-8')
                    new_f = open('./new{}.java'.format(str(i)), 'w', encoding='utf-8')
                    old_str = str(old_content, encoding='utf-8')
                    new_str = str(new_content, encoding='utf-8')
                    old_f.write(old_str)
                    old_f.close()
                    new_f.write(new_str)
                    new_f.close()
                except:
                    continue
                # 使用jpype调用gumtree解析变更
                args = [JString("diff"), JString("old{}.java".format(str(i))), JString("new{}.java".format(str(i)))]
                diff_res = str(MyClass.parseDiff(args))
                patches.append(diff_res)
        parse_res.append([job.hex, patches])

    with open('../data/raw_data/{}{}.pkl'.format(project,str(i)), 'wb') as f:
        pickle.dump(parse_res, f)


if __name__ == '__main__':
    #projects = ["ambari", "ant", "felix", "jackrabbit", "jenkins", "lucene-solr"]
    projects=["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene-solr"]
    for project in projects:
        dirs = "/home/wanghao/Work/TOSEM/data/raw_data/{}".format(project)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            continue

        repo = Repository('./{}'.format(project))
        commits = []
        try:
            head = repo.references.get('refs/heads/trunk')
            print(head.target)
        except:
            head = repo.references.get('refs/heads/master')
            print(head.target)
        for commit in repo.walk(head.target, GIT_SORT_TIME | GIT_SORT_REVERSE):
            commits.append(commit)

        cpus = int(cpu_count()) // 2 - 1
        quote, remainder = divmod(len(commits), cpus)

        processes = [
           Process(target=handle_commit_batch, args=(i, i * quote + min(i, remainder), (i + 1) * quote + min(i + 1, remainder),project)) for i in
           range(cpus)
        ]

        start_time = time.time()
        for process in processes:
           process.start()
        for process in processes:
           process.join()
        end_time = time.time()
        print("done")
        print("process time: {}".format(end_time - start_time))

