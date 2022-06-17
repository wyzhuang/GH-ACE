# GH-ACE
KBS_Just-in-time defect prediction based on AST change embedding
At present, the source code files uploaded by this warehouse are not sorted out, and will be further supplemented later.

1. "corpus" directory: The corpus of the project is stored in the "corpus" directory.

2. "data" directory: Relevant data is stored in the "data" directory, and "tradition_data" stores traditional features, "raw_ data" stores the extracted ace change node, "ACE_ data" stores the digital vector of the pre processed AST change node and the related submission information.

3. "jars" directory: The jar files related to gumtree are stored in the "jars" directory

4. "model" directory: The training model is stored in the "model" directory

5. "project" directory: The project history data is stored in the "project" directory. The "ace_pygit2_test_multi.py" file is used to extract the AST change token from the project.

6.preprocess.py file for processing raw_ Raw data in data

7.train_ test_ sets_ generation.py file is used to convert the original data in the "data/raw_ data" directory into training sets and test sets that can be learned directly, and the generated files are stored in "data/ACE_data" directory

中文说明：
1.corpus目录下存放的是项目的语料库。
2.data目录下存放的是相关数据，tradition_data存放的是传统特征，raw_data存放的是提取出来的ACE变更节点，ACE_data存放的是预处理好的AST变更节点的数字向量，以及相关的提交信息。
3.jars目录下存放的是gumtree的相关jar文件
4.model目录下存放的是训练模型
5.project目录中存放的是项目历史数据，“ACE_pygit2_test_multi.py”文件用于从项目中提取AST更改令牌。
6.preprocess.py文件用于处理raw_data中的原始数据
7.train_test_sets_generation.py文件用于将data/raw_data中的原始数据转换成可直接学习的训练集和测试集，生成的文件存放在data/ACE_data目录下