# RAG-Medical-model
基于RAG的私有知识库体系，添加了6万条医疗数据作为私有知识库，旨在提升大模型在医疗相关领域的回复的准确性和可靠性
逻辑：首先对于引入的病例文本进行切割，指保存名称与病症作为chunk，把chunk通过local embedding model 进行向量化处理，存储与chromadb向量数据库中，之后进行归一化，对于主体中的run函数首先包括引入LLM chat API同时，把用户的问题与向量数据库中进行比较，选择优秀的进行正则召回，同时传送给大模型进行润色最后输出,最后在代码中我们添加了测试集（来源于HUGGING FACE），以及对应的测试结果（结果的主要内容包括余弦相似度的差异以及欧式距离的差异）测试的主体主要是本地的大模型和RAG-medical MODEL通过对比两个模型之间的距离差别来体现我们模型在专业领域的优越。
注意：1.想要运行模型首先要内置embedding模型，同时配置LLM的API
2.本代码展示不支持互动，后续会持续更新
3.medical_filtered_60000.txt文件是私有知识库的原始数据
4.集成.py为可以正常运行的model，其余是各部分的模块，仅用于参考不可运行
5.模型为了能够正常运行，在读取文档时做了限制，可以自行更改限制
6.后续可能会优化的有：（1）多轮对话使推断更加严谨（2）增加rerank模型使结果更加优秀（3）增加ui界面和流对话，增加交互性质
  ENGLISH:
Based on the RAG private knowledge base system, 60000 pieces of medical data have been added as a private knowledge base, aiming to improve the accuracy and reliability of the large model's response in the medical related field
Logic: Firstly, the introduced case text is segmented by saving the name and condition as chunks, vectorizing the chunks through a local embedding model, storing them in the Chromadb vector database, and then normalizing them. For the run function in the subject, the LLM chat API is first introduced, and the user's questions are compared with the vector database to select the best ones for regular recall. At the same time, the chunks are sent to the large model for polishing and finally output.Finally, we added a test set (derived from HUGGING FACE) and the corresponding test results in the code (the main contents of the results include the difference in cosine similarity and the difference in Euclidean distance). The main subjects of the test are the local large model and the RAG-medical MODEL. By comparing the distance difference between the two models, we can reflect the superiority of our model in the professional field.
Note: 1. To run the model, you need to first embed the model and configure the LLM API
2. This code does not support interaction and will be continuously updated in the future
The medical_filted-60000.txt file is the raw data of a private knowledge base
4. Integrate. py as a model that can run normally, while the rest are modules from various parts, only for reference and cannot be run
5. In order to run normally, the model has imposed restrictions when reading documents, which can be changed by oneself
6. Possible optimizations in the future include: (1) multiple rounds of dialogue to make inference more rigorous, (2) adding rerank models to improve results, and (3) adding UI interfaces and stream dialogues to enhance interaction properties
