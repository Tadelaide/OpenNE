from __future__ import print_function
import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from .classify import Classifier, read_node_label
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from pandas import DataFrame
from sklearn.metrics import pairwise_kernels


class _LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, negative_ratio=5, order=3):
        self.cur_epoch = 0
        self.order = order
        self.g = graph
        self.node_size = graph.G.number_of_nodes()

        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

        self.gen_sampling_table()
        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])
        self.lr = tf.Variable(2e-3, dtype = tf.float32)

        #self.embeddings 为[self.node_size 总共的 node 数量, self.rep_size 所嵌入的大小] 随机生成的embeddings，即 2407*128
        cur_seed = random.getrandbits(32)

        self.embeddings = tf.get_variable(name="embeddings"+str(self.order), shape=[self.node_size, self.rep_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))

        self.context_embeddings = tf.get_variable(name="context_embeddings"+str(self.order), shape=[self.node_size, self.rep_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        
        #一个批次1000
        self.embeddingNodeSize = tf.constant(self.node_size, tf.float32)

        # self.h_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.h), 1)
        # self.t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.t), 1)
        # self.t_e_context = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.context_embeddings, self.t), 1)
        #从 embedding 中将对应的行列取出
        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        #self.h_e 这个 tensor 是所有edge(x,y), x 所组成的 tensor 从embeddings 里取出的随机 128维度的tensor, 他的 shape 神奇的是2*128的一个tensor
        self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
        self.t_e_context = tf.nn.embedding_lookup(self.context_embeddings, self.t)
        
        '''
        定义 loss 在这里定义的
        '''
        #Baseline
        # self.second_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        # self.first_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))

        #Baseline2
        # self.second_loss = tf.reduce_mean(1 - tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        # self.first_loss = tf.reduce_mean(1 - tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        #New
        #softmax, 0.35
        # self.second_loss = -tf.reduce_mean(tf.nn.softmax(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        #改变sign 的位置 0,40
        # self.first_loss = -tf.reduce_mean(self.sign*tf.log_sigmoid(tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        # self.first_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        # 牛逼的0.55
        # self.first_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = -self.sign, logits = tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        
        # self.first_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = -self.sign, logits = tf.reduce_mean(tf.multiply(self.h_e, self.t_e), axis=1)))
        # self.second_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = -self.sign, logits = tf.reduce_mean(tf.multiply(self.h_e, self.t_e_context), axis=1)))

        # (tf.nn.sigmoid_cross_entropy_with_logits(labels = self.h_e, logits = self.t_e_context)
        # tf.nn.sigmoid_cross_entropy_with_logits(labels = self.h_e, logits = self.t_e)

        #Origin kernel
        # self.kernal_first = tf.multiply(self.h_e, self.t_e)
        # self.kernal_second = tf.multiply(self.h_e, self.t_e_context)

        #kernel
        self.kernal_first = self.compute_cosine(self.h_e, self.t_e)
        self.kernal_second = self.compute_cosine(self.h_e, self.t_e_context)
        
        self.second_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(self.kernal_second, axis=1)))
        self.first_loss = -tf.reduce_mean(tf.log_sigmoid(self.sign*tf.reduce_sum(self.kernal_first, axis=1)))



        #loss function
        #pairwise rbf distance
        '''
        # self.probability_f = tf.reshape(tf.sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)),[-1, tf.shape(self.h)[0]])
        # self.probability_s = tf.reshape(tf.sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)),[-1, tf.shape(self.h)[0]])

        # self.normTensor_f = tf.zeros_like(self.probability_f)+1.0
        # self.normTensor_s = tf.zeros_like(self.probability_s)+1.0

        # self.first_loss = tf.reduce_mean(self.compute_rbf(self.normTensor_f ,self.probability_f))

        # self.second_loss = tf.reduce_mean(self.compute_rbf(self.normTensor_s ,self.probability_s))
        '''
        # self.probability_f = tf.reshape(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)),[-1, tf.shape(self.h)[0]])
        # self.probability_s = tf.reshape(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)),[-1, tf.shape(self.h)[0]])

        # self.normTensor_f = tf.zeros_like(self.probability_f)+1.0
        # self.normTensor_s = tf.zeros_like(self.probability_s)+1.0

        # self.first_loss = tf.reduce_mean(self.compute_cosine_loss(self.normTensor_f ,self.probability_f))

        # self.second_loss = tf.reduce_mean(self.compute_cosine_loss(self.normTensor_s ,self.probability_s))
        #2*128维度，通过tf.reduce_sum 变成了1*128维度



        #Newnew
        
        #继续
        # self.first_loss = self.compute_mmd(self.h_e, self.t_e)
        # self.second_loss = self.compute_mmd(self.h_e, self.t_e_context)


        
        


        if self.order == 1:
            self.loss = self.first_loss
        else:
            self.loss = self.second_loss
        #Baseline
        optimizer = tf.train.AdamOptimizer(1e-3)
        #修改之后的结果
        # optimizer = tf.train.AdamOptimizer(self.lr)
        #加入激活函数,有提高
        self.train_op = optimizer.minimize(self.loss)


    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            feed_dict = {
                self.h : h,
                self.t : t,
                self.sign : sign,
            }
            #修改
            # self.sess.run(tf.assign(self.lr, 2e-3*( 0.95 ** self.cur_epoch)))
            _, cur_loss = self.sess.run([self.train_op, self.loss],feed_dict)


            # print(self.sess.run(self.compute_rbf(self.normTensor_f ,self.probability_f),feed_dict),'nmnmmnmnmnmnmnmn')
            # print(self.sess.run(tf.shape(self.compute_rbf(self.normTensor_f ,self.probability_f)),feed_dict),'nmnmmnmnmnmnmnmn')

            # print(self.sess.run(tf.shape(self.t)[0],feed_dict),'ttttttttttttttttttttttttttttt')
            # print((self.sess.run(self.t,feed_dict)),'ttttttttttttttttttttttttttttt')

            # print(self.sess.run(tf.shape(tf.nn.embedding_lookup(self.embeddings, self.t)),feed_dict),'qweqweqweqweqweq')
            # print(self.sess.run(tf.nn.embedding_lookup(self.embeddings, self.t),feed_dict),'qweqweqweqweqweq')

            # print(self.sess.run(tf.shape(tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1))),feed_dict),'zxczxczxczxczxczxc')
            # print(self.sess.run((tf.log_sigmoid(self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1))),feed_dict),'zxczxczxczxczxczxczxczxc')


            # print(self.sess.run(tf.shape(tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)),feed_dict),'.......................')


            sum_loss += cur_loss
            batch_id += 1
        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1

    def batch_iter(self):
        look_up = self.g.look_up_dict

        table_size = 1e8
        numNodes = self.node_size
        #取出来的是edge(1,2)，一堆边的向量
        edges = [(look_up[x[0]], look_up[x[1]]) for x in self.g.G.edges()]

        data_size = self.g.G.number_of_edges()
        edge_set = set([x[0]*numNodes+x[1] for x in edges])
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # positive or negative mod, negative sampleing
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)

        while start_index < data_size:
            if mod == 0:
                sign = 1.
                h = []
                t = []
                for i in range(start_index, end_index):
                    if not random.random() < self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(h)):
                    t.append(self.sampling_table[random.randint(0, table_size-1)])

            yield h, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index+self.batch_size, data_size)

    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        numNodes = self.node_size

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes) # out degree

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

        data_size = self.g.G.number_of_edges()
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"] for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"]*data_size/total_sum for edge in self.g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1


    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self.sess)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors

    def compute_kernel(self,x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def compute_mmd(self,x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def compute_rbf(self,x,y):
        gamma = tf.constant(-1.0)
        sq_vec = tf.multiply(2., tf.matmul(tf.transpose(x), (y)))
        return tf.exp(tf.multiply(gamma, tf.abs(sq_vec)))
    

        
    def compute_cosine(self, x, y):
        normalize_a = tf.nn.l2_normalize(x,1)        
        normalize_b = tf.nn.l2_normalize(y,1)
        return tf.multiply(normalize_a,normalize_b)
        
    def compute_cosine_loss(self, x, y):
        return tf.losses.cosine_distance(tf.nn.l2_normalize(x,0), tf.nn.l2_normalize(y,0))

class LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, epoch=10, negative_ratio=5, order=3, label_file = None, clf_ratio = 0.5, auto_save = True):
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        self.resultRank = []
        self.best_label = None
        if order == 3:
            self.model1 = _LINE(graph, rep_size/2, batch_size, negative_ratio, order=1)
            #graph 是输入的图，rep_size 是 represent 的大小，batch 是批次大小
            self.model2 = _LINE(graph, rep_size/2, batch_size, negative_ratio, order=2)
            for i in range(epoch):
                self.model1.train_one_epoch()
                self.model2.train_one_epoch()
                if label_file:
                    self.get_embeddings()
                    X, Y = read_node_label(label_file)
                    print("Training classifier using {:.2f}% nodes...".format(clf_ratio*100))
                    clf = Classifier(vectors=self.vectors, clf=LogisticRegression())
                    result = clf.split_train_evaluate(X, Y, clf_ratio)
                    self.resultRank.append(result['macro'])
                    if result['macro'] > self.best_result:
                        self.best_result = result['macro']
                        #label Tade add
                        self.best_label = Y
                        if auto_save:
                            self.best_vector = self.vectors


        else:
            self.model = _LINE(graph, rep_size, batch_size, negative_ratio, order=self.order)
            for i in range(epoch):
                self.model.train_one_epoch()
                if label_file:
                    self.get_embeddings()
                    X, Y = read_node_label(label_file)
                    print("Training classifier using {:.2f}% nodes...".format(clf_ratio*100))
                    clf = Classifier(vectors=self.vectors, clf=LogisticRegression())
                    result = clf.split_train_evaluate(X, Y, clf_ratio)
                    self.resultRank.append(result['macro'])
                    if result['macro'] > self.best_result:
                        self.best_result = result['macro']
                        #label Tade add
                        self.best_label = Y
                        if auto_save:
                            self.best_vector = self.vectors

        self.get_embeddings()
        if auto_save and label_file:
            self.vectors = self.best_vector

        # plt.plot(self.resultRank,'bx')
        # plt.plot(self.resultRank,'-r')
        # plt.grid(True)
        # plt.title('Wiki')
        # plt.savefig('test.png')
        # plt.show()

        # plt.scatter(self.best_vector,self.best_label,c=self.best_label,s=20,marker='o')
        # plt.show()

        # fig = plt.figure(figsize=(8, 8))
        # n_components = 2
        # tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        # Y = tsne.fit_transform(DataFrame(self.best_vector).T)
        # color = (np.array(self.best_label).T)[0]
        # color_int = np.array([int(item) for item in color])
        # colorMax, colorMin = color_int.max(), color_int.min()
        # color = (color_int - colorMin) / (colorMax - colorMin)
        # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        # plt.show()


    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = {}
        if self.order == 3:
            vectors1 = self.model1.get_embeddings()
            vectors2 = self.model2.get_embeddings()
            for node in vectors1.keys():
                self.vectors[node] = np.append(vectors1[node], vectors2[node])
        else:
            self.vectors = self.model.get_embeddings()

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

        ffout = open('result.txt','w')
        ffout.write("{} {}\n".format(node_num, self.rep_size))
        for item in self.best_label:
            ffout.write("{}\n".format(item[0]))
        ffout.close()


