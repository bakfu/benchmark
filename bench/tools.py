import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import nltk        


from bakfu.core.routes import register
from bakfu.core.classes import Processor


import logging
log = logger = logging.getLogger('bench')

result_logger = logging.getLogger('bench_results')

@register('bench.ml1')
class BenchProcess(Processor):   
    '''
    Run a random forest and test quality.    
    '''
    
    init_args = ()
    init_kwargs = ('n_estimators',)
    run_args = ()
    run_kwargs = ()


    def run(self, baf, *args, **kwargs):
        print('BENCH....')
        baf = self.chain = bakfu.Chain(lang=lang)

    def __init__(self, *args, **kwargs):
        super(BenchProcess, self).__init__(*args, **kwargs)
        self.n_estimators = kwargs.get('n_estimators', 50)

    def run(self, caller, *args, **kwargs):
        super(BenchProcess, self).run(caller, *args, **kwargs)
        baf = caller
        data_source = caller.get_chain('data_source')




        language = baf.get('language')

        stop_words = nltk.corpus.stopwords.words(baf.get('language'))
        if language == 'french':
            stop_words.extend([u'les',u'la|le',u'\xeatre',u'@card@',u'avoir',u'@ord@',u'aucun',u'oui',u'non',
                         u'aucune',u'jamais',u'voir',u'n/a',u'ras',u'nil',u'nous',
                         u'chez','nous','quand',u'',u'',u'Les',u'[i]',u'si',u'\xe0','('])

        data_source.get_data()


        labels  = baf.get_chain('targets')
        answers = data_source.get_data()


        #classifier = RandomForestClassifier(n_estimators=self.n_estimators)
        classifier = baf.get_chain("classifier")

        X=baf.data['vectorizer_result']

        score=[0,0]
        NUM_RUNS = 50
        SAMPLE_SIZE = 50

        if os.environ.get('BENCH_FAST','0')=='1':
            #Fast mode...
            NUM_RUNS = 2
            SAMPLE_SIZE = 5

        if len(answers)<SAMPLE_SIZE:
            SAMPLE_SIZE = len(answers)

        for run in range(NUM_RUNS):
            print("run {}".format(run))
            for i in range(SAMPLE_SIZE):
            #for i in range(X.shape[0]):
                print(i)
                #X2 = np.array([X[i].toarray()[0] for j in range(X.shape[0])] if j!=i)
                #labels2 = [labels[j] for j in range(X.shape[0])] if j!=i]
                X2 = np.array([X[j].toarray()[0] for j in range(X.shape[0]) if i!=j])
                labels2 = [labels[j] for j in range(X.shape[0]) if j!=i]

                classifier.fit(X2,np.array(labels2))
                pred = classifier.predict([X.toarray()[i],])[0]

                if pred==labels[i]:
                    score[0]+=1
                else:
                    score[1]+=1

        result_logger.info(score)
        R=score[0]/float(sum(score))
        result_logger.info("Score :   good : \t {} ; \t bad : \t : {}  \t  ratio {}" .format(score[0],score[1],R))

        self._data['score'] = (score[0],score[1],R)

        return self
