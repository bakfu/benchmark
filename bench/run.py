#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os

import itertools
from itertools import product
from itertools import izip_longest  
import copy
from glob import glob

import six
from six import string_types, iteritems

import yaml
import json
import jsonpickle
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)


import numpy as np
from scipy.optimize import minimize, basinhopping
import deepdiff

import logging

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger('bench')

import bakfu
import tools
if six.PY2:
    from functools32 import lru_cache
else:
    from functools import lru_cache
    


# Print results out to a file
# TODO: send results to a DB
result_logger = logging.getLogger('bench_results')
logfile = logging.FileHandler('results.log')
formatter = logging.Formatter('%(message)s')
logfile.setFormatter(formatter)
result_logger.addHandler(logfile ) 
result_logger.setLevel(logging.INFO)

            
def filterFor( path, element ):
    print(path,element)

def walkDict(vDict, func, caller, context, path=()):
    '''
    Process a dictionnary recursively.
    
    Looks for string values starting with special characters and process them accordingly : 
      % : replace variable
          ex : %name will be replaced by the name variable defined in the benchmark parameters
      !: evaluates the expression
          ex : !range(1,4)
    '''
    for k,v in iteritems(vDict):
        # key has type '%var_name'
        if isinstance(k,str):
            if k[0]=='%':
                old_k = k
                k = context[k[1:]]
                vDict.pop(old_k)
                vDict[k] = v
        if type(vDict[k]) == dict:
            walkDict( vDict[k], func, caller, context, path+(k,) )    
        elif isinstance(v,str):
            if len(v)>0 and v[0]=='%':
                #func(vDict,k,v, caller, context)
                vDict[k] =  context[v[1:]]
            if len(v)>0 and v[0]=='!':
                #func(vDict,k,v, caller, context)
                vDict[k] =  eval(v[1:])
        elif isinstance(v,list):
            for idx, elt in enumerate(v):
                if isinstance(elt,string_types):
                   if elt[0] == "%":
                       v[idx] = copy.deepcopy(context[elt[1:]])
                       walkDict( v[idx], func, caller, context, path+(k,) )    
                elif type(elt) == dict:
                    walkDict( elt, func, caller, context, path+("/list/",) )                
        else:
            pass

def replacer(vDict,k,v,caller,context):
    vDict[k] =  context[v[1:]]



class Iterator(object):
    '''
    Returns  an iterator over a list of values.
    '''
    def __init__(self, elements):
        log.debug('Creating iterator {}'.format(elements))


class Parameter(object):
    '''Returns and register an object storing a value/iterator for a given parameter.'''    
    def __init__(self, name, value):
        self.name = name
        if isinstance(value, string_types):
            if value[0] == '!':
                #Evaluate string
                self.values = eval(value[1:])
            elif value[0] == '#':
                #bash eval
                #TODO: implement
                pass
            else:
                self.values = [value,]
        elif isinstance(value, dict):
            # If dict, we parse the dictionnary and evaluate its contents
            data = copy.deepcopy(value)
            walkDict(data, replacer, self, {})
            self.values = data
        elif isinstance(value, list):
            self.values = value
        else:
            self.values = [value,]
    
    def __repr__(self):
        return '<Parameter> {} = {}'.format(self.name, self.values)


class BenchElement(object):
    ''' 
    A run for given set of parameters.
    
    TODO : detect if parameter combination is equivalent to a previous one.
    '''
    def __init__(self, name, bench_data, parameters):
        self.name = name
        self.bench_data = bench_data
        self.parameters = parameters

    def run(self):
        log.info('Running : {name}'.format(name=self.name))
        log.info('  parameters: {}'.format(self.parameters))

        self.baf = bakfu.Chain.load_chain(self.bench_data)

        score = self.baf.get_chain('score')

        results = dict(
            name=self.name,
            bench_data=self.bench_data,
            parameters=str(self.parameters),
            #parameters contain lambdas -> must be serialized as str
            score=score,
            )

        return results

class Benchmark(object):
    def __init__(self, path):
        file = open(path)
        self.bench_data = yaml.load(file.read())
        self.parameters = []
        self.bench_results = []
        parameters_data = self.bench_data['bench']['parameters'].items()

        result_logger.info('Starting benchmark.')
        result_logger.info('name :  {}'.format(self.bench_data['bench']['name']))


        for parameter_name, parameter_value in parameters_data:
            new_parameter = Parameter(parameter_name, parameter_value)
            self.parameters.append(new_parameter)


    def run(self):
        task = self.bench_data['bench'].get('task','bench')

        self.parameters_dict = {
                parameter.name:parameter.values 
                for parameter in self.parameters
                }

        if task == 'bench':
            self.run_bench()
        elif task == 'optimize':
            self.optimize()

    def optimize(self):
        '''
        Run tests and optimize against a given set of parameters.
        Find the best parameter set by varying within provided bounds.
        '''
        log.info('Starting optimization.')
        dictList = {
                parameter.name:parameter.values
                for parameter in self.parameters}

        parameter_list = {
                parameter.name:parameter.values ['init']
                for parameter in self.parameters}

        idx=0

        data = copy.deepcopy(self.bench_data)
        data.pop('bench')
        walkDict(data, replacer, self, parameter_list)
        
        variables = [(k,v) for k,v in iteritems(dictList) if v['type']==int]
        variables_init = [v['init'] for k,v in variables]

        @lru_cache(maxsize=10000)
        def int_f(*args):
            x=args
            
            parameter_list = dict([(v[0],x) for v,x in zip(variables,x)])
            
            data = copy.deepcopy(self.bench_data)
            data.pop('bench')
            walkDict(data, replacer, self, parameter_list)            
           
            result_logger.info('\nRun :\n---------------------')
            result_logger.info(parameter_list)
            bench_element = BenchElement(idx, data, parameter_list)
            try:
                result = bench_element.run()
                result_logger.info('Score : {}/{}/{}'.format(*result['score']))
                return 1-result['score'][2]
            except:
                return 1
            
        
        def f(x):
            x=[int(v) for v in x]
            return int_f(*x)
        
        x0 = np.array(variables_init,dtype=np.dtype("i"))
        #res = minimize(f,x0)
        res = basinhopping(f,x0, stepsize=50)
        print(res)


    def run_bench(self):
        '''
        Run a benchmark.
        '''
        log.info('Starting benchmark.')
        parameter_combinations = []

        parameter_combinations = [
            dict(izip_longest(self.parameters_dict, v))
            for v in product(*self.parameters_dict.values())]
        
        previous_data = []

        for idx, parameter_list in enumerate(parameter_combinations):
            result_logger.info("\n\n")
            result_logger.info("---------------------------")
            result_logger.info("Run : ")
            result_logger.info("---------------------------")
            result_logger.info('\n\n')
            result_logger.info('Parameter list :')


            data = copy.deepcopy(self.bench_data)
            data.pop('bench')
            walkDict(data, replacer, self, parameter_list)


            #Check if this run is a duplicate
            duplicate = False

            for prev_data in previous_data:
                diff = deepdiff.DeepDiff(data,prev_data)
                if diff.changes == {}:
                    duplicate = True
                    result_logger.info("duplicate ... skipping ...")
                    break

            if not duplicate:
                result_logger.info(parameter_list)              

                previous_data.append(data)

                bench_element = BenchElement(idx, data, parameter_list)
                result = bench_element.run()
                self.bench_results.append(result)

        #write results to json file
        log.info("Writing results to results.json file.")
        with open("results.json","w") as f:
            f.write(jsonpickle.dumps(self.bench_results))
        
        return self.bench_results




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('''Usage : run.py bench.yaml''')
    else:    
        path = sys.argv[1]
        bench = Benchmark(path)
        bench.run()
