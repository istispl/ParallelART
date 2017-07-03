#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import os.path
import platform
from ordereddict import OrderedDict
import numpy
import numpy.testing as testing
import scipy
import matplotlib.pyplot as pl
import operator
import pprint
import sklearn
import sklearn.metrics
import sklearn.datasets
import sklearn.cross_validation
import sklearn.naive_bayes
import sklearn.grid_search
import sklearn.linear_model
from sklearn import preprocessing 
import collections
from collections import namedtuple
import joblib
from joblib import Parallel, delayed, logger

from numpy import *
from Bar import *

import Art_test
import pylab as pl
import pstats

import config
import csv
import traceback
import itertools
import time 
import timeit

import inspect
# Dill extends python’s ‘pickle’ module for serializing and de-serializing python objects to the majority of the built-in python types. Serialization is the process of converting an object to a byte stream, and the inverse of which is converting a byte stream back to on python object hierarchy.
# import dill
# our modules
import benchmark
import minitable
import test_datasets

from test_datasets import loadClusteringDataset

def new_Bar(n_a=None, n_b=None, S_MAX_A=None, S_MAX_B=None, logS_MAX_A=None, logS_MAX_B=None,P_min=None, dtype=None, plambda_a=0.01, plambda_b=0.01,allowExtendedVigilance=None,diag=None,
		N0=None,N0_b=None,enable_growth=True,Knuth_variance=None,regression=False,usecl=None,preselect_P_min=None):
	"""returns a new Bar instance, possibly using parameters from g_options"""
	global g_ctx, g_queue, g_profiler, g_options
	if diag is None: diag = g_options.diag
	if dtype is None: dtype = g_options.dtype
	if allowExtendedVigilance is None: allowExtendedVigilance = g_options.allowExtendedVigilance;
	if P_min is None: P_min = g_options.P_min
	if usecl is None: usecl=config.g_options.usecl
	if Knuth_variance is None: Knuth_variance=config.g_options.Knuth_variance
	if preselect_P_min is None:	preselect_P_min=g_options.preselect_P_min
	bar=Bar(n_a=n_a, n_b=n_b, S_MAX_A=S_MAX_A, S_MAX_B=S_MAX_B, logS_MAX_A=logS_MAX_A,logS_MAX_B=logS_MAX_B,
			P_min=P_min, dtype=dtype, plambda_a=plambda_a, plambda_b=plambda_b, 
			allowExtendedVigilance=allowExtendedVigilance, Knuth_variance=Knuth_variance,diag=diag,
			N0=N0,N0_b=N0_b,
			enable_growth=enable_growth,preselect_P_min=preselect_P_min,
			regression=regression,
			debug=g_options.debug,usecl=usecl,
			use_compact_kernel=config.g_options.use_compact_kernel)
	return bar

def write_to_file(fn,contents):
	f = open(fn, 'w')
	f.write(contents)
	f.close()
	return

def send_result(bar,score,dur, tbl=None, ideal_input_categories=None, validation_run=False,rmse=0.0):
	"""
	validation_run 
	if 1 means predict is called on the validation dataset, 
	if 0 it is called on the test dataset 
	"""	
	if ideal_input_categories is None:
		ideal_input_categories=bar.art_b.N
		
	if tbl is not None:
		data=dict({
				'acc_score': 			float(score),
				'rmse':					float(rmse), 
				'execution_time':		float(dur),
				'ideal_input_categories': int(ideal_input_categories),
				'validation_run': 		int(validation_run),
				'N_a': 					bar.art_a.N,
				'N_b': 					bar.art_b.N,				
				}.items() + bar.get_params().items())
		tbl.write(**data)
	return
	

def print_result(bar,score,dur, dur_predict=0.0, f=None, header=False,rmse=0.0):
	"""Print out the result of a training and optionally send to DB
	score = accuracy score
	dur = training duration in seconds
	f=text file to write
	tbl = db table to write
	"""
	from config import tee
	if header:
			tee(f,'#%-9s %10s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % ('ln(S_MAX)', 'ln(S_B)','lmb_a','lmb_b','P_min','N_a','N_b','accscore','rmse','time[ms]','t_pred'))
			header=False
	tee(f,'%10.3f %10.3f %8.2e %8.2e %8.2e %8d %8d %8.5f %8.3f %8.3f %8.3f' % 
			(bar.art_a.logS_MAX,bar.art_b.logS_MAX, 
			 bar.art_a.plambda,
			 bar.art_b.plambda,
			 bar.P_min,
			 bar.art_a.N, bar.art_b.N,score, rmse,
			 dur*1E3, dur_predict*1E3))
		
	return


def train_and_print_result(bar, A_train, B_train, A_test, B_test,f=None,tbl=None, header=False):
	t0 = time.time()
	bar.fit(A_train, B_train)
	dur = time.time()-t0
	
	t0 = time.time()
	C = bar.predict(A_test)
	dur_predict = time.time()-t0
	C=squeeze(C).astype(int)
	
	score = sklearn.metrics.accuracy_score(B_test, C)
	print_result(bar,score,dur, dur_predict=dur_predict, f=f,tbl=tbl,header=header)
	return



def run_job(**kwargs):
			"""
				Runs a job called by test_scan_S_MAX.
				WARNING. This might be run in a different process, even different machine!
				kwargs params:
					test_size
					A
					B
			"""
			dur=0.0
			score=0.0
			rmse=1E9
			dur_predict=0.0
			# kwargs which are not passed to new_Bar are popped
			test_size=kwargs.pop('test_size')
			A=kwargs.pop('A')
			B=kwargs.pop('B')
			r=kwargs.pop('r')
			regression		= kwargs.get('regression')

			if test_size>0:
				a_train, a_test, b_train, b_test = sklearn.cross_validation.train_test_split(A,B, test_size=test_size)
			else:
				a_train, a_test, b_train, b_test = A,A,B,B
			N_train = b_train.shape[0]
			N_test = b_test.shape[0]
					
			bar = new_Bar(**kwargs)
			t0 = time.time();
			try:
					bar.fit(a_train,b_train)
			except Art.TrainingOverflowException as ex:
					pass
			dur = time.time()-t0;
			t0 = time.time()
			b_pred = bar.predict(a_test)
			dur_predict += time.time()-t0
			if regression:
					score = sklearn.metrics.r2_score(b_test, b_pred)
			else:
					score = sklearn.metrics.accuracy_score(b_test, b_pred)	
			
			rmse = sqrt(sklearn.metrics.mean_squared_error(b_test,b_pred))

			job_results = {
				'score':     score,
				'rmse':      rmse,
				'N_a':       bar.art_a.N,
				'N_b':       bar.art_b.N,
				'N_train':   N_train,
				'N_test':    N_test,
				'params':    bar.get_params(False),
				'params_str':bar.get_params_str(),
				'regression':regression,
				'duration':  dur
			}
			return job_results

def init_job_results():
	job_results = {
				'score':     -1E20,
				'rmse':       1E20,
				'N_a':0,
				'N_b':0,
				'N_train':   N_train,
				'N_test':    N_test,
				'params':    None,
				'params_str':None,
				'regression':False,
				'duration':0,
			}
	return job_results

# compares to job_result dicts based on score
def compare_job_results(a,b):
	if a.get('score') > b.get('score'):
		return a
	else:
		return b

def test_scan_S_MAX(A,B,
	N_MAX=10, 
	logS_MAX_A=None, 
	logS_MAX_FINAL=-100,
	S_MAX_B=0.5,
	logS_MAX_B=None,
	P_min=None,
	allowExtendedVigilance=None,
	plambda_a=1E-2,
	plambda_b=1E-2,
	deltaS_MAX=None,
	numSteps=10,
	ideal_input_categories=None,
	name=None,
	append=False,
	nrep=None,
	test_size=0.33,
	regression=False
	):
	"""Reduce logS_MAX_A incrementally until number of input categories exceeds N_MAX
	A,B = training set and labels
	C,D = validation set and labels
	"""
	
	from config import tee
	
	if nrep is None: nrep=config.g_options.nrep
	n_a = A.shape[1] if A.ndim>1 else 1
	n_b = B.shape[1] if B.ndim>1 else 1
	
	bar = None
	tbl = None
	
	tee(None,name+', training set: '+str(A.shape)+' labels:'+str(B.shape))
	

	if allowExtendedVigilance is None: allowExtendedVigilance = config.g_options.allowExtendedVigilance
	if logS_MAX_A is None:
		print 'input A=',A.shape
		var_A = var(A,axis=0)
		logvar_A = log(var_A)
		print 'var_A',var_A.shape,var_A
		idx_inf_A=nonzero(isinf(logvar_A))
		print 'number of inf entries:',size(idx_inf_A),' out of ',n_a
		logvar_A[idx_inf_A]  = 0
		logS_MAX_A=sum(logvar_A)
		print 'sum logvar_A', logS_MAX_A
		logS_MAX_A *= 1.05

	# variables that are modified from run_job
	header = [True]
	
	
	try:
		if P_min is None: P_min=config.g_options.P_min
		if isinstance(P_min, collections.Iterable): 
			P_mins=P_min
		else: P_mins=(P_min,)

		if isinstance(logS_MAX_A, collections.Iterable):
			logS_MAX_As = logS_MAX_A
		else:
			if numSteps is None and deltaS_MAX is not None:
				numSteps = round(abs(logS_MAX_A - logS_MAX_FINAL)/deltaS_MAX)
			elif numSteps is not None and deltaS_MAX is None:
				deltaS_MAX = (logS_MAX_A - logS_MAX_FINAL)/numSteps
			else: 
				raise Exception('exactly one of numSteps or deltaS_MAX should be given')
			logS_MAX_As = numpy.linspace(logS_MAX_A, logS_MAX_FINAL, num=numSteps) 
		
		
		if logS_MAX_B is not None:
			if isinstance(logS_MAX_B, collections.Iterable):
				logS_MAX_Bs = logS_MAX_B
			else:
				logS_MAX_Bs = (logS_MAX_B,)
		else:
			if isinstance(S_MAX_B, collections.Iterable):
				S_MAX_Bs = S_MAX_B
			else:
				S_MAX_Bs = (S_MAX_B,)
			logS_MAX_Bs = log(S_MAX_Bs)
			

		job_params = []
			
		for P_min in P_mins:
			for logS_MAX_B in logS_MAX_Bs:
				for logS_MAX_A in logS_MAX_As:
					for r in xrange(nrep):
						job_params.append({
										# fixed job data
										'test_size': test_size,
										'A': A,
										'B': B,
										# fixed BAR initialization 
										'n_a':n_a,
										'n_b':n_b,
										'plambda_a':plambda_a,
										'plambda_b':plambda_b,
										'allowExtendedVigilance':allowExtendedVigilance, 
										'N0':	N_MAX, 
										'N0_b': ideal_input_categories,
										'regression':regression,

										# variable parameters
										'logS_MAX_A':logS_MAX_A, 
										'logS_MAX_B':logS_MAX_B, 
										'P_min':P_min, 
										'r':r}) 
					
		
#		all_results = [ run_job(**p) for p in job_params ]
		all_results = Parallel(n_jobs=2)(delayed(run_job)(**p) for p in job_params)
		print 'all_results:',len(all_results), '\n'
		for r in all_results: print r.get('score'), ' ', r.get('params_str')
		job_results = reduce(compare_job_results, all_results)	
			
	finally:
		pass
	if regression:
		print 'Best rmse achieved: '+str(job_results.get('rmse'))
	else:
		print 'Best score achieved: %s N_a: %d   N_b: %d ' %(job_results.get('score'),job_results.get('N_a'),job_results.get('N_b') )
	print 'Best params: ' + str(job_results.get('params'))

	# send results to DB
	if config.g_session is not None:
		config.g_session.problemId=name
		config.g_session.remarks = 'training set: '+str(A.shape)+' labels:'+str(B.shape)+"\n" 
		config.g_session.remarks += str(job_results.get('params'))
		config.g_session.params = job_results.get('params_str')
		config.g_session.writeIdToFile(config.g_options.outdir + "/scan/db_session_id")
		if ideal_input_categories is None: ideal_input_categories = 0
		
		tbl = config.open_tbl_bar_parameters_scan()	
		try:
			for r in all_results:
				data=dict({
				'acc_score': 			float(r.get('score')),
				'rmse':					float(r.get('rmse')),
				'execution_time':		float(r.get('duration')),
				'ideal_input_categories': ideal_input_categories,
				'validation_run': 		0,
				'N_a': 					r.get('N_a'),
				'N_b': 					r.get('N_b'),
				'N_train':				r.get('N_train'),
				'N_test': 				r.get('N_test'),
				}.items() + r.get('params').items())
				data['dtype'] = str(data['dtype'])
				tbl.write(**data)
		finally:
			tbl.close()
	return

def characterize_data(A,B):
	var_A = var(A,axis=0)	
	log_sig = log(var_A)
	logS_A    = sum(log_sig[isfinite(log_sig)])
	
	print 'input A.shape=',A.shape
	print 'var_A=',var_A.shape,var_A

	
	if B.ndim==1: B=B.reshape(size(B),1)
	var_B = var(B,axis=0)
	log_sig = log(var_B)
	logS_B    = sum(log_sig[isfinite(log_sig)])
	
	print 'output B.shape=',B.shape
	u_B=unique(B.ravel())
	print 'unique elements in B=',u_B.size,u_B
	print 'var_B=',var_B.shape,var_B
	
	print 'logS_A=',logS_A
	print 'logS_B=',logS_B
	return logS_A, logS_B



def cross_validate(A,B,logS_MAX_A=1E-3,S_MAX_B=1E-3,P_min=0,tbl=None,name=None):
	
	header=True
	f = None
	print 'cross_validate logS_MAX_A=',logS_MAX_A,'S_MAX_B=',S_MAX_B
	B=B.ravel()
	characterize_data(A,B) 
	kf = sklearn.cross_validation.KFold(A.shape[0], n_folds=3)
	for train,test in kf:
		X_train, X_test, y_train, y_test = A[train], A[test], B[train], B[test]
		bar = new_Bar(size(X_train,1), 1, logS_MAX_A=logS_MAX_A, S_MAX_B=S_MAX_B, P_min=P_min, plambda_a=1E-2, plambda_b=1E-4)
		if f is None: 
			namepar = bar.get_params_str() 
			fn=namepar+".txt"
			fdir = config.g_options.outdir+"/crossval/"+name+"/"+config.get_cl_device_name_short();
			f = config.open_output_file(fdir,fn,access_mode=('a' if append else 'w'))
			print fdir+'/'+fn
		train_and_print_result(bar, X_train, y_train, X_test, y_test,f=f,tbl=tbl, header=header)
		header = False	 
	return

def draw2DClassifier(bar, A=None,B=None,W=64, x0=(0,0), w=(3,3),name=None,description=None):
	"""Creates a 2D image from classifier results. X and Y coordinates represent inputs, while color represents the class
	Parameters:
	W     = image width (also height) in pixels
	x0=(0,0)  = image center corresponds to x0 in parameter space
	w=(2,2)   = image width in parameter space
	"""
	pal=numpy.fromstring("""
         0         0    1.0000
         0    0.5000         0
    1.0000         0         0
         0    0.7500    0.7500
    0.7500         0    0.7500
    0.7500    0.7500         0
    0.2500    0.2500    0.2500
         0         0    1.0000
         0    0.5000         0
    1.0000         0         0
         0    0.7500    0.7500
    0.7500         0    0.7500
    0.7500    0.7500         0
    0.2500    0.2500    0.2500
         0         0    1.0000
         0    0.5000         0
    1.0000         0         0
         0    0.7500    0.7500
    0.7500         0    0.7500
    0.7500    0.7500         0
    0.2500    0.2500    0.2500
         0         0    1.0000
         0    0.5000         0
    1.0000         0         0
         0    0.7500    0.7500
    0.7500         0    0.7500
    0.7500    0.7500         0
    0.2500    0.2500    0.2500
         0         0    1.0000
         0    0.5000         0
    1.0000         0         0
         0    0.7500    0.7500
    0.7500         0    0.7500
    0.7500    0.7500         0
    0.2500    0.2500    0.2500
         0         0    1.0000
         0    0.5000         0
    1.0000         0         0
         0    0.7500    0.7500
    0.7500         0    0.7500
    0.7500    0.7500         0
    0.2500    0.2500    0.2500
""",sep=" ").reshape((42,3))
	print "palette=",pal.shape
	assert bar.art_a.n == 2, "This method requires 2 dimensional input samples"
	I=numpy.zeros((W,W,3),dtype=float32)
	for i in range(W):
		for j in range(W):
			x=numpy.array([(i/W - 0.5)*w[0]+x0[0], (j/W-0.5)*w[1]+x0[1]])
			c = int(bar.predict(x))
			p = bar.predict1_P(x)[c]
			C = pal[c,:]*p
			I[W-i-1,j,:] = C

	formatter = config.get_formatter("Classifier")
	if description is not None:
		formatter.par(description)
	Art_test.drawArt(bar.art_a)
	fullpath, figname = config.format_next_fig_name(name + "%06d."+config.g_options.fig_format)
	pl.grid(True)
	pl.savefig(fullpath, dpi=config.g_options.dpi, transparent=True)
	formatter.write_imgref(figname, caption="ART_a")

	Art_test.printArtTable(bar.art_a, formatter)

	if A is not None:
		Art_test.drawAndExplainART(name='Classifier',A=A,B=B,art=bar.art_a)


	fullpath, figname = config.format_next_fig_name(name + "%06d.png")
	scipy.misc.imsave(fullpath,I);
	formatter.write_imgref(figname, caption="Classification results of the input sample space")

	if A is not None:
		gnb = sklearn.naive_bayes.GaussianNB()
		a_test=A
		b_test=B
		b_pred = gnb.fit(A, B).predict(A)
		Art_test.drawAndExplainGaussianNB(A, B,A_test=a_test, B_test=b_test, B_pred=b_pred, gnb=gnb, axislabels=None,name=name,numfigs=1,f=formatter)


	return


def run_gmm(a_train, a_test, b_train, b_test):
	"""Run Gaussian Naive Bayes classifier on the data set"""
	gnb = sklearn.naive_bayes.GaussianNB()
	y_pred = gnb.fit(a_train, b_train).predict(a_test)
	print 'classes'
	print gnb.classes_
	print 'prior'
	print gnb.class_prior_
	print 'mu'	
	print gnb.theta_
	print 'sigma'
	print gnb.sigma_

	print "accuracy_score = ",sklearn.metrics.accuracy_score(b_test, y_pred)
	return

# penalize score by considering # of attributes
def penalizer(score, Na=2, Nb=2):
	Npenalty = Nb*3.0
	if Na > Npenalty: score += Npenalty / float(Na) - 1.0
	return score

def penalizer2(score, Na=2, Nb=2):
	score += float(Nb) / float(max(Na,Nb))
	return score

def scorer(estimator, X, y, validation_run=True):
	"""callback by sklearn.grid_search.
	if validation_run=True it means its called from validation grid search
	if False, it's called on the test set 
	"""
	s = estimator.score(X,y)
	if config.g_session is not None:
		rmse = sqrt(sklearn.metrics.mean_squared_error(y,estimator.predict(X)))
		send_result(estimator,s,estimator.fit_duration, 
				tbl=config.g_session.tbl_bar_parameters_scan,validation_run=validation_run,rmse=rmse)
	if validation_run and config.g_options.penalize:
		s=penalizer(s,Na=estimator.art_a.N, Nb=estimator.art_b.N)
	return s

def regression_report(y_true, y_pred):
	return """
	r2 score:				%g
	mean squared error:		%g
	mean absolute error:	%g
	""" % (sklearn.metrics.r2_score(y_true,y_pred), 
		sklearn.metrics.mean_squared_error(y_true,y_pred),
		sklearn.metrics.mean_absolute_error(y_true,y_pred)
		)

class RecordEstimator:
	"""This class encapsulates another Estimator but it logs results"""
	def __init__(self,**kwargs):
		return
	def predict(self, X):
		return 0
	def fit():
		return 0
		
def drawAndDescribeDataset(dataset):
	name   = dataset.name
	A	   = dataset.data
	B	   = dataset.target
	title  = name
	html = config.get_formatter("The %s dataset" % (name,))
	html.par("The dataset consists of %d attributes and %d instances" % (A.shape[1],A.shape[0]))
	Art_test.drawMultiProjections(A=A,B=B,name=name,f=html,axislabels=dataset.attr_labels)
	html.par('categories (%s):' % (dataset.num_classes))
	html.par(str(dataset.target))

	if config.g_options.pca:
		html.par("PCA was performed.")

	if config.g_options.standardize:
		html.par("The dataset was scaled to zero mean and unit variance")
		
	html.printVector(amin(A,0), "min")
	html.printVector(mean(A,0), "\mu")
	html.printVector(amax(A,0), "max")
	
	if config.g_options.standardize:
		html.printVector(dataset.orig_std,"original  \sigma")
		html.printVector([sum(log(dataset.orig_std))],  'original \sum{\log{\sigma_i}}')
		html.printVector([amin(log(dataset.orig_std))], 'original \min{\log{|\sigma_i|}}')
		html.printVector([amax(log(dataset.orig_std))], 'original \max{\log{|\sigma_i|}}')

	html.printVector(dataset.new_std,"std")
	html.printVector([sum(dataset.log_std)],  '\sum{\log{\sigma_{i,i}}}')
	html.printVector([amin(dataset.log_std)], '\min{\log{\sigma_{i,i}}}')
	html.printVector([amax(dataset.log_std)], '\max{\log{\sigma_{i,i}}}')

		#	html.printVector('logS_MAX_A_RANGE ' + str(dataset.logS_MAX_A_RANGE))
	return

		
def testDrawDataset():
	global g_options
	datasetName=config.g_options.datasetName
	dsets = (dd for dd in test_datasets.get_datasets_descriptors() if (datasetName is None) or (dd.get('name')==datasetName))
	for dd in dsets:	
		dataset	=	test_datasets.get_dataset(**dd)
		drawAndDescribeDataset(dataset)
		
	return

	
	
def gridSearchCV(dataset):
	global g_options
	name   = dataset.name
	A	   = dataset.data
	B	   = dataset.target
	nrep   = g_options.nrep
	ninner = g_options.ninner
	logS_MAX_A_RANGE = dataset.logS_MAX_A_RANGE
	logS_MAX_B_RANGE = dataset.logS_MAX_B_RANGE
	P_min_RANGE      = dataset.get('P_min_RANGE',None)
	# logS_MAX is scanned backwards, starting from large value
	param_grid={
		'logS_MAX_A':linspace(float(logS_MAX_A_RANGE[1]), float(logS_MAX_A_RANGE[0]), num=g_options.n_grid),
		}
#		'P_min':numpy.array([0, 0.1, 0.5, 0.9, 1.0])

#	if logS_MAX_B_RANGE is not None:
#		param_grid.update({'logS_MAX_B':linspace(float(logS_MAX_B_RANGE[1]),float(logS_MAX_B_RANGE[0]),num=8)})
#	if P_min_RANGE is not None:
#		param_grid.update({'P_min':linspace(float(P_min_RANGE[0]),float(P_min_RANGE[1]),num=3)})

	param_grid.update({'N0_b':(dataset.num_classes,)})
# outer xval iterator
#   iterate 10 times, each time using test_size % as testing data
	if dataset.regression:
		outer_cv = sklearn.cross_validation.ShuffleSplit(B.shape[0], n_iter=nrep, test_size=g_options.test_size)
	else:
		outer_cv = sklearn.cross_validation.StratifiedShuffleSplit(B, n_iter=nrep, test_size=g_options.test_size)
	
	scores=()
	Na_s=()
	Nb_s=()
	logS_MAX_A_s=()
	pred_durations=()	# list of prediction durations
	fit_durations=()    # list of fit durations
	best_params=()
# prepare output report file
	config.g_options.testName=name+"_allow_%d_diag_%d_presel_%d_std_%d_penal_%d" % (config.g_options.allowExtendedVigilance, config.g_options.diag, 
		config.g_options.preselect_P_min,
		config.g_options.standardize,
		config.g_options.penalize)
	title = "%s %s using %s" % (
			('Predicting' if dataset.regression else 'Classifying'),
			name,
			'BAR'
		)
	
	html = config.get_formatter(title)
	
	html.section("Introduction")
	html.par("""In this paper, we analyze numerically the hyperparameter tuning of  %(problemtype)s 
	on the %(name)s dataset"""
		% 	{'problemtype':('regression' if dataset.regression else 'classification'),
			 'name': dataset.name
			}
	)
	html.sectione()
	drawAndDescribeDataset(dataset)
	html.section("Training using BAR")
	
	txt="The BAR classifier variant %s exceeding $S_{MAX}$,	uses %s covariance matrices and	preselects based on $P_{min}$: %f""" % (
		("allows" if config.g_options.allowExtendedVigilance else "doesn\'t allow"), 
		("diagonal" if config.g_options.diag else "generic"), 
		config.g_options.preselect_P_min)
	html.par(txt)
	html.par("""The dataset consists of %d attributes and %d instances"""
		% (A.shape[1],A.shape[0]))
		
	html.par("""A grid search using the following parameters is performed:""")
	for key in param_grid:
		if key=='P_min': skey='P_{min}'
		elif key=='logS_MAX_A': skey=r"\ln{S_{MAX}^a}"
		elif key=='logS_MAX_B': skey=r"\ln{S_{MAX}^b}"
		else: skey=key
		html.printVector(param_grid[key],skey)

	if config.g_options.penalize:
		html.par("For parameter validation, the following score penalizer function is used to reduce the number of input categories")
		#src=inspect.getsourcelines(penalizer)
		html.pre(inspect.getsource(penalizer))

	html.par("""The search is repeated %(nrep)d times, and the average score is reported at the end.""" % locals())
	html.sectione()

	r=0
	#for r in xrange(nrep):
	#	a_train, a_test, b_train, b_test = sklearn.cross_validation.train_test_split(A,B, test_size=0.33)
	for train_index, test_index in outer_cv:
		a_train, a_test = A[train_index], A[test_index]
		b_train, b_test = B[train_index], B[test_index]

		html.subsection("Run %d/%d" % (r+1,nrep))
		
		scaler=None
		if config.g_options.standardize_trainingset:
			scaler=sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
			a_train = scaler.fit_transform(a_train)
			a_test  = scaler.transform(a_test)
			html.par("The training set was standardized to zero mean and unit variance")
			html.printVector(scaler.mean_,'\mu')
			html.printVector(scaler.std_,'\sigma')
			html.par("The above parameters were applied to the test set, resulting:")
			html.printVector(numpy.mean(a_test,0),"\mu'")
			html.printVector(numpy.std(a_test,0),"\sigma'")
		pl.clf()
		if 1:
	# inner xval iterator
			if dataset.regression:
				inner_cv = sklearn.cross_validation.ShuffleSplit(a_train.shape[0],n_iter=ninner, test_size=g_options.test_size)
			else:
				inner_cv = sklearn.cross_validation.StratifiedShuffleSplit(b_train, n_iter=ninner, test_size=g_options.test_size)
	# construct grid searcher
			estimator = new_Bar(regression=dataset.regression)
			clf = sklearn.grid_search.GridSearchCV(estimator, param_grid, scoring=scorer, cv=inner_cv, verbose=config.g_options.vv, n_jobs=config.g_options.n_jobs, refit=True)
			# refit on the whole training set, measure time
			t0 = time.time()
			clf.fit(a_train,b_train)
			d = time.time()-t0
			bar    = clf.best_estimator_
			fit_durations += bar.fit_durations

			serializable_params=bar.get_params().copy()
			serializable_params.pop('dtype')
			serializable_params['usecl']=bar.usecl
			best_params += (serializable_params,)
			
			t0 = time.time()
			b_pred = bar.predict(a_test)
			pred_durations += (time.time()-t0,)
			testing.assert_array_equal(b_pred, clf.predict(a_test))
			s      = scorer(bar, a_test, b_test, validation_run=False)
			html.dlb().dlr("Grid Search time","%.3f sec / %d folds"         % (d,ninner)).dlr("Score on the Test Set", "(%d instances): %.3f"  % (a_test.shape[0], s)).dlr("Number of input categories created", "%d"     % (bar.art_a.N,)).dlr("Number of output categories created", "%d"    % (bar.art_b.N,)).dle()

			Art_test.printGridSearchCVResults(html,clf)
			html.subsection("Representation of the BAR trained with the best parameters determined by the grid search")
			Art_test.drawAndExplainART(a_train, b_train, A_test=a_test, B_test=b_test, B_pred=b_pred, art=bar.art_a, axislabels=dataset.attr_labels, name=name, f=html)
			
			if bar.regression:
				txt=regression_report(b_test, b_pred)
			else:
				txt=sklearn.metrics.classification_report(b_test, b_pred)
				html.subsection("Classification report")
	 		
			html.pre(txt)
			
			scores+=(s,)
			Na_s+=(bar.art_a.N,)
			Nb_s+=(bar.art_b.N,)
			logS_MAX_A_s += (bar.art_a.logS_MAX,)
		
		pl.clf()
		html.sectione()
		# perform a different fit
		if dataset.regression:
			regressor = sklearn.linear_model.BayesianRidge(compute_score=True)
			regressor.fit(a_train, b_train)
			b_pred = regressor.predict(a_test)
			html.par("BayesianRidge results:")
			html.pre(regression_report(b_test, b_pred))
		else:
			gnb = sklearn.naive_bayes.GaussianNB()
			b_pred = gnb.fit(a_train, b_train).predict(a_test)			
			Art_test.drawAndExplainGaussianNB(a_train, b_train,A_test=a_test, B_test=b_test, B_pred=b_pred, gnb=gnb, 
				axislabels=dataset.attr_labels, name=name, f=html,numfigs=1)

		# copy gnb to BAR (just an internal verification)
		if 0:
			pl.clf()
			bar = GaussianNB_to_BAR(gnb)
			b_pred=bar.predict(a_test) 
			Art_test.drawAndExplainART(a_train, b_train,A_test=a_test, B_test=b_test, B_pred=b_pred, art=bar.art_a, 
				axislabels=dataset.attr_labels, name=name, f=html,numfigs=1)
			txt=sklearn.metrics.classification_report(b_test, b_pred)
			html.pre("BAR from GaussianNB:\n"+txt)
		
		html.sectione()
		
		r+=1
	# end of outer loop	
	
	# create a 'master' results namedtuple and print it
	if len(scores)>0:
		
		p=None
		if config.g_options.penalize:
			p=inspect.getsourcelines(penalizer)

		MasterResults = namedtuple('MasterResults',('name','avgScore','stdScore','avgNa', 'scores', 'Na_s', 'Nb_s', 'logS_MAX_A_s','penalizer','argv','pred_durations', 'fit_durations','best_params'))
		r = MasterResults(name,
			float(mean(scores)),
			float(std(scores)),
			float(mean(Na_s)),
			scores,
			Na_s,
			Nb_s,
			logS_MAX_A_s,
			p,
			sys.argv,
			pred_durations,fit_durations,best_params )
		
		Art_test.printGridSearchMasterResults(r,html)
	html.close()
	return


def GaussianNB_to_BAR(gnb):
	N=gnb.theta_.shape[0]	
	n=gnb.theta_.shape[1]
	bar=new_Bar(n_a=n, n_b=1, N0=N, diag=1)
	bar.reinit()
	bar.art_a.set_MU_SIGMA_P(mu=gnb.theta_, sigma=gnb.sigma_, P=gnb.class_prior_)
	mu_b = numpy.arange(N)[:,newaxis]
	sigma_b = numpy.ones((N,1))*1E-3
	P_b=numpy.ones((N))
	bar.art_b.set_MU_SIGMA_P(mu=mu_b, sigma=sigma_b, P=P_b)
	
	bar.w=numpy.eye(N,N)
	bar.P_w=numpy.eye(N,N)
	bar.P_b_a=numpy.eye(N,N) 
	return bar

def gridSearchCV_descr(**kwargs):
	dataset	=	test_datasets.get_dataset(**kwargs)
	gridSearchCV(dataset)
	return

def drawBAR(bar):
	pl.matshow(bar.get_joint_p())
	pl.colorbar()
	pl.title("P_w")
	pl.show()
	return

def to_device(a):
    import pyopencl as cl
    return cl.array.to_device(config.g_queue,a.astype(config.g_options.dtype))
 
# benchmark on synthetic datasets
def benchmark_fit():
	from config import tee
	pred = False
	config.g_options.testName=('benchmark_%s_compact_%d' % ('predict' if pred else 'fit', config.g_options.use_compact_kernel))
	print config.g_options.testName
	print 'GPU:', config.get_cl_device_name_short()
	print 'numpy version:', numpy.__version__
	print platform.processor()
	print 'n = number of features, N=number of classes' 
	print 'TIME_CPU = execution time using numpy (ms)'
	print 'T_GPU_TOT = wall-clock time measured on host side, includes various overheads (ms)'  
	print 'T_GPU_KRNL = OpenCL kernel execution time (ms)'
	print 'execution times are in milliseconds'
	
	with config.open_output_file(config.g_options.outdir+"/benchmark_fit/"+config.get_cl_device_name_short(),config.g_options.testName ) as f:
		header  = True
		for exL in xrange(0, 1):
			for exN in xrange (5,7,1):
				for exn in xrange(1, 2): 
					n = 2 ** exn  # number of features
					N = 2 ** exN  # number of categories
					L = 2 ** exL
					if exn+exN > 25:
						break
	
					rep = 1
					S_MAX=1
					x,y,MU,SIGMA = Art_test.create_random_mixture(n=n,N=N,L=L,diag=config.g_options.diag, S_MIN=S_MAX, S_MAX=S_MAX,dtype=config.g_options.dtype)
					nsamples=x.shape[0]
					
					logS_MAX_A = n*math.log(S_MAX,n)+10.0
					
					art_ref = new_Bar(n, 1, logS_MAX_A=logS_MAX_A, S_MAX_B=1E-4, N0=N, usecl=False)
					art_gpu = new_Bar(n, 1, logS_MAX_A=logS_MAX_A, S_MAX_B=1E-4, N0=N, usecl=True)
					art_ref.name = 'art_ref'
					art_gpu.name = 'art_gpu'
					#print repr(x)
					config.start_profiling_if_required()
					art_gpu.reinit()
					x_gpu = to_device(x)
					y_gpu = art_gpu.art_b.intern(y)
	
					if pred:
						art_ref.fit(x,y)
						art_gpu.fit(x_gpu,y_gpu)
	
					def fit_ref():
						art_ref.fit(x,y)
						
					def fit_gpu():
						art_gpu.art_a.duration_cl_fit_bar=0
						art_gpu.fit(x_gpu,y_gpu)
	
					def pred_ref():
						art_ref.predict(x)
						
					def pred_gpu():
						art_gpu.art_a.duration_cl_fit_bar=0
						art_gpu.predict(x_gpu)
	
					s_ref = pred_ref if pred else fit_ref
					s_gpu = pred_gpu if pred else fit_gpu
						
					d_ref = Art_test.run_art_bench(art_ref, None, stmt=s_ref,setup=s_ref, rep=rep)/nsamples
					d_gpu = Art_test.run_art_bench(art_gpu, None, stmt=s_gpu,setup=s_gpu, rep=rep)/nsamples
					d_krnl = art_gpu.art_a.duration_cl_fit_bar/nsamples
					
		
					testing.assert_equal(art_ref.art_a.nsamples,nsamples)
					testing.assert_equal(art_ref.art_b.nsamples,nsamples)
					testing.assert_equal(art_gpu.art_a.nsamples,nsamples, err_msg='art_gpu.art_a.nsamples')
					testing.assert_equal(art_gpu.art_b.nsamples,nsamples)
					
				
					if art_gpu.art_a.N == art_ref.art_a.N:
						re1 = Art_test.rel_error(art_gpu.art_a.get_MU(), art_ref.art_a.get_MU())
						re2 = Art_test.rel_error(art_gpu.art_a.get_SIGMA(), art_ref.art_a.get_SIGMA())
						re = max(re1,re2)
					else:
						mN = min(art_gpu.art_a.N,art_ref.art_a.N)
						re1 = Art_test.rel_error(art_gpu.art_a.get_MU()[:mN,:], art_ref.art_a.get_MU()[:mN,:])
						re2 = Art_test.rel_error(art_gpu.art_a.get_SIGMA()[:mN,:], art_ref.art_a.get_SIGMA()[:mN,:])
						re = max(re1,re2)
	
	
					#testing.assert_allclose(art_gpu.art_a.get_MU(), art_ref.art_a.get_MU(), rtol=1E-6)
					# we have 3 vectors: a, mu and sigma
					nbytes  = 0
					bw = 1E-9 * nbytes / d_krnl
					
					if header:
						tee(f, '#%-6s %6s %6s %10s %10s %10s %10s %10s %10s %7s' % ('N','n','L','TIME_CPU', 'T_GPU_TOT', 'T_GPU_KRN','SPEEDUP','SPEEDUP_K', 'BW[GB/s]','ERROR'))
						header = False
						config.start_profiling_if_required()
					
					tee(f,'%7d %6d %6d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %7.1e %s' % (art_ref.art_a.N,n,L,d_ref * 1E3, d_gpu * 1E3, 
						d_krnl*1E3,
						d_ref / d_gpu, 
						d_ref / d_krnl,
						bw, re, '!' if re>1E-6 else ''))
	if config.g_session is not None:
		config.g_session.problemId='mixture'
		if config.g_session.params is None: config.g_session.params = ''
		config.g_session.params+=' use_compact_kernel=%d ' % art_gpu.use_compact_kernel

	return


# benchmark on a 'real-world' dataset
def benchmark_data():
	from config import tee
	from test_hyperparams import get_best_hyperparams
	rep  = 1 # repetitions of fit()
	pred = True
	name = config.g_options.datasetName
	dd = test_datasets.get_dataset_descriptor_by_name(name)
	if dd is None:
		raise Exception('Dataset %s was not found' % (name))
		return 
	dataset = test_datasets.get_dataset(**dd)

	config.g_options.testName=('benchmark_%s_compact_%d' % ('predict' if pred else 'fit', config.g_options.use_compact_kernel))
	print config.g_options.testName
	print 'GPU:', config.get_cl_device_name_short()
	print 'numpy version:', numpy.__version__
	print platform.processor()
	print 'n = number of features, N=number of classes' 
	print 'TIME_CPU = execution time using numpy (ms)'
	print 'T_GPU_TOT = wall-clock time measured on host side, includes various overheads (ms)'  
	print 'T_GPU_KRNL = OpenCL kernel execution time (ms)'
	print 'execution times are in milliseconds'
	params = get_best_hyperparams(name).copy()
	if params is None:
		raise Exception('no known hyperparameter values for %s' % (name,))
		return
	x        = dataset.data
	y        = dataset.target
	nsamples = x.shape[0]
	n_a      = x.shape[1]
	n_b      = 1
	params.update({'n_a':n_a})
	params.update({'n_b':n_b})
	params.update({'N0_b':dataset.num_classes})
	params.update({'regression':dataset.regression})
	params_ref = params.copy()
	params_ref.update({'usecl':False})
	params_gpu = params.copy()
	params_gpu.update({'usecl':True})
	print 'number of instances:  ',nsamples
	print 'number of attributes: ',n_a
	with config.open_output_file(config.g_options.outdir+"/benchmark_fit/"+config.get_cl_device_name_short(),config.g_options.testName ) as f:
					header  = True

					art_ref = Bar(**params_ref)
					art_gpu = Bar(**params_gpu)
					art_ref.name = 'art_ref'
					art_gpu.name = 'art_gpu'
					config.start_profiling_if_required()
					art_gpu.reinit()

					x_gpu = to_device(x)
					y_gpu = art_gpu.art_b.intern(y)
	
					if pred:
						art_ref.fit(x,y)
						art_gpu.fit(x_gpu,y_gpu)
	
					def fit_ref():
						art_ref.fit(x,y)
						
					def fit_gpu():
						art_gpu.art_a.duration_cl_fit_bar=0
						art_gpu.fit(x_gpu,y_gpu)
	
					def pred_ref():
						art_ref.predict(x)
						
					def pred_gpu():
						art_gpu.art_a.duration_cl_fit_bar=0
						art_gpu.predict(x_gpu)
	
					s_ref = pred_ref if pred else fit_ref
					s_gpu = pred_gpu if pred else fit_gpu
						
					d_ref = Art_test.run_art_bench(art_ref, None, stmt=s_ref,setup=s_ref, rep=rep)/nsamples
					d_gpu = Art_test.run_art_bench(art_gpu, None, stmt=s_gpu,setup=s_gpu, rep=rep)/nsamples
					d_krnl = art_gpu.art_a.duration_cl_fit_bar/nsamples
					
		
					testing.assert_equal(art_ref.art_a.nsamples,nsamples)
					testing.assert_equal(art_ref.art_b.nsamples,nsamples)
					testing.assert_equal(art_gpu.art_a.nsamples,nsamples, err_msg='art_gpu.art_a.nsamples')
					testing.assert_equal(art_gpu.art_b.nsamples,nsamples)
					
				
					if art_gpu.art_a.N == art_ref.art_a.N:
						re1 = Art_test.rel_error(art_gpu.art_a.get_MU(), art_ref.art_a.get_MU())
						re2 = Art_test.rel_error(art_gpu.art_a.get_SIGMA(), art_ref.art_a.get_SIGMA())
						re = max(re1,re2)
					else:
						mN = min(art_gpu.art_a.N,art_ref.art_a.N)
						re1 = Art_test.rel_error(art_gpu.art_a.get_MU()[:mN,:], art_ref.art_a.get_MU()[:mN,:])
						re2 = Art_test.rel_error(art_gpu.art_a.get_SIGMA()[:mN,:], art_ref.art_a.get_SIGMA()[:mN,:])
						re = max(re1,re2)
	
	
					#testing.assert_allclose(art_gpu.art_a.get_MU(), art_ref.art_a.get_MU(), rtol=1E-6)
					# we have 3 vectors: a, mu and sigma
					nbytes  = 0
					bw = 1E-9 * nbytes / d_krnl
					
					if header:
						tee(f, '#%-6s %6s %6s %10s %10s %10s %10s %10s %7s' % ('N','n','TIME_CPU', 'T_GPU_TOT', 'T_GPU_KRN','SPEEDUP','SPEEDUP_K', 'BW[GB/s]','ERROR'))
						header = False
						config.start_profiling_if_required()
					
					tee(f,'%7d %6d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %7.1e %s' % (art_ref.art_a.N,n_a,d_ref * 1E3, d_gpu * 1E3, 
						d_krnl*1E3,
						d_ref / d_gpu, 
						d_ref / d_krnl,
						bw, re, '!' if re>1E-6 else ''))

	if config.g_session is not None:
		config.g_session.problemId='mixture'
		if config.g_session.params is None: config.g_session.params = ''
		config.g_session.params+=' use_compact_kernel=%d ' % art_gpu.use_compact_kernel

	return


def save_dataset(name='',train=None, test=None,r=0):
	dd = 'out/scan/'+name
	if not os.path.exists(dd): os.makedirs(dd)
	if train is not None:
		savetxt( dd+"/train_%d.txt" % r, train, fmt='%12g', delimiter='\t')
	if test is not None:
		savetxt( dd+"/test_%d.txt" % r, train, fmt='% 12g', delimiter='\t')
	return


def normalize_data(A):
	return sklearn.preprocessing.normalize(A, norm='l1',axis=0)



def scale_0_1(A,B):
	"""
	scale input and output sets to 0...1
	see http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
	"""
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
	A = min_max_scaler.fit_transform(A)
	
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
	if len(B.shape)<2: B=B[:,np.newaxis]
	B = min_max_scaler.fit_transform(B)
	return A,B


def test_scan_dataset():
	name=config.g_options.datasetName
	dd = test_datasets.get_dataset_descriptor_by_name(name)
	if dd is None:
		raise Exception('Dataset %s was not found' % (name))
		return 
	dataset = test_datasets.get_dataset(**dd)
	logS_MAX_A_RANGE=dataset.logS_MAX_A_RANGE
	A = dataset.data
	B = dataset.target
	test_scan_S_MAX(A, B, N_MAX=100, 
				logS_MAX_A = numpy.linspace(logS_MAX_A_RANGE[0],logS_MAX_A_RANGE[1], num=10), 
				logS_MAX_B = -1,
				P_min=0.0,
				nrep=1,
				name=dataset.name,
				regression=dataset.regression,
				ideal_input_categories=dataset.num_classes)


def testAll():
	datasetName=config.g_options.datasetName
	
	dsets = (dd for dd in test_datasets.get_datasets_descriptors() 
			if (datasetName is None) or (dd.get('name')==datasetName))
	#Parallel(n_jobs=2)(delayed(gridSearchCV_descr)(**dd) for dd in dsets)
	for dd in dsets: gridSearchCV_descr(**dd) 
	return

def benchAll():
	
	rep = 5 # how many times to repeat 
	datasetName=config.g_options.datasetName
	
	dsets = (dd for dd in test_datasets.get_datasets_descriptors() 
			if (datasetName is None) or (dd.get('name')==datasetName))
	
	archname = config.get_cl_device_name_short()+" "+("compact kernel" if config.g_options.use_compact_kernel else "separate kernels")
	ffn = config.format_output_file_name(
				d="out/benchAll", name=archname.replace(' ','_')+"."+config.g_options.format)
	print ffn
	fmt = minitable.new_Formatter(ffn, 'Benchmark')
	
	fmtxtfn = config.format_output_file_name(d="out/benchAll", name=archname.replace(' ','_')+".txt")
	print fmtxtfn
	fmtxt = minitable.new_Formatter(fmtxtfn)

	
	fmt.table('Benchmark')
	
	fmt.trh()
	fmt.th("%-30s", "Dataset")
	fmt.th("%4s", "Type")
	fmt.th("%9s", "Features")
	fmt.th("%9s", "Instances")
	fmt.th("%20s", archname)
	fmt.trhe()

	fmtxt.table('Benchmark')
	fmtxt.trh()
	fmtxt.th("%-30s", "Dataset")
	fmtxt.th("%20s", archname)
	fmtxt.trhe()
	
	config.start_profiling_if_required()
	for dd in dsets: 
		dataset	=	test_datasets.get_dataset(**dd)
		A,B = dataset.data, dataset.target
		bar=new_Bar(A.shape[1], 1, logS_MAX_A=dataset.logS_MAX_A, logS_MAX_B=dataset.logS_MAX_B, N0=50, regression=dataset.regression)
		# warmup
		t0 = time.time()		
		bar.fit(A,B)
		dur = time.time()-t0
		
		def run_bar():
			bar.reset()
			bar.partial_fit(A,B)
			return
			
		t_ref = timeit.Timer(stmt=run_bar)	
		dur  = t_ref.timeit(number=rep) / float(rep)
		# run second time to get time
		
#		t0 = time.time()		
#		bar.partial_fit(A,B)
#		dur = time.time()-t0


		print "%20.2f" % (dur*1E3)
		fmt.tr();
		fmt.td("%-30s", dataset.name)
		fmt.td("%4s", "R" if dataset.regression else "C")
		fmt.tdn("%9d", A.shape[1])
		fmt.tdn("%9d", A.shape[0])
		fmt.tdn("%20.2f", dur*1E3)
		fmt.tre();
		
		fmtxt.tr()
		fmtxt.td("%-30s", dataset.name)
		fmtxt.tdn("%20.2f", dur*1E3)
		fmtxt.tre()
	fmt.tablee()
	fmt.close()
	fmtxt.tablee()
	fmtxt.close()
	return


def testGridSearch():
	datasetName=config.g_options.datasetName
	
	dsets = (dd for dd in test_datasets.get_datasets_descriptors() 
			if (datasetName is None) or (dd.get('name')==datasetName))
	#Parallel(n_jobs=2)(delayed(gridSearchCV_descr)(**dd) for dd in dsets)
	if not dsets:
		raise Exception("No such dataset '%s'" % (datasetName,))
		return
	for dd in dsets: gridSearchCV_descr(**dd) 
	return


	
if __name__ == '__main__':
	global g_options
	g_options, args,g_ctx,g_queue = config.main_init(module=sys.modules[__name__])	
	if len(args) > 0: testName = args[0];	 
	else: testName = 'testAll'
	
	config.g_options.fig = pl.figure(figsize=(8, 8))  # figsize in inches
	try:
		eval(testName+"()")
	except Exception as e:
		print e
		traceback.print_exc()
		
	config.main_done()

