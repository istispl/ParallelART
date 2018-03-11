#!/bin/python
# -*- coding: utf-8 -*-
"""holds global variables for testing with pyopencl:
 g_ctx, g_queue and g_options""" 

import sys
from ordereddict import OrderedDict
import math
import os
import numpy
#import cProfile
import pstats
import ctypes
import benchmark
import minitable
import time

# import pyopencl as cl

# globals
global g_ctx,g_queue,g_options,g_profiler,g_session

g_ctx=None
g_queue=None
g_options=None
g_profiler=None
g_session=None


def get_cl_device_name():
	global g_ctx,g_queue,g_options,g_profiler
	if g_queue is None:
		return "numpy"
	return g_queue.get_info(cl.command_queue_info.DEVICE).name.strip()

def get_cl_device_name_short():
	global g_ctx,g_queue,g_options,g_profiler
	if g_queue is None:
		return "numpy"	
	s = g_queue.get_info(cl.command_queue_info.DEVICE).name.strip()
	if s.startswith('GeForce'):
		s=s[8:]
	elif s.startswith('Intel(R) Core(TM) '):
		s=s[len('Intel(R) Core(TM) '):] 
		i=s.find('@')
		if i>0:
			s=s[0:i]
	s=s.strip()
	return s


def get_formatter(title):
	"""Returns a Formatter objects, based on the current command line options.
		The first invocation creates it.
		Subsequenct invocations create new sections.
	"""
	global g_options
	if g_options.formatter is None:
		ffn = format_output_file_name(name="index."+g_options.format)
		print ffn
		g_options.formatter = minitable.new_Formatter(ffn, title)
	else:
		g_options.formatter.section(title)
	return g_options.formatter

def cl_init():
	"""CUDA Driver supports profiling, controlled by env variables:
	export COMPUTE_PROFILE=1
	export COMPUTE_PROFILE_CSV=1
	export COMPUTE_PROFILE_LOG=/home/isti/cuda-profile.log
	"""
	global g_ctx,g_queue,g_options
	import pyopencl as cl
    
#	os.environ['PYOPENCL_CTX'] = '1' 
	os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
	if 'COMPUTE_PROFILE' in os.environ:
		print 'COMPUTE_PROFILE=',os.environ['COMPUTE_PROFILE']
	
	g_ctx = cl.create_some_context()
	g_queue = cl.CommandQueue(g_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
	return g_ctx, g_queue

def main_init(module=sys.modules[__name__]):
	global g_ctx,g_queue,g_options,g_profiler
	from optparse import OptionParser
	usage="""usage: %prog [options] testName
Test names:\n"""
	for i in dir(module): 
		if i.startswith("test"):
			usage += "	"+i+"\n"
	parser = OptionParser(usage)
	parser.add_option("--allow", action="store_true", dest="allowExtendedVigilance", help="If set to true, it allows exceeding vigilance |S_MAX|, like published in original BAR",default=False)
	parser.add_option("-c", "--usecl",action="store_true", dest="usecl", help="Use OpenCL", default=False)
	parser.add_option("--compact", action="store_true", dest="use_compact_kernel", help="Use compact OpenCL kernel (for small instances)", default=False)
	
	parser.add_option("-d", "--diag",action="store_true", dest="diag", help="use diagonal covariance matrices", default=False)
	parser.add_option("--dpi",action="store", dest="dpi", help="Set figure resolution, in dots per inch", default=75,type='int')
	parser.add_option("--db", action="store_true", dest="save_to_db", help="Enable saving profiling session to DB",default=False)
	parser.add_option("--dbsession", action="store", dest="save_to_db_session", help="Append data to an existing session",default=0, type="int")
	parser.add_option("--dump_sets", action="store_true", dest="dump_sets", help="Dump the training sets to .csv files in the outdir",default=False)
	
	parser.add_option("--format",action="store", dest="format", help="Set output file format: html|tex", default='tex')
	parser.add_option("--fig_format",action="store", dest="fig_format", help="Set figure format: eps|png|svg|gif", default='eps')
	
	parser.add_option("-g", "--fig",action="store_true", dest="enable_fig", help="Generate graphics", default=False)
	parser.add_option("--Knuth", action="store", dest="Knuth_variance", help="If set to true, use Knuth's incremental variance formula, otherwise use the one published in original BAR",default=True, type="int")
	
	parser.add_option("-n", "--num-features",action="store", dest="n", help="Number of features", default=4, type="int")
	parser.add_option("--nrep", action="store", dest="nrep", help="Number of repetitions of outer loop of Grid Search",default=1, type="int")
	parser.add_option("--ninner", action="store", dest="ninner", help="Number of repetitions of inner loop of Grid Search",default=10, type="int")
	parser.add_option("--njobs", action="store", dest="n_jobs", help="Number of parallel jobs",default=1,type='int')
	parser.add_option("--ngrid", action="store", dest="n_grid", help="Number of grid steps",default=20,type='int')
	
	parser.add_option("--outdir", action="store", dest="outdir", help="Output Directory",default="out")	

	parser.add_option("-p", "--variate_params",action="store_true", dest="variate_params", help="variate parameters for multiple training", default=False)
	parser.add_option("-P","--profile",action="store_true", dest="profile", help="Enable profiling",default=False)
	parser.add_option("--penalize",action="store_true", dest="penalize", help="Penalize category proliferation",default=False)
	parser.add_option("--PCA",action="store", dest="pca", help="Perform PCA on the dataset to obtain at most n features, 0=no PCA",default=0)
	parser.add_option("--preselect", action="store_true", dest="preselect_P_min", help="If set to true, match-tracking is performed by pre-selecting matching categories that are >= P_min. For the original BAR article default=False",default=False)	
	parser.add_option("--P_min", action="store", dest="P_min", help="P_min",default=0.0, type="float")
	parser.add_option("--resample", action="store", dest="resample", help="Reduce number of samples by stratified random resampling", default=0.0, type="float")

	parser.add_option("--standardize", action="store_true", dest="standardize", help="Standardize the entire dataset (train and test) to zero mean and unit variance",default=False)
	parser.add_option("--standardize_trainingset", action="store_true", dest="standardize_trainingset", help="Standardize only the training set, then apply the scaling factors to the test set",default=False)
	
	
	parser.add_option("--test_size", action="store", dest="test_size", help="Percentage of test set size",default=0.3,type='float')

	parser.add_option("-v","--verbose",action="store", dest="debug", help="Verbosity level of BAR",default=0,type="int")
	parser.add_option("--vv",action="store", dest="vv", help="Verbosity level of GridSearchCV",default=0,type="int")
	parser.add_option("--variance_threshold", action="store", dest="variance_threshold", 
		help="Removes attributes from the data set that have variance below the threshold",default=0.0,type='float')
		
	parser.add_option("--64",action="store_true", dest="double_precision", help="use 64-bit double precision", default=False)
		
	g_options, args = parser.parse_args()
	

	if len(args) > 0: testName = args[0];	 
	else: testName = 'testAll'
	
	datasetName = None
	if len(args) > 1:
		datasetName = args[1]
	g_options.datasetName=datasetName
	# html + eps != love
	if g_options.format=='html' and g_options.fig_format=='eps':
		g_options.fig_format='svg'

	# disable meaningless combinations
	if not g_options.usecl:
		g_options.use_compact_kernel=False	

	if g_options.usecl:
		import pyopencl as cl
		g_ctx, g_queue = cl_init()
	g_options.testName = testName
	g_options.fig_nr=0
	g_options.text_nr=0
	g_options.formatter=None

	if g_options.double_precision:
		g_options.dtype=numpy.float64
	else:
		g_options.dtype=numpy.float32
		
	g_options.t0=time.time()
	numpy.set_printoptions(threshold=3,linewidth=1000)
	

	if g_options.test_size > 1: # assume percentage was given
		g_options.test_size /= 100.0;
	g_profiler=None
	#g_profiler = cProfile.Profile()  # timer=millisecond_timer)
		
	if g_profiler is not None:
		g_profiler.enable()
		g_profiler.disable()
	if g_options.save_to_db:
		create_db_session(g_options.save_to_db_session)
	# not an option, but a global counter
	g_options.fit_ctr=0
	return g_options, args,g_ctx,g_queue

def cudaDeviceReset():
	cudart = ctypes.CDLL("libcudart.so")
	cudart.cudaDeviceReset()
	return


def main_done():
	global g_ctx,g_queue,g_options,g_profiler
	
	if benchmark.profiler_enabled and benchmark.profiler_table:
		benchmark.dump(title='OpenCL profile')
	
	if g_options.profile and g_profiler is not None:
		g_profiler.dump_stats('restats')
		p = pstats.Stats('restats')
		p = p.strip_dirs()
		p.sort_stats('cumulative').print_stats(50)
	if g_options.save_to_db:
		update_db_session()
		if g_session!=None and g_session.id>0:
			g_session.writeIdToFile(g_options.outdir+"/db_session_id")
	del g_queue
	g_queue = None
	del g_ctx
	g_ctx   = None

	if g_options.formatter is not None:
		if g_options.save_to_db and g_session!=None and g_session.id>0:
			g_options.formatter.par("Data for this session was recorded under ID %d" % (g_session.id,))
		g_options.formatter.close()
		g_options.formatter=None

	if g_options.enable_fig:  
		pl.show()
	return


def open_tbl_bar_parameters_scan():
	global g_options,g_session
	if g_session is None: return None
	ddl=OrderedDict((
		('logS_MAX_A', 'double'), 
		('logS_MAX_B', 'double'),
		('plambda_a',  'double'),
		('plambda_b',  'double'),
		('P_min'    ,  'double'),
		('allowExtendedVigilance','int'),
		('preselect_P_min','int'),
		('N_a','int'), 
		('N_b','int'),
		('N_train','int'),
		('N_test','int'),
		('ideal_input_categories','int'),
		('acc_score','double'),
		('rmse','double'), 
		('execution_time','double'),
		('validation_run', 'int'), # if 1 means predict is called on the validation dataset, if 0 it is called on the test dataset 
		('dtype','varchar(64)')
		))	
	tbl = benchmark.MeasurementTable(g_session,"bar_parameters_scan", ddl)
	return tbl

def create_db_session(existing_id):
	global g_options,g_session
	durationSec = time.time() - g_options.t0
	g_session = benchmark.MeasurementSession(clDevice=get_cl_device_name_short(),
									floatPrecision=numpy.dtype(g_options.dtype).itemsize*8,
									durationSec=durationSec,
									problemId=g_options.testName
									)
	if (existing_id <= 0):
		g_session.writeToDB()
		sid = g_session.id
		print 'Created DB session id:', sid
	else:
		g_session.id=existing_id
		sid = g_session.id
		print 'Appending to DB session id:', sid

#	g_session.tbl_bar_parameters_scan = open_tbl_bar_parameters_scan()	
	with open('misc/db_session_id','w') as f:
		f.write('%d' % sid)


def update_db_session():
	global g_options,g_session
	if g_session is not None:
		durationSec = time.time() - g_options.t0
		g_session.update_duration(durationSec)
		benchmark.dump_to_db(g_session)
		g_session.setSessionNumericAttribute('use_compact_kernel', g_options.use_compact_kernel)

		
#	if g_session.tbl_bar_parameters_scan is not None:
#		g_session.tbl_bar_parameters_scan.close()
#		g_session.tbl_bar_parameters_scan=None
	return


def start_profiling_if_required():
	"""Starts profiling, but only if enabled by command-line option """
	global g_ctx,g_queue,g_options,g_profiler
	if g_options.profile and g_profiler is not None:
		g_profiler.enable()

def stop_profiling_if_required():
	global g_ctx,g_queue,g_options,g_profiler
	if g_options.profile  and g_profiler is not None:
		g_profiler.disable()
	
def format_output_file_name(d=None,name=None):
	"""Formats output file name prepending output directory and current test name.
	if d is not None, it is treated as directory.
	Otherwise outdir/testnName is used as directory.
	
	Creates the output directory if not exist"""
	global g_options
	
	if d is None and name is None:
		return None
	
	if d is None: 
		d = g_options.outdir+"/"+g_options.testName
		
	if not os.access(d, os.R_OK):
		os.makedirs(d)
	if name is not None:
		d+="/"+name
	return d

def open_output_file(d=None, name=None, access_mode='w'):
	fn = format_output_file_name(d=d,name=name)
	return open(fn,access_mode)

def format_next_fig_name(base="fig%06d.png"):
	"""Formats output file name for figures,
		returns fullpath, name
	"""
	global g_options
	g_options.fig_nr+=1
	g_options.imgfn = base % (g_options.fig_nr)
	return format_output_file_name(name=g_options.imgfn), g_options.imgfn

def format_next_outfile_name(base="out_%06d.html"):
	"""Formats output file name for documents (html or latex)
		returns fullpath, name
	"""
	global g_options
	g_options.text_nr+=1
	if not "%" in base:
		fn = base
	else:
		fn = base % (g_options.text_nr)
	full_fn=format_output_file_name(name=fn)
	g_options.current_output_file_name=full_fn
	return full_fn, fn

def tee(f,s):
	if f is not None: f.write(s+'\n')
	print s
	
