#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
from numpy import shape
import scipy.misc
import pyopencl as cl
import pyopencl.array
import pyopencl.clmath as clmath
import pyopencl.characterize
import math
import Art
import benchmark

class Art_cl(Art.Art):
	
	def __init__(self, n, S_MAX=None, logS_MAX=None, plambda=0.1, dtype=np.float32, 
		diag=True,allowExtendedVigilance=True,
		Knuth_variance=True,normalized_classifier_categories=False,
		ctx=None,queue=None,N0=None, enable_growth=True, max_blocksize=None,
		name='art',usecl=True,debug=0, buildProgram=True):

		Art.Art.__init__(self, n, S_MAX=S_MAX, logS_MAX=logS_MAX,plambda=plambda,
		dtype=dtype,diag=diag,allowExtendedVigilance=allowExtendedVigilance,
		Knuth_variance=Knuth_variance,normalized_classifier_categories=normalized_classifier_categories,
		N0=N0,name=name,usecl=usecl,debug=debug)

		if ctx is None or queue is None:
			raise Exception(self.name+": No OpenCL context/queue given, aborting")
		if not diag:
			raise Exception(self.name+': non-diagonal case must be reimplemented in OpenCL')
		self.ctx=ctx
		self.queue=queue
		self.allocator = cl.tools.MemoryPool(pyopencl.tools.ImmediateAllocator(queue))
		self.allocator = None

# convert arrays allocated in superclass to OpenCL
		self.enable_growth=enable_growth
		self.mu      = cl.array.to_device(queue,self.mu, allocator=self.allocator)
		self.sigma   = cl.array.to_device(queue,self.sigma, allocator=self.allocator)
		# fields computed from others:
		self.logS    = cl.array.to_device(queue, self.logS, allocator=self.allocator) # log(S)
		self.sigi    = cl.array.to_device(queue, self.sigi, allocator=self.allocator)
		self.nj      = cl.array.to_device(queue, self.nj,   allocator=self.allocator)		
		self.P_w     = cl.array.to_device(queue, self.P_w,  allocator=self.allocator)
		# log(P_w)
		self.logP_w  = cl.array.zeros(queue,(self.N0),dtype=self.dtype,allocator=self.allocator)
		self.P_w_a   = cl.array.zeros(queue,(self.N0),dtype=self.dtype,allocator=self.allocator)
		self.logp    = None 
		
		

		# allocate for a single category		
		self.mu_tmp = cl.array.zeros(queue, self.mu.shape, dtype=self.dtype,allocator=self.allocator)
		self.sigma_tmp  = cl.array.zeros(queue,self.sigma.shape,dtype=self.dtype,allocator=self.allocator)
		self.logS_tmp   = cl.array.zeros(queue,(self.N0,),dtype=self.dtype,allocator=self.allocator)
		# allocate 4 items: j, flag_a, k, flag_b
		self.j_buf = cl.array.zeros(queue,(4),dtype=np.int32, allocator=self.allocator)
		self.selection_tmp_1_buf = cl.array.zeros(queue,(1),dtype=np.int32, allocator=self.allocator)
		
		with open ("Art_cl.cl", "r") as myfile:
			source = myfile.read()
			
		deviceName=self.queue.get_info(cl.command_queue_info.DEVICE).name.strip()
		# OpenCL block size. If n>1024, then 1024 is chosen
		# otherwise the next power-of-two(n)
		if max_blocksize is None: 
			max_blocksize=1024
			if deviceName.startswith("Intel"):
				max_blocksize=64 
				
		self.blocksize 	= Art_cl.nextPowerTwo(min(n,max_blocksize))
		self.nblocks 	= Art_cl.roundUpDiv(n, self.blocksize*self.blocksize)
		self.nblocks 	= min(self.nblocks, self.blocksize)		
		self.p_tmp_buf  = cl.array.empty(self.queue,(1,self.N0, self.nblocks),dtype=self.dtype,order='C',allocator=self.allocator)
		self.logS_tmp_1 = cl.array.zeros(self.queue,(self.N0, self.nblocks),dtype=self.dtype,order='C',allocator=self.allocator)		


		self.initSigmaValues =  self.plambda * np.exp(float(self.logS_MAX)/float(self.n))
		if buildProgram:
			red = self.generate_reduction_macro(self.blocksize)		
	#		red = "#define local_reduce_op(op,sdata,n,tid) local_reduce_op_safe(op,sdata,n,tid)"
	#		print 'Art_cl: n=%d blocksize=%d nblocks=%d' % (n, self.blocksize,self.nblocks)
	

			if deviceName.startswith("I") and False:
				source = "#define TRACE(x) printf x\n" + source
			else:
				source = "#define TRACE(x) \n" + source
			source = ("#define allowExtendedVigilance %d\n" % self.allowExtendedVigilance) + source
			if self.dtype == np.float64:
				source = "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"+source
			if self.Knuth_variance:
				source = "#define USE_KNUTH_VARIANCE 1\n" + source
			source = source.replace('REDUCE_blockSize', str(self.blocksize))
			source = source.replace('_LOCAL_REDUCE_OP_PLACEHOLDER_',  red)
			if self.diag:
				source = source.replace('_REDUCE_KERNEL_GLOBAL_ADDER_', 'REDUCE_KERNEL_GLOBAL_ADDER_DIAG')
			else:
				source = source.replace('_REDUCE_KERNEL_GLOBAL_ADDER_', 'REDUCE_KERNEL_GLOBAL_ADDER_MATRIX')
			source = source.replace('__DATA_TYPE__', cl.tools.dtype_to_ctype(self.dtype))
			
			
			self.prg = cl.Program(ctx, source)
			self.prg.build()
			self.logpdf_stage_1 = self.prg.logpdf_stage_1
			self.logpdf_stage_1.set_scalar_arg_dtypes([None, np.int32,np.int32,None,None,None,np.int32,None,np.int32, np.int32,np.int32, np.int32])
			self.logpdf_stage_2 = self.prg.logpdf_stage_2
			self.logpdf_stage_2.set_scalar_arg_dtypes([None,None,None,np.int32,None,np.int32,np.int32, np.int32, np.int32])
	
			self.scal_dev_kernel 	 = self.prg.scal_dev_kernel
			self.scal_inv_dev_kernel = self.prg.scal_inv_dev_kernel
			
			self.update_mu_sigma_kernel = self.prg.update_mu_sigma_kernel
			self.update_mu_sigma_kernel.set_scalar_arg_dtypes(
	            [None,np.int32,None,None,None,None,None,None,None,np.int32,None,np.int32,np.int32, np.int32])
	
			self.copy_mu_sigma_kernel = self.prg.copy_mu_sigma_kernel
			self.copy_mu_sigma_kernel.set_scalar_arg_dtypes([None,None,None,None,None,None,None,None,None,None,np.int32,np.int32,np.int32, np.int32])
			
			self.create_new_category_kernel = self.prg.create_new_category_kernel
			self.create_new_category_kernel.set_scalar_arg_dtypes([None, np.int32,None,None,None,None,None,None,None,self.dtype,np.int32,np.int32,np.int32, np.int32])
	
			self.compute_new_logS_kernel_stage_1 = self.prg.compute_new_logS_kernel_stage_1
			self.compute_new_logS_kernel_stage_1.set_scalar_arg_dtypes([None,np.int32,None,None,None,None,None,np.int32,np.int32, np.int32, np.int32])
			
			self.compute_new_logS_kernel_stage_2 = self.prg.compute_new_logS_kernel_stage_2
			self.compute_new_logS_kernel_stage_2.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32, np.int32])
	
			self.predict_kernel = self.prg.predict_kernel
			self.predict_kernel.set_scalar_arg_dtypes([None,None,None, np.int32,np.float32, None,None])
			
			
			self.art_fit_kernel = self.prg.art_fit_kernel
			self.art_fit_kernel.set_scalar_arg_dtypes([
				None, np.int32,None,None,
				None,None,None,None,
				None,None,None,None,
				None,None,None,np.int32, # ... 16:j0
				None,np.int32,np.int32,np.int32,
				np.float32,self.dtype,np.int32, 
				np.int32])
	
	
			self.bar_fit_kernel = self.prg.bar_fit_kernel
			self.bar_fit_kernel.set_scalar_arg_dtypes([
				None, np.int32,None,None,
				None, None,None,None,
				None, None,None,None,
				None, None,None,np.int32, 		# ... 16:j0
				None, np.int32,np.int32,np.int32,#j_indicies, use, Nsel,Na ...20
				np.float32, self.dtype, np.int32, # ... nsamples 23
				np.int32, # Na max 24
				None, np.int32,	# B, offset
				None, # g_nj_b,  27
				None, # g_mu_b,  28
				np.int32, 		# Nb 29 
				np.int32, np.int32, #  max, ldw
				None,None,None,
				np.int32, np.int32, # n_a, n_b
				self.dtype	# b_min
				])

			self.art_update_nj_P_w_kernel=self.prg.art_update_nj_P_w_kernel
			self.art_update_nj_P_w_kernel.set_scalar_arg_dtypes([None,np.int32,None,None,None,np.int32,np.int32])
			
			
			self.bar_fit_regression_kernel = self.prg.bar_fit_regression_kernel
	
			self.bar_fit_regression_kernel.set_scalar_arg_dtypes([
				None, np.int32,None,None,
				None, None,None,None,
				None, None,None,None,
				None, None,None,np.int32, 		# ... 16:j0
				None, np.int32,np.int32,np.int32,#j_indicies, use, Nsel,Na ...20
				np.float32, self.dtype, np.int32, # ... nsamples 23
				np.int32, 	# 24 Na max
				None, 		# const __global TYPE * restrict B,  	// 25. in
				np.int32, 	# int B_offset,              			// 26
				None, # __global TYPE * restrict g_nj_b,	// 27
				None, # __global TYPE * restrict g_mu_b,	// 28
				None, # __global TYPE * restrict g_sigma_b, // 29
				None, # __global TYPE * restrict g_sigi_b,  // 30
				None, # __global TYPE * restrict logS_b,    // 31
				None, # __global TYPE * restrict g_mu_b_tmp,    // 32. [Nxn] out
				None, # __global TYPE * restrict g_sigma_b_tmp, // 33. [Nxn] out
				None, # __global TYPE * restrict logS_b_tmp,    // 34. [N] out
				None, # __global TYPE * restrict log_p_b_w,		// 35 [N] tmp 
				None, # __global TYPE * restrict logP_b_w,		// 36 [N] tmp
				None, # __global TYPE * restrict P_prior_b,		// 37 [N] tmp
				None, # __global TYPE * restrict P_w_b,			// 38 [N] out: normalized prob. field P(w_j|a) eq. (3)
				np.int32, # int Nb,								// 39
				np.int32, # int Nb_max,							// 40
				np.float32, # float logS_MAX_b,					// 41
				self.dtype, # TYPE initSigmaValues_b,				// 42
				np.int32, 	# int ldw,							// 43 stride of w, P_b_a matrices
				None, 		# __global TYPE * w,			// 44 matrix of [Na_max x Nb_max]
				None, 		# __global TYPE * P_b_a,		// 45 matrix of [Na_max x Nb_max]
				None, 		# __global TYPE * sum_w_j		// 46 vector of [Na_max]
				np.int32,	# n_a 
				np.int32	# n_b
			])
	
			
			self.bar_update_mapfield_kernel=self.prg.bar_update_mapfield_kernel
			self.bar_update_mapfield_kernel.set_scalar_arg_dtypes([np.int32,np.int32,np.int32,np.int32, None,None,None])
	
			
			
			self.copy_matrix_kernel = self.prg.copy_matrix_kernel
			self.copy_matrix_kernel.set_scalar_arg_dtypes([None, np.int32, None, np.int32, np.int32, np.int32])
		return 

	def __str__(self):
		s = Art.Art.__str__(self)
#		s += ' blocksize='+str(self.blocksize)
#		s += ' ' + str(self.ctx.get_info(cl.context_info.DEVICES)[0].name)
#		s += ' ' + 'nj.data=' + str(self.nj.data) + ' P_w.data=' + str(self.P_w.data)
		return s
	
	"""
	def __del__(self):
		print 'Art_cl del',self.name
		if self.allocator is not None:
			del self.allocator
			self.allocator = None
		Art.Art.__del__(self)
	"""
	
	@staticmethod
	def roundUp(a,b):
		"""make a multiple of b"""
		return long(np.ceil(float(a)/float(b))*b)
	
	@staticmethod
	def roundUpDiv(a,b):
		return int(np.ceil(float(a)/float(b)))
	
	@staticmethod
	def nextPowerTwo(n):
		return 2**int(math.ceil(math.log(n, 2)))
	
	def set_row(self,X,x,j):
		"""Assuming X a matrix and x a vector, sets the j-th row of the matrix.
		If X is a vector and x is a scalar, then sets the j-th element
		In a special case it can append a last element to the array (extend the 0-th axis)
		
		Two cases: 
			either
				ndim(X) == ndim(x)+1 in which x is expanded by one dimension
			or
				ndim(X) == ndim(x) in which case x.shape[0] must be 1 
		returns a pyopencl array
		
		"""
#		if self.debug>=0:print 'set_row j=',j,'X=',type(X), repr(X),'x=',type(x),repr(x)
		newshape=None
		assert(isinstance(j,int))

		if not hasattr(x, 'shape'):
			x=self.intern(x)
		
		# compute new shape of x. note: ndim is not defined for pyopencl arrays
		if len(X.shape)==len(x.shape)+1:	newshape=(1,) + shape(x)
		elif len(X.shape)==len(x.shape):	newshape=shape(x)
		else: raise Exception("set_row incompatible shapes: %s and %s " % (str(shape(X)),str(shape(x))))
		if newshape[0]!=1:raise Exception("set_row first dimension of x must be singleton instead of shape(X)=%s, shape(x)=%s " % (str(X.shape), str(x.shape)))
		if j<X.shape[0] and len(newshape)>1:
			newshape=newshape[1:]	# remove first singleton dimension in case we set items
			
		x=self.intern(x,newshape)

#		if self.debug>=0:print 'set_row j=',j,'X=',type(X), repr(X),'x=',type(x),repr(x)
		
		if j<X.shape[0]:
			# special case if x is scalar, pyopencl needs treatment
			if (len(x.shape)==1) and (x.shape[0]==1):
				X[j]=float(x[0].get())
			else:
				X[j]=x
#			subarray=X[j]
#			print 'subarray ',type(subarray),shape(subarray),repr(subarray)
			
		elif j==X.shape[0]: # extend one row
			print 'WARNING, set_row SLOW PATH, consider increasing N0 above %d ' % (j)			
			X = cl.array.concatenate((X, x), axis=0)
		else:
			raise Exception("set_row can extend by exactly one row X.shape[0]=%d, j=%d given" % (X.shape[0],j))

#		print 'set_row returns ',type(X),repr(X)
		
		return X

	def ensure_numrows(self,X, newshape, fillZeros=False):
		"""ensures that X has at lest the given shape
		X must be an opencl array
		"""
		M = X.shape[0] if X is not None else 0
		dtype = X.dtype if X is not None else self.dtype
		if M >= newshape[0]:
			return X
#		print 'ensure_numrows old shape=',X.shape,'newshape=',newshape
#		print 'old data:\n',X
		if fillZeros:
			Y = cl.array.zeros(self.queue, newshape, dtype=dtype)
		else:
			Y = cl.array.empty(self.queue, newshape, dtype=dtype)
		if X is not None and M>0:
			Y[0:M,] = X[0:M,] # copy
#		print 'copied data:\n',Y
		return Y
	



	def intern(self,x, newshape=None):
		"""
		convert x to the internal format, optionally reshape it
		Creates a cl.array.Array from x (returns it unmodified if already such one).
		Optionally reshapes the new array to the specified shape
		"""
#		print 'intern ',repr(x)
		if not hasattr(x, 'shape'):
			x = np.array(x,dtype=self.dtype)
			
		if newshape is not None and x.shape!=newshape and x.shape!=():
			x=x.reshape(newshape)
#		print 'intern ',type(x),repr(x)
		if not isinstance(x, cl.array.Array):
			if x.shape == (): # x is a scalar?
				x=cl.array.to_device(self.queue, np.array([x],dtype=self.dtype),allocator=self.allocator)
			else:
				x=cl.array.to_device(self.queue, x.astype(self.dtype),allocator=self.allocator)
#		print 'intern return x=',type(x),repr(x)
		return x
		
	def extern(self, x):
		"""convert x from internal representation to numpy array"""
		if isinstance(x,cl.array.Array):
			return x.get(queue=self.queue)
		return x
		
	def logpdf(self,x,selection=None):
		""" compute logarithm of multivariate normal distribution between an input vector x (row)
			and a set of parameters mu and sigma. Sigma is considered a diagonal matrix 
		if x is a matrix, then each logpdf using each row is computed independently
		"""
		
				
#		print 'Nsel=',Nsel,' type=',self.selection.dtype
		n = self.n
		N = self.N
		# L = number of inputs
		inshape=x.shape
		L = 1 if len(x.shape)<2 else x.shape[0]
		x = self.intern(x, newshape=(L,n))
		
		if selection is not None:
			self.selection = cl.array.to_device(self.queue, selection.astype(np.int32),allocator=self.allocator)
			Nsel = selection.size
		else:
			Nsel = N
		
		# we pass a real selection index buffer to the OpenCL kernel only if Nsel<N
		# otherwise we consider the entire array is selected, and no selection buffer needed
		# However, to avoid passing null pointers, we send selection_tmp_1_buf.data in that case.
		selection_ptr = self.selection.data if (Nsel<N) else self.selection_tmp_1_buf.data 
		use_selection_ptr = (Nsel<N)
		lws		= [self.blocksize,1,1]		
		gws		= [self.nblocks * lws[0], Nsel, L]
		if self.debug>=5:
			print 'logpdf N=',N,'Nsel=',Nsel,'selection_ptr=',selection_ptr,'use_selection_ptr=',use_selection_ptr,\
			'gws=',gws,'lws=',lws,'nblocks=',self.nblocks,'blocksize=',self.blocksize
		if (self.p_tmp_buf is None) or (self.p_tmp_buf.shape[1] < N) or (self.p_tmp_buf.shape[0] < L):
			N0 = max(N, self.N0)
			self.p_tmp_buf      = cl.array.empty(self.queue,(L, N0, self.nblocks),dtype=self.dtype,order='C',allocator=self.allocator)

		if (self.logp is None) or (len(inshape)>len(self.logp.shape)) or (self.logp.size < N*L):
			N0 = max(N, self.N0)
			self.logp		= cl.array.empty(self.queue, (L,N0,), self.dtype, order='C',allocator=self.allocator)
#			print 'ALLOCATED N=',self.N,'p_a_w',self.p_a_w.shape

		# fill the array in case not all rows where selected 
		if Nsel < N:
			self.logp.fill(self.log0, queue=self.queue)

		if Nsel > 0:
			if self.debug>=5:
				print 'logpdf_stage_1: x=',x
				print 'mu=',self.mu
				print 'sigi=',self.sigi
			assert len(x.shape)==2
			assert self.p_tmp_buf.strides[1]//self.dtype_size==self.nblocks
			events = [None,None]
			events[0] = self.logpdf_stage_1(self.queue, gws, lws, 
						x.base_data, x.offset//self.dtype_size,x.strides[0]//self.dtype_size,
						self.mu.data, self.sigi.data,
						selection_ptr, use_selection_ptr, 
						self.p_tmp_buf.data, N,
						self.p_tmp_buf.strides[0]//self.dtype_size, 
						self.p_tmp_buf.strides[1]//self.dtype_size,
						self.n)
			
			lws2 = [self.nblocks,1,1]
			gws2 = [self.nblocks,Nsel,L]
			
			if self.debug>=5:
 				print 'p_tmp_buf after stg 1:',self.p_tmp_buf
				print 'self.logpdf_stage_1: ',gws2,self.nblocks, self.p_tmp_buf
				print 'logS=',self.logS
				
			assert self.nblocks == self.p_tmp_buf.strides[1]//self.dtype_size
			
			events[1] = self.logpdf_stage_2(self.queue, gws2, lws2, self.p_tmp_buf.data,
								  self.logS.data,
								  selection_ptr, use_selection_ptr,								 
								  self.logp.data,
								  self.p_tmp_buf.strides[0]//self.dtype_size, 
								  self.p_tmp_buf.strides[1]//self.dtype_size,
								  self.logp.strides[0]//self.dtype_size,
								  self.n
								  )
			
			self.duration_cl_fit_bar += benchmark.profile_cl('logpdf'+self.profilesuffix ,events,lws=lws)
#			print 'event time=',time
			if self.debug>=5:
				print 'logpdf_stage_2 result logp=',self.logp
			if self.debug>=4: self.verify_logpdf(self.extern(x),self.extern(self.logp),allselected=~use_selection_ptr,selection=selection)
		
		# bring result to match input x's shape
		if len(inshape)>1:
			assert self.logp.shape[0] == L
			assert self.logp.shape[1] >= N
			return self.logp[:L,:N]
		else:
			return self.logp
		
	@staticmethod
	def next_pow_2(x):
		return int(2**math.ceil(math.log(x)/math.log(2.0)))
	
	def call_predict_kernel(self,logS_MAX=None):
		N	= self.N
		if N<1: return -1
		lws	= min(self.blocksize, Art_cl.next_pow_2(N))
		lws	= (lws,)
		gws	= lws
		logS = self.logS if self.allowExtendedVigilance else self.logS_tmp
		
		if (self.P_w_a is None) or (self.P_w_a.size < N):
			N0 = max(N, self.N0)
			self.P_w_a		= cl.array.zeros(self.queue, (N0,), self.dtype,allocator=self.allocator)
		if logS_MAX is None: logS_MAX=self.plus_inf
		events=[None]
		events[0]=self.predict_kernel(self.queue, gws, lws, 
								self.logp.data,
								logS.data, 
								self.logP_w.data, 
								self.N, logS_MAX,
								self.j_buf.data, self.P_w_a.data)
		dur=benchmark.profile_cl('predict'+self.profilesuffix,events,lws=lws)
		self.duration_cl_fit_bar += dur
#		print 'pid=%d id=%d %s cl=%d: predict N=%4d lws=%s dur=%8.3f us' % (os.getpid(), id(self),self.name, self.usecl,N,lws,dur*1E6)
		if self.debug>=4:
			print 'predict_kernel input logS_MAX=',logS_MAX,'logS=',logS,'returns j=',self.j_buf, 'P_w_a=',self.P_w_a
		return int(self.j_buf.get(self.queue)[0])
	
	def predict1(self,a,logS_MAX=None,selection=None):
		N = self.N
		# allocate output buffer
		if N < 1 or (selection is not None and len(selection)<1):
			j = -1
		else:		
			self.logpdf(a,selection=selection)
			j = self.call_predict_kernel(logS_MAX=logS_MAX)
		
		return j
	
	def predict(self,A,logS_MAX=None,selection=None):
		"""NOT called during training.
		if logS_MAX=None is given, +inf is assumed (no vigilance test)
		"""
		N = self.N

		if logS_MAX is None:
			logS_MAX = self.plus_inf
			
		if A.size==self.n:
			A=A.reshape(1,self.n)

		nsamples = A.shape[0]
		assert(A.shape[1] == self.n)
		if nsamples>1:
			jj=np.empty(nsamples,dtype=np.int32)
		else:
			jj=None
			
		for i in xrange (nsamples):
			a=A[i]
			j = self.predict1(a,logS_MAX,selection)
			if nsamples > 1:
				jj[i] = j
			else:
				return j
		return jj


	def new_category(self,a):
		"""The method new_category feed the first pattern a to an empty ART, which will create the first category
			
			Keyword arguments:
				a -- array
			Result:
				j -- index of new category
		"""
		
		if self.N >= self.N0 and not self.enable_growth:
			raise Art.TrainingOverflowException(self.N0)
		
		a = self.intern(a) # make sure a is row vector

		j=int(self.N)
		self.N = self.N+1
		N = self.N
		n = self.n		
		if self.debug > 1: print '     %s: new category: %d with input:%s' % (self.name,j, str(a))
		self.sdim = (N,) + self.sdim[1:]
		self.mu    = self.ensure_numrows(self.mu,    (N, n))
		self.sigma = self.ensure_numrows(self.sigma, self.sdim)
		self.sigi  = self.ensure_numrows(self.sigi,  self.sdim)
		self.logS  = self.ensure_numrows(self.logS,  (N,))
		self.mu_tmp    = self.ensure_numrows(self.mu_tmp,    (N, n))
		self.sigma_tmp = self.ensure_numrows(self.sigma_tmp, self.sdim)
		self.logS_tmp  = self.ensure_numrows(self.logS_tmp,  (N,))
		self.P_w   = self.ensure_numrows(self.P_w,   (N,))
		self.logP_w= self.ensure_numrows(self.logP_w,(N,))
		self.nj    = self.ensure_numrows(self.nj,    (N,), fillZeros=True)
		
		gws	= (self.blocksize,)
		lws	= None
		f =  self.plambda * np.exp(float(self.logS_MAX)/float(n))
#		print 'new_category j=',j,'n=',n,'data=',a.base_data, 'offset=',a.offset
		events=[ None, ]
		events[0]=self.create_new_category_kernel(self.queue, gws, lws,
			a.base_data, a.offset/self.dtype_size,
			self.nj.data, self.mu.data, self.sigma.data, self.sigi.data,self.logS.data,
			self.P_w.data, self.logP_w.data,
			f, j, N, self.nsamples,
			self.n)
			
		self.duration_cl_fit_bar += benchmark.profile_cl('create_new_category'+self.profilesuffix,events,lws=lws)
		
		self.nsamples += 1
		return j
	
	def fit1_normalized_classifier_categories(self,a):
		"""Special case of fit1 when using normalized category labels: a=0,1,..N-1"""

		lws=None
		gws=(self.N,)
		assert self.N > 1
		assert self.N == self.N0
		assert self.n==1

		events=[None]
		events[0] = self.art_update_nj_P_w_kernel(self.queue, gws, lws,
			a.base_data, a.offset/self.dtype_size,
			self.nj.data, 
			self.P_w.data,
			self.logP_w.data, 
			self.N,
			self.nsamples)

		self.duration_cl_fit_bar += benchmark.profile_cl('fit'+self.profilesuffix,events,lws=lws)
		self.nsamples	=	self.nsamples + 1
 		return 0



	def fit1(self,a):
		if self.normalized_classifier_categories:
			return self.fit1_normalized_classifier_categories(a)

		N = self.N+1  # do not commit yet the increment
		n = self.n
				
		self.sdim = (N,) + self.sdim[1:]
		self.mu    = self.ensure_numrows(self.mu,    (N, n))
		self.sigma = self.ensure_numrows(self.sigma, self.sdim)
		self.sigi  = self.ensure_numrows(self.sigi,  self.sdim)
		self.logS  = self.ensure_numrows(self.logS,  (N,))
		self.mu_tmp    = self.ensure_numrows(self.mu_tmp,    (N, n))
		self.sigma_tmp = self.ensure_numrows(self.sigma_tmp, self.sdim)
		self.logS_tmp  = self.ensure_numrows(self.logS_tmp,  (N,))
		self.P_w   	= self.ensure_numrows(self.P_w,   (N,))
		self.P_w_a	= self.ensure_numrows(self.P_w_a,   (N,))
		self.logP_w	= self.ensure_numrows(self.logP_w,(N,))
		self.logp	= self.ensure_numrows(self.logp,    (N,))
		self.nj		= self.ensure_numrows(self.nj,    (N,))

		N = self.N # restore
		Nsel = N
		selection_ptr = self.selection_tmp_1_buf.data 
		use_selection_ptr = 0


		lws=(self.blocksize,)
		gws=lws
		
		events=[None]
		events[0] = self.art_fit_kernel(self.queue, gws, lws,
			a.base_data, a.offset/self.dtype_size,
			self.nj.data, self.mu.data, self.sigma.data, self.sigi.data,self.logS.data,
			self.mu_tmp.data, self.sigma_tmp.data, self.logS_tmp.data,
			self.logp.data, self.logP_w.data, self.P_w.data, self.P_w_a.data,
			self.j_buf.data,
			0, #j0
			selection_ptr,use_selection_ptr,
			Nsel,N,
			self.logS_MAX,
			self.initSigmaValues,
			self.nsamples,
			self.n)		
		
		self.duration_cl_fit_bar += benchmark.profile_cl('fit'+self.profilesuffix,events,lws=lws)
		self.nsamples +=1
		if self.j_buf[1]: self.N+=1
		
		self.sdim = (N,) + self.sdim[1:]
		
		return self.j_buf[0]




	def ensure_matrix_size(self, w,Na,Nb):
		"""
			resize matrix w to at least Na x Nb
			Na = rows
			Nb = columns
			if w is already of this size (or bigger) then it is returned unmodified
        """
		N = w.shape[0] # existing rows
		M = w.shape[1] # existing cols
		
		if (N >= Na) and (M >= Nb):
			return w
		v = cl.array.empty(self.queue,(max(Na,N), max(Nb,M)), dtype=self.dtype, allocator=self.allocator)
		
		# subsititute of v[:N, :M] = w
		gws=(v.shape[0],v.shape[1])
		lws=None
		events = [ self.copy_matrix_kernel(self.queue,gws,lws, w.data, w.shape[1], v.data, v.shape[1], w.shape[0],w.shape[1]) ]
		self.duration_cl_fit_bar += benchmark.profile_cl('copy_matrix'+self.profilesuffix,events,lws=lws)
		return v

	
	
	def update_mu_sigma_tmp(self,a,selection=None):
		N = self.N
		a = self.intern(a)
		j0=0	# offset of first category
		if selection is not None:
			Nsel = len(selection)
			if Nsel==1: 
				j0=selection[0]
				if j0<0: return
			else:
				self.selection = cl.array.to_device(self.queue, selection.astype(np.int32),allocator=self.allocator)
		else:
			Nsel = N
		
		# we pass a real selection index buffer to the OpenCL kernel only if Nsel>1 and Nsel<N
		# Nsel=1 is special case handled by passing the j0 = j and avoid global mem  
		# otherwise we consider the entire array is selected, and no selection buffer needed
		# However, to avoid passing null pointers, we send selection_tmp_1_buf.data in that case.
		use_selection_ptr = (Nsel>1 and Nsel<N)
		selection_ptr = self.selection.data if use_selection_ptr else self.selection_tmp_1_buf.data 
		
#		print 'j0=',j0,'Nsel=',Nsel
		lws		= (self.blocksize,1)		
		gws		= (self.blocksize,Nsel)
		events  = [ self.update_mu_sigma_kernel(self.queue, gws, lws, 
			a.base_data, a.offset/self.dtype_size,
			self.nj.data,
			self.mu.data,     self.sigma.data,     self.logS.data,
			self.mu_tmp.data, self.sigma_tmp.data, self.logS_tmp.data, 
			j0,selection_ptr,use_selection_ptr,Nsel, 
			self.n) ]
			
		self.duration_cl_fit_bar += benchmark.profile_cl('update_mu_sigma'+self.profilesuffix,events,lws=lws)
		
		if self.debug>=4: 
			print self.name,'    after update_mu_sigma_kernel a=',a
			print '          mu=\n',self.mu
			print '          sigma=\n',self.sigma 
			print '          sigi=\n',self.sigi
			print '          logS=',self.logS 
			print '          mu_tmp=',self.mu_tmp
			print '          sigma_tmp=',self.sigma_tmp
			print '          logS=',self.logS_tmp

		return
	
	def copy_mu_sigma(self,j):
		lws		= None
		gws		= (self.n,1)
		events = [ self.copy_mu_sigma_kernel(
				self.queue,gws,lws,
				self.nj.data,		# 1. [N]
				self.mu.data,       # 2. [N x n]
				self.sigma.data,    # 3. [N x n]
				self.sigi.data,     # 4. [N x n]
				self.logS.data,     # 5. [N]
				self.P_w.data,      # 6. [N]
				self.logP_w.data,   # 7. [N]
				self.mu_tmp.data,   # 8. [Nxn] 
				self.sigma_tmp.data,# 9. [Nxn] 
				self.logS_tmp.data, # 10. [N] 
				j, 					# 11. winner category number
				self.N, 			# 12. number of categories
				self.nsamples, 		# 13. number of samples seen previously
				self.n
				) ]
		self.duration_cl_fit_bar += benchmark.profile_cl('copy_mu_sigma'+self.profilesuffix,events,lws=lws)
		return
	
	def commit(self,a,j):
		self.update_mu_sigma_tmp(a,selection=[j,])
		self.copy_mu_sigma(j)
		self.nsamples+=1
		return
	
	def compute_new_logS(self,a,selection=None):
		N = self.N
		a = self.intern(a)
		
		if selection is not None:
			self.selection = cl.array.to_device(self.queue, selection.astype(np.int32),allocator=self.allocator)
			Nsel = selection.size
		else:
			Nsel = N
			
		if Nsel < 1: return
		
		# we pass a real selection index buffer to the OpenCL kernel only if Nsel<N
		# otherwise we consider the entire array is selected, and no selection buffer needed
		# However, to avoid passing null pointers, we send selection_tmp_1_buf.data in that case.
		selection_ptr = self.selection.data if (Nsel<N) else self.selection_tmp_1_buf.data 
		use_selection_ptr = (Nsel<N)

		
		lws		= [self.blocksize,1]
		gws		= [self.nblocks * lws[0], Nsel]
		
		if (self.logS_tmp_1 is None) or (self.logS_tmp_1.shape[0] < N):
			N0 = max(N, self.N0)
			self.logS_tmp_1 = cl.array.empty(self.queue,(N0, self.nblocks),dtype=self.dtype,order='C',allocator=self.allocator)

		events=[None,None]
		events[0]=self.compute_new_logS_kernel_stage_1(
			self.queue,gws,lws,
			a.base_data, a.offset/self.dtype_size,
			self.nj.data,
			self.mu.data,		
			self.sigma.data,	
			self.logS_tmp_1.data, 
			selection_ptr,use_selection_ptr,Nsel,
			self.nblocks, self.n)
		
		lws2 = [self.nblocks,1]
		gws2 = [self.nblocks,Nsel]
		
		events[1]=self.compute_new_logS_kernel_stage_2(self.queue, gws2, lws2, 
			self.logS_tmp_1.data,
			self.logS_tmp.data,
			selection_ptr, 
			use_selection_ptr,
			Nsel,								 
			self.nblocks)
			
		self.duration_cl_fit_bar += benchmark.profile_cl('compute_new_logS'+self.profilesuffix,events,lws=lws)
		
		return
		
	def choose(self,a,selection=None,logS_MAX=None):
		if logS_MAX is None: logS_MAX=self.logS_MAX;
		self.compute_new_logS(a,selection=selection)
		self.logpdf(a,selection=selection)
		j = self.call_predict_kernel(logS_MAX=logS_MAX)
		return j




	def reorder(self):
		"""Re-orders categories based by P_w"""
		if self.N<2: return
		ix = np.argsort(self.extern(self.P_w[0:self.N]))[::-1]
		self.P_w 	= self.intern(self.P_w.get()[ix])
		self.nj 	= self.intern(self.nj.get()[ix])
		self.logS 	= self.intern(self.logS.get()[ix])
		self.mu 	= self.intern(self.mu.get()[ix,])
		self.sigma 	= self.intern(self.sigma.get()[ix,])
		self.sigi 	= self.intern(self.sigi.get()[ix,])
		

	def generate_reduction_macro(self,n):
		"""generates an OpenCL macro that does parallel reduction, taking account the device SIMD width 
		   to avoid  barriers where not needed. Inspired from PyOpencl's reduction.
		   
		   param n = block size (power of 2)
		   """
		assert (n & (n-1))==0
		from pyopencl.characterize import get_simd_group_size
		dev = self.queue.get_info(cl.command_queue_info.DEVICE)
		simd_width = pyopencl.characterize.get_simd_group_size(dev, self.dtype_size)
		s="""#define local_reduce_op(op,sdata,n,tid) {\\
			barrier(CLK_LOCAL_MEM_FENCE);\\
			TYPE mySum = sdata[tid]; \\
			"""
		
		barrier = True
		while n>1:
			if n <= 2*simd_width and barrier:
				barrier = False
				s += """if (tid < %d){\\
					volatile __local TYPE *smem = sdata;\\
				""" % (n//2)

				
			if barrier:
				s += """if (n >= %4d){\\
					if (tid < %4d) sdata[tid] = mySum = op(mySum, sdata[tid + %d]);\\
					barrier(CLK_LOCAL_MEM_FENCE);\\
					}\\
				""" % (n, n//2, n//2) 
			else:
				s += """	if (n >= %4d) smem[tid] = mySum = op(mySum, smem[tid + %4d]);\\
				""" % (n,n//2)



			n = n // 2
		if not barrier:
			s+="""}\\
			"""
		s+="""}
		"""
		return s


	def allocbuffers(self):
		N0 = max(self.N+1,self.N0)
		n = self.n

		# resize art_a
		self.sdim = (N0,) + self.sdim[1:]
		self.mu    = self.ensure_numrows(self.mu,    (N0, n))
		self.sigma = self.ensure_numrows(self.sigma, self.sdim)
		self.sigi  = self.ensure_numrows(self.sigi,  self.sdim)
		self.logS  = self.ensure_numrows(self.logS,  (N0,))
		self.mu_tmp    = self.ensure_numrows(self.mu_tmp,    (N0, n))
		self.sigma_tmp = self.ensure_numrows(self.sigma_tmp, self.sdim)
		self.logS_tmp  = self.ensure_numrows(self.logS_tmp,  (N0,))
		self.P_w   	= self.ensure_numrows(self.P_w,   (N0,))
		self.P_w_a	= self.ensure_numrows(self.P_w_a,   (N0,))
		self.logP_w	= self.ensure_numrows(self.logP_w,(N0,))
		self.logp	= self.ensure_numrows(self.logp,  (N0,))
		self.nj		= self.ensure_numrows(self.nj,    (N0,), fillZeros=True)
		return

	# ------------------------------------------------------------------------
	# --- BAR related functions whose OpenCL implementation was moved here ---
	# ------------------------------------------------------------------------
	def bar_fit1(self,a,b, art_b, w,P_b_a,sum_w_j, regression=False):
		N0 = max(self.N+1,self.N0)
		n = self.n
		self.allocbuffers()
		
		N = self.N # restore
		Nsel = N
		selection_ptr = self.selection_tmp_1_buf.data 
		use_selection_ptr = 0
		self.N0 = N0
		self.sdim = (self.N,) + self.sdim[1:]		# restore sdim
		
		# resize art_b
		k	= int(b.get()[0])
		nb	= art_b.n
		N0b = max(art_b.N0, k+1)
		art_b.nj = art_b.ensure_numrows(art_b.nj,    (N0b, ), fillZeros=True)
		art_b.mu = art_b.ensure_numrows(art_b.mu,    (N0b, nb))
		art_b.N0 = N0b
		# resize mapfield
		w			= self.ensure_matrix_size(w, N0, N0b)
		P_b_a		= self.ensure_matrix_size(P_b_a, N0, N0b)
		sum_w_j		= self.ensure_numrows(sum_w_j, (N0,))

		#
		lws=(self.blocksize,)
		gws=lws
		
		events=[None]
		if not regression:
			events[0] = self.bar_fit_kernel(self.queue, gws, lws,
			a.base_data, a.offset/self.dtype_size,
			self.nj.data, self.mu.data, self.sigma.data, self.sigi.data,self.logS.data,
			self.mu_tmp.data, self.sigma_tmp.data, self.logS_tmp.data, # ... 10
			self.logp.data, self.logP_w.data, self.P_w.data, self.P_w_a.data,
			self.j_buf.data, # 15
			0, # 16 j0
			selection_ptr,use_selection_ptr,
			Nsel,N,
			self.logS_MAX, #31
			self.initSigmaValues,
			self.nsamples,
			
			self.N0,							# 24. max Na allowed
			b.base_data, b.offset/self.dtype_size,
			art_b.nj.data,
			art_b.mu.data,  # 28
			art_b.N,		# 29
			art_b.N0,
			w.shape[1],
			w.data,			# matrix of [Na_max x Nb_max]
			P_b_a.data,		# matrix of [Na_max x Nb_max]
			sum_w_j.data,	# vector of [Na_max]
			self.n, art_b.n, art_b.b_min				
			)	
		else:
			art_b.allocbuffers()
			if self.debug>=5:
				print 'invoking bar_fit_regression_kernel gws=',(
					 gws, lws,
					 a.base_data, a.offset/self.dtype_size,
					 self.nj.data, self.mu.data, self.sigma.data, self.sigi.data,self.logS.data,
					 self.mu_tmp.data, self.sigma_tmp.data, self.logS_tmp.data, # ... 10
					 self.logp.data, self.logP_w.data, self.P_w.data, self.P_w_a.data,
					 self.j_buf.data, # 15
					 0, # 16 j0
					 selection_ptr,use_selection_ptr,
					 Nsel,N,
					 self.logS_MAX, #31
					 self.initSigmaValues,
					 self.nsamples,
					 
					 self.N0,							# 24. max Na allowed
					 b.base_data, b.offset/self.dtype_size,
					 art_b.nj.data,
					 art_b.mu.data,  	# 28
					 art_b.sigma.data, 	# 29
					 art_b.sigi.data, 	# 30
					 art_b.logS.data,	# 31
					 art_b.mu_tmp.data,	# 32. [Nxn] out
					 art_b.sigma_tmp.data,	# 33. [Nxn] out
					 art_b.logS_tmp.data,   	# 34. [N] out
					 art_b.logp.data,		# 35 [N] tmp 
					 art_b.logP_w.data,		# 36 [N] tmp
					 art_b.P_w.data,			# 37 [N] tmp
					 art_b.P_w_a.data,		# 38 [N] out: normalized prob. field P(w_k|b)
					 art_b.N,				# 39
					 art_b.N0,				# 40
					 art_b.logS_MAX,			# 41
					 art_b.initSigmaValues,	# 42
					 # and now, the map field
					 w.shape[1],		# 43
					 w.data,			# matrix of [Na_max x Nb_max]
					 P_b_a.data,		# matrix of [Na_max x Nb_max]
					 sum_w_j.data,	# vector of [Na_max],
					 self.n, art_b.n)
			
			events[0] = self.bar_fit_regression_kernel(self.queue, gws, lws,
				a.base_data, a.offset/self.dtype_size,
				self.nj.data, self.mu.data, self.sigma.data, self.sigi.data,self.logS.data,
				self.mu_tmp.data, self.sigma_tmp.data, self.logS_tmp.data, # ... 10
				self.logp.data, self.logP_w.data, self.P_w.data, self.P_w_a.data,
				self.j_buf.data, # 15
				0, # 16 j0
				selection_ptr,use_selection_ptr,
				Nsel,N,
				self.logS_MAX, #31
				self.initSigmaValues,
				self.nsamples,
			
				self.N0,							# 24. max Na allowed
				b.base_data, b.offset/self.dtype_size,
				art_b.nj.data,
				art_b.mu.data,  	# 28
				art_b.sigma.data, 	# 29
				art_b.sigi.data, 	# 30
				art_b.logS.data,	# 31
				art_b.mu_tmp.data,	# 32. [Nxn] out
				art_b.sigma_tmp.data,	# 33. [Nxn] out
				art_b.logS_tmp.data,   	# 34. [N] out
				art_b.logp.data,		# 35 [N] tmp 
				art_b.logP_w.data,		# 36 [N] tmp
				art_b.P_w.data,			# 37 [N] tmp
				art_b.P_w_a.data,		# 38 [N] out: normalized prob. field P(w_k|b)
				art_b.N,				# 39
				art_b.N0,				# 40
				art_b.logS_MAX,			# 41
				art_b.initSigmaValues,	# 42
				# and now, the map field
				w.shape[1],		# 43
				w.data,			# matrix of [Na_max x Nb_max]
				P_b_a.data,		# matrix of [Na_max x Nb_max]
				sum_w_j.data,	# vector of [Na_max],
				self.n, art_b.n					
			)	
				
		
		self.duration_cl_fit_bar += benchmark.profile_cl('fit_bar',events,lws=lws)
		self.nsamples +=1
		art_b.nsamples+=1
		J=self.j_buf.get()	# copy from device
		if self.debug>=5: print 'bar fit result j_buf=',J
		j = J[0]
		k = J[2]
		if J[1]:	self.N+=1
		if regression:
			if J[3]: art_b.N+=1
		else:
			art_b.N = J[3]
#		print repr(j)
		assert k>=0 and j>=0, ("negative values k=%d j=%d signal an error" % (k,j))
		if self.debug>1:print '  art_b: accepted output category k=%d' % (k)
		if self.debug>1:print '  art_a: accepted input category j=%d' % (j)
		if self.debug>1:print '  updated map-field w[%d,%d] to %f' % (j,k,self.extern(w)[j,k])
		 
		
		return j,k,w,P_b_a,sum_w_j

	def bar_grow_mapfield(self, w,P_b_a,sum_w_j, N0a, N0b):
		"""resize mapfield to accomodate N0a and N0b """
		w			= self.ensure_matrix_size(w, N0a, N0b)
		P_b_a		= self.ensure_matrix_size(P_b_a, N0a, N0b)
		sum_w_j		= self.ensure_numrows(sum_w_j, (N0a,),fillZeros=True)
		return w,P_b_a,sum_w_j

	def bar_update_mapfield(self,w,P_b_a,sum_w_j, j, k, Na,Nb):
		"""Increments w[j,k] and update sums"""
		lws=(min(w.shape[1],1024), )
		gws=lws
		self.bar_update_mapfield_kernel(
			self.queue,gws,lws,
			j,k,Nb,
			w.shape[1],
			w.data,P_b_a.data,sum_w_j.data)
		return w,P_b_a,sum_w_j
	
	def bar_predict1_P(self,a, art_b, P_b_a):
		"""Computes eq. 15 for one input pattern a, in OpenCL"""
		j		= self.predict1(a)
		Na		= self.N
		Nb		= art_b.N
		
		# compute eq. (15)    
		P_w_a = self.P_w_a[:Na]
		#print 'Na=',Na,'Nb=',Nb,' P_w_a.shape=',P_w_a.shape,' P_b_a.shape=',P_b_a.shape
		P = cl.array.dot(P_w_a, P_b_a[:Na,:Nb]) # (1,Na) x (Na,Nb) vector x matrix
		print 'bar_predict1_P P=',P
		s = cl.array.sum(P)
		P /= s				# normalize
		return P  
	

	@property
	def profilesuffix(self):
		return "_"+self.name
	


def logsumexp(a,queue=None):
	"""OpenCL implementation of scikit.logsumexp returns a scalar """
	a_max = cl.array.max(a, queue=queue).get(queue=queue)
	out = np.log(cl.array.sum(cl.clmath.exp(a - a_max, queue=queue), queue=queue).get(queue=queue))
	out += a_max
	return out


