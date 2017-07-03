# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from numpy import dot,eye,diag,sum,amax,argmax,flatnonzero,zeros,empty,float32,float64,int32,int64,newaxis,ndarray,array,exp,sqrt,log,isfinite,isscalar,concatenate,hstack,vstack,array_repr,savetxt
import sklearn.base
import sklearn.metrics
import numpy.testing as testing
import Art
import config
import time
	
class Bar(sklearn.base.BaseEstimator):
	"""Bayesian ARTMAP. Contains two Art objects"""
	def __init__(self,**kwargs):
		"""	 BAR_create(n_a, n_b, S_MAX_A, S_MAX_B, P_min)
	Constructor creates a Bayesian Artmap (bart) structure
	   inputs
	        n_a: dimensionality of input vectors
	        n_b: dimensionality of output vectors
	    S_MAX_A: vigilance of ARTMAP_a
	    S_MAX_B: vigilance of ARTMAP_b
	      P_min: vigilance tracking (optional, default 0)
	   the bart structure contains the following fields:
	      art_a: the ART for inputs
	      art_b: the ART for outputs
	          w: a->b mapping field, of size Na x Nb
	 regression: False = predict() is a classifier, returns the class number rounded to nearest int
	 			 True =  predict() is regressor, returns an interpolated value
	      debug: set to 1 to print
	      
	     This method just copies the arguments. The actual initialization is done later in reinit(), in order to be re-initializable after set_args follow
		n_a, n_b, S_MAX_A=None, S_MAX_B=None, logS_MAX_A=None, P_min=0.0, dtype=float64, 
		plambda_a=0.01, plambda_b=0.01,
		allowExtendedVigilance=False,diag=None,N0=None,enable_growth=True,Knuth_variance=True,preselect_P_min=False
	      
	"""
		
		self.dtype 	= kwargs.get('dtype',float32)
		self.n_a 	= Bar.int_arg('n_a',None,**kwargs)
		self.n_b 	= Bar.int_arg('n_b',None,**kwargs)
		self.P_min 	= kwargs.get('P_min', 0.0)
		
		self.logS_MAX_A = Bar.logS_MAX_arg('S_MAX_A','logS_MAX_A',**kwargs)
		self.logS_MAX_B = Bar.logS_MAX_arg('S_MAX_B','logS_MAX_B',**kwargs)
		self.plambda_a 	= kwargs.get('plambda_a', 0.01)
		self.plambda_b 	= kwargs.get('plambda_b', 0.01)
		self.diag 				= kwargs.get('diag', None)
		self.N0 				= Bar.int_arg('N0', None,**kwargs)
		self.N0_b 				= Bar.int_arg('N0_b', None,**kwargs)
		self.allowExtendedVigilance = Bar.int_arg('allowExtendedVigilance', False,**kwargs)
		self.enable_growth 		= Bar.int_arg('enable_growth', True,**kwargs)
		self.Knuth_variance 	= Bar.int_arg('Knuth_variance', True,**kwargs)
		self.preselect_P_min 	= Bar.int_arg('preselect_P_min', False,**kwargs)
		self.regression 		= Bar.int_arg('regression', False,**kwargs)
		self.debug				= Bar.int_arg('debug',0,**kwargs)
		self.params_changed = True
		self.usecl				= Bar.int_arg('usecl',False,**kwargs)
		self.use_compact_kernel = Bar.int_arg('use_compact_kernel',False,**kwargs)
		self.art_a = None
		self.art_b = None
		self.b_min = None # min value of output category when classification
		self.fit_durations=() # list of durations of fit()

#		print '%d: Bar.__init__ %s' % (id(self),repr(kwargs))
		return
	
	@staticmethod
	def int_arg(name, defvalue, **kwargs):
		v=kwargs.get(name)
		if v is None: return defvalue
		return int(v)
	
	@staticmethod
	def logS_MAX_arg(namea,nameb,**kwargs):
		a=kwargs.get(namea,None)
		b=kwargs.get(nameb,None)
		if a is None and b is None:
			#raise Exception("Neither args specified: "+namea+" "+nameb)
			return log(0.1)
		elif (a is not None) and (b is not None):
			raise Exception("Both args specified: "+namea+" "+nameb)
		elif (a is not None):
			return log(float(a))
		else:
			return float(b)
        pass
	
	def reinit(self):
#		print 'self.usecl, self.use_compact_kernel=',self.usecl, self.use_compact_kernel
		self.art_a = self.new_art(self.n_a, logS_MAX=self.logS_MAX_A, 
								plambda=self.plambda_a, dtype=self.dtype,
								allowExtendedVigilance=self.allowExtendedVigilance,
								diag_ = self.diag,N0=self.N0,enable_growth=self.enable_growth,
								usecl=self.usecl,
								Knuth_variance=self.Knuth_variance,name='art_a')
		
		self.art_b = self.new_art(self.n_b, S_MAX=exp(self.logS_MAX_B), 
								plambda=self.plambda_b, dtype=self.dtype,
								allowExtendedVigilance=0, #self.allowExtendedVigilance,
								diag_ = self.diag,
								N0=self.N0_b,enable_growth=self.enable_growth,
								usecl=self.usecl,
								Knuth_variance=self.Knuth_variance,
								normalized_classifier_categories=not self.regression,
								name='art_b', 
								buildProgram=not self.use_compact_kernel)
		
		self.w       = zeros((self.art_a.N0, self.art_b.N0), dtype=self.dtype)
		self.sum_w_j = zeros((self.art_a.N0), dtype=self.dtype) # compute sums across every row
		self.P_b_a =   empty(self.w.shape, dtype=self.dtype)

		if self.use_cl_mapfield():
			self.w       = self.art_a.intern(self.w)
			self.sum_w_j = self.art_a.intern(self.sum_w_j) # compute sums across every row
			self.P_b_a   = self.art_a.intern(self.P_b_a)
		
		self.params_changed = False
		return
	
	def reset(self):
		self.art_a.reset()
		self.art_b.reset()
		return
	
	def set_debug(self,v):
		self.debug=v
		if self.art_a is not None:
			self.art_a.debug=v
		if self.art_b is not None:
			self.art_b.debug=v
		return
	
	def use_cl_mapfield(self):
		"""Returns True if the mapfield arrays (w, P_b_a and sum_w_j) are stored in OpenCL"""
		return self.art_a.usecl
	
	def __str__(self):
		s  = "BAR P_min=%g preselect=%d\n" % (self.P_min,self.preselect_P_min)
		s += str(self.art_a) +"\n"
		s += str(self.art_b) +"\n"
		if self.debug>=2:
			Na = self.art_a.N
			Nb = self.art_b.N
			s += 'P_b_a=\n'+str(self.extern(self.P_b_a)[:Na, :Nb])+"\n"
			s += 'w=\n'+str(self.extern(self.w)[:Na,:Nb])+"\n"
			s += 'sum_w_j='+str(self.extern(self.sum_w_j)[:Na])
		return s;


	def new_art(self,n, S_MAX=None, logS_MAX=None,plambda=0.1, dtype=None, diag_=None,
			allowExtendedVigilance=True,N0=None,enable_growth=True, usecl=None,
			Knuth_variance=True,
			normalized_classifier_categories=False,
			name='art', buildProgram=True):
		"""This function is used to instantiate for testing an Art (or Art_cl) object"""
		if diag_ is None: diag_ = config.g_options.diag if config.g_options is not None else True

		if dtype is None:
			if config.g_options.double_precision:
				dtype=float64
			else:
				dtype=float32
		if usecl is None:
			usecl = config.g_options is not None and config.g_options.usecl 
		if usecl:
			import Art_cl
			art=Art_cl.Art_cl(n,S_MAX=S_MAX, logS_MAX=logS_MAX, plambda=plambda, dtype=dtype, diag=diag_,allowExtendedVigilance=allowExtendedVigilance,
				Knuth_variance=Knuth_variance,
				normalized_classifier_categories=normalized_classifier_categories,
				N0=N0,enable_growth=enable_growth,ctx=config.g_ctx, queue=config.g_queue,name=name,debug=self.debug, buildProgram=buildProgram)
		else:
			art=Art.Art(n,S_MAX=S_MAX,logS_MAX=logS_MAX, plambda=plambda, dtype=dtype, diag=diag_,
				allowExtendedVigilance=allowExtendedVigilance,
				Knuth_variance=Knuth_variance,
				normalized_classifier_categories=normalized_classifier_categories,
				N0=N0,enable_growth=enable_growth,name=name,debug=self.debug)
		return art


	def predict1_P(self,a):
		"""Computes eq. 15 for one input pattern a"""
		art_a	= self.art_a
		if self.usecl and False:
			P = art_a.bar_predict1_P(a,self.art_b, self.P_b_a).get()
		else:
			j		= art_a.predict(a)
			Na		= art_a.N
			Nb		= self.art_b.N
			if self.debug > 1:	print art_a.name+' predict for input: ', a, 'Results: j=',j,'P_w=', self.art_a.P_w[:Na]
		
			# compute eq. (15)    
			P_w_a = art_a.extern(art_a.P_w_a[:Na])
			P_b_a = art_a.extern(self.P_b_a)[:Na,:Nb]	# first extern then subview
			#print 'Na=',Na,'Nb=',Nb,' P_w_a.shape=',P_w_a.shape,' P_b_a.shape=',P_b_a.shape
			P = dot(P_w_a, P_b_a) # (1,Na) x (Na,Nb) vector x matrix
			P = P / sum(P)				# normalize  
		return P
	
	def classify1(self,a):
		"""Classify a single pattern. Returns the centroid"""
		P = self.predict1_P(a)
		k = argmax(P)
		mu = self.art_b.extern(self.art_b.mu[k]).squeeze()
		return round(mu)
		
	
	def regress1(self,a):
		Nb=self.art_b.N
		P = self.predict1_P(a).transpose()
		MU_B = self.art_b.extern(self.art_b.mu[:Nb,])
		
		Bpred = dot(P, MU_B)		# eq. (18)
		
		if self.debug>=3:
			print 'regress1 mu_b=',MU_B,'P=',P,' Bpred=',Bpred, 'sum(P)=',sum(P)
		return Bpred
	
	
	def predict1(self,a):
		"""based on the self.regression boolean variable, either classifies sample a or else"""			
		return self.regress1(a) if self.regression else self.classify1(a)
	
	def predict_proba(self,X):
		"""Probability estimates.
		returns P : array-like, shape = [n_samples, n_classes]
		"""
		if len(X.shape)==1 and X.shape[0]==self.art_a.n:
			X=X.reshape((1,self.art_a.n))
		nsamples=X.shape[0]
		X = self.intern(X)
		P = zeros((nsamples,self.art_b.N))
		for  i in xrange(nsamples):
			P[i] = self.predict1_P(X[i])
		return P


	def predict(self, X):
		if len(X.shape)==1 and X.shape[0]==self.art_a.n:
			X=X.reshape((1,self.art_a.n))
		nsamples=X.shape[0]
		X = self.intern(X)
		C = empty((nsamples))
		for i in xrange(nsamples):
			C[i] = self.predict1(X[i])
		return C
					
	def pad_w(self, w,Na,Nb):
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
		v=zeros((max(Na,N), max(Nb,M)), dtype=self.dtype)
		v[:N, :M] = w
		return v


	def match_track(self, j, k):
		art_a=self.art_a
		art_b=self.art_b
		P = self.P_b_a
		P_min = self.P_min
		eps=1
		if j<0 or P_min<=0.0: return j
		if art_b.nj[k]==1: return j # k was a newly added B category
		logS = art_a.logS if art_a.allowExtendedVigilance else art_a.logS_tmp
		
		l=0
		while j>=0:			
			if P[j,k] >= P_min: return j;
			logS_MAX = logS[j] - eps
			j = art_a.get_winner(logS=logS, logS_MAX=logS_MAX)
			l+=1
			if l > art_a.N+1:
				print self
				raise Exception("match_track encountered a halting problem.. still on after %d iterations" % l)
			
		return j  

	def update_new(self,a,b):
		""" 
		function bart = BAR_update(bart, a, b)
		BAR_update the algorithm described in Section 2 of the paper
		trains the BAR map using input pattern a and output pattern b
		Modified algorithm since 2014-03-09
		
		"""
		art_a=self.art_a
		art_b=self.art_b

		k  = art_b.choose(b)
		if k<0:
			k=art_b.new_category(b)
			self.grow_mapfield()
			if self.debug>1:print '  append new column to w',self.w.shape
		else:
			if self.debug>1:print '  art_b: accepted output category k=%d' % (k)
			art_b.commit(b,k)
		
		selection = None
		if self.preselect_P_min and self.P_min>0.0:
			selection = flatnonzero(self.P_b_a[0:art_a.N,k]>=self.P_min)
		j = art_a.choose(a,selection=selection)
		j = self.match_track(j,k)
		if j<0: 
			j = self.art_a.new_category(a)
			self.grow_mapfield()
			if self.debug>1:print '  append new row to w',self.w.shape
		else:
			if self.debug>1:print '  art_a: accepted input category j=%d' % (j)
			art_a.commit(a,j)
			
		self.update_mapfield(j,k)
		return

	def grow_mapfield(self):
		"""resize mapfield matrix to accomodate new Na and Nb"""
		Na = self.art_a.N
		Nb = self.art_b.N
		if self.use_cl_mapfield():
			self.w, self.P_b_a, self.sum_w_j = self.art_a.bar_grow_mapfield(self.w, self.P_b_a, self.sum_w_j, Na, Nb)
		else:
			self.w = self.pad_w(self.w, Na,Nb) # append new column to w
			self.P_b_a = self.pad_w(self.P_b_a, Na,Nb) # append new column to P(b|a)
			
		return

	def set_unit_mapfield(self):
		"""For testing only. Sets the w matrix as if each input-output category was activated once"""
		Na = self.art_a.N
		Nb = self.art_b.N
		assert Na==Nb, 'set_unit_mapfield requires the same number of categories'
		for i in range(Na):
			self.w[i,i]=1
		w  = self.w[:Na,:Nb]
		self.sum_w_j = np.sum(w, axis=1) # compute sums row by row
		self.P_b_a = w / (self.sum_w_j[:,np.newaxis])		   		# eq. (2)
		return

	def update_mapfield(self,j,k):
		assert j>=0
		assert k>=0

		Na = self.art_a.N
		Nb = self.art_b.N
		if self.use_cl_mapfield():
			self.w, self.P_b_a, self.sum_w_j = self.art_a.bar_grow_mapfield(self.w, self.P_b_a, self.sum_w_j, Na, Nb)
			self.w, self.P_b_a, self.sum_w_j = self.art_a.bar_update_mapfield(self.w, self.P_b_a, self.sum_w_j, j,k,Na,Nb )
		else:
			self.w[j,k] = self.w[j,k] + 1		
			# update joint probab and conditional probab
			w  = self.w[:Na,:Nb]
			self.sum_w_j = np.sum(w, axis=1) # compute sums row by row
			self.P_b_a = w / (self.sum_w_j[:,np.newaxis])		   		# eq. (2)
		if self.debug>1: print '  updated map-field w[%d,%d] to %f' % (j,k,self.extern(self.w)[j,k])
		return
	
	def update_cl(self,a,b):
		art_a=self.art_a
		art_b=self.art_b
		j, k, self.w, self.P_b_a, self.sum_w_j = art_a.bar_fit1(a,b,art_b,self.w, 
               self.P_b_a, self.sum_w_j, regression = self.regression)
		return			
	
	def update(self,a,b):
		if self.art_a.usecl and self.use_compact_kernel:
			self.update_cl(a,b)
		else:
			self.update_new(a,b)
		return


	def get_joint_p(self):
		"""returns joint prob. field P(w^a, w^b)"""
		Na=self.art_a.N
		Nb=self.art_b.N
		w=self.extern(self.w)[:Na,:Nb]
		return w/sum(w, axis=None) # sum over both rows and columns
	
	def intern(self,x):
		if self.art_a is None: return x;
		return self.art_a.intern(x)
	
	def extern(self,x):
		import pyopencl as cl
		if x is None: return x;
		elif isinstance(x, cl.array.Array): return x.get();
		return x; 

		
		
	def fit(self,A,B):
		"""
		BAR_train trains the Bayesian Artmap using inputs from matrix A and 
		associated outputs in matrix B. The number of rows of A must match the number of rows B
		
		According to sklearn guidelines:
		
		Attributes that have been estimated from the data must always have a name ending with trailing underscore, 
		for example the coefficients of some regression estimator would be stored in a coef_ attribute after fit() has been called.
		(IL note: the underscore convention is not respected in BAR currently)
		
		
		The last-mentioned attributes are expected to be overridden when you call fit a second time without taking any previous value into account: 		
		fit should be idempotent.
		
		Fit resets the BAR to initial state prior fitting.
		To continue fitting on new data, use partial_fit 
		
		
		"""
		if self.n_a is None:
			self.n_a=A.shape[1] if len(A.shape)>1 else 1
			self.params_changed=True
		if self.n_b is None:
			self.n_b=B.shape[1] if len(B.shape)>1 else 1
			self.params_changed=True

		# always reinit
		self.reinit()
		self.partial_fit(A,B)
		self.fit_durations += (self.fit_duration,)
		return self    # returning self by following sklearn conventions
	
	def partial_fit(self,A,B,classes=None,sample_weight=None):
		"""See 
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.partial_fit
"""
		if self.params_changed: self.reinit()
		art_a=self.art_a
		art_b=self.art_b
		ndata=A.shape[0]
		
		if len(B.shape)==1: 
			n_b = B.shape[0]
			B = B.reshape((n_b,1))	# make sure B is two dimensional 

		assert A.shape[0] == B.shape[0]
		assert A.shape[1] == art_a.n
		assert B.shape[1] == art_b.n

		if self.debug>=1:
			print "%d %d: start fit A.shape=%s B.shape=%s %s %s" % (os.getpid(), id(self), A.shape, B.shape, str(self.art_a), str(self.art_b))
		if self.debug>=4:
			print 'A=',repr(A)
			print 'B=',repr(B)
			print 'dtype=',repr(art_b.dtype)

		# write input arrays to file for diagnostic
		if config.g_options.dump_sets:
			C=hstack((A,B))
			savetxt(config.g_options.outdir+"/fit-input-%04d.csv" % (config.g_options.fit_ctr,), C, delimiter=",\t", fmt='%g')
			config.g_options.fit_ctr+=1
			
		if self.b_min is None:
			self.b_min = 0
		art_b.b_min = self.b_min
		A = self.intern(A.astype(art_a.dtype))
		B = art_b.intern(B.astype(art_b.dtype))
			
		i0 = self.art_a.nsamples
		
		t0 = time.time()
		for i in xrange(ndata):
			ii=i+i0
			a=A[i]
			b=B[i]
			if self.debug>=2:
				print '>>>>>>>>> train pattern:',ii,'a=',a,'b=',b
			
			self.update(a, b)
			if self.debug>=2:
				print '<<<<<<<<< after train pattern:',ii, 'a=',a,'b=',b
				print self
				
				assert(self.art_a.nsamples==(ii+1))
				assert(self.art_b.nsamples==(ii+1))
				assert(sum(self.art_a.nj[:self.art_a.N])==(ii+1))
				
		self.fit_duration = time.time()-t0
		if self.debug>=2:
			print '<<<<<<<<< END OF TRAINING '
		if self.debug>=1:
			print "%d %d: end fit %s" % (os.getpid(), id(self), str(self))
			testing.assert_equal(sum(art_b.extern(art_b.nj[:art_b.N])),art_b.nsamples)

		return self
	
	@staticmethod
	def classify(sample, t, group, bart):
		"""function [C,bart] = BAR_classify(sample, t, group, bart)
		
		 (using similar prototype as Matlab's classify functions)
		
		   C,bart = Bar.classify(SAMPLE,TRAINING,GROUP) classifies each row of the
		   data in SAMPLE into one of the groups in TRAINING. 
		
		   SAMPLE and TRAINING must be matrices with the same
		   number of columns. GROUP is a grouping variable for TRAINING. Its
		   unique values define groups, and each element defines the group to
		   which the corresponding row of TRAINING belongs. GROUP can be a
		   numeric vector. 
		
		   TRAINING and GROUP must have the same number of rows
		
		   The optional input structure bart is used to an already initialized
		   B-ART. If not present, BAR_classify creates one internally.
		
		
		   The resulting CLASS indicates which group each row of SAMPLE has
		   been assigned to, and is of the same type as GROUP.
		   The function also returns the trained bart structure that can be
		   further reused
		"""
		if bart is None:
			smax_a = 0.01;
			smax_b = 0.01;
			P_min = 0;
			bart = Bar(t.shape[1], group.shape[1], S_MAX_A=smax_a, S_MAX_B=smax_b, P_min=P_min)
			
		bart.fit(t,group);
		C=bart.predict(sample)
		return C,bart
	


	def get_params(self, deep=True):
		"""Return all relevant parameters that are required to clone the object by sklearn"""
		p = {
				'logS_MAX_A': float(self.logS_MAX_A),
				'logS_MAX_B': float(self.logS_MAX_B),
				'plambda_a':  float(self.plambda_a),
				'plambda_b':  float(self.plambda_b),
				'P_min'    :  float(self.P_min),
				'n_a':		  self.n_a, 
				'n_b':		  self.n_b,
				'allowExtendedVigilance': int(self.allowExtendedVigilance),
				'Knuth_variance': int(self.Knuth_variance),
				'preselect_P_min': int(self.preselect_P_min),
				'diag': self.diag,
				'dtype': self.dtype,
				'regression':self.regression,
			}

		if self.usecl:
			p['usecl'] = True
			if self.use_compact_kernel: p['use_compact_kernel'] = int(self.use_compact_kernel)

		if self.debug:
			p['debug'] = self.debug

		if self.N0:   p['N0'] = int(self.use_compact_kernel)
		if self.N0_b: p['N0_b'] = int(self.use_compact_kernel)
		return p

	def get_params_str(self):
		"""Returns a single string containing parameters suitable for file names"""
		return ("Pmin_%g_preselect_%d_" % (self.P_min,self.preselect_P_min)) + self.art_a.get_params_str()

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		if self.debug>=1:
			print '%d %d: Bar.set_params %s ' % (os.getpid(), id(self),repr(parameters))
		self.params_changed = True
		return self # returns self by sklearn conventions


	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Training set.

		y : array-like, shape = [n_samples]
			Labels for X.

		Returns
		-------
		z : float

		"""
		
		y_pred=self.predict(X)
		if self.regression:
			s = sklearn.metrics.r2_score(y, y_pred)
		else:
			s = sklearn.metrics.accuracy_score(y, y_pred)
		return s

