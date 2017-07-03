# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy
from numpy import dot,eye,diag,sum,amax,argmax,sort,argsort,flatnonzero,zeros,empty,float32,float64,int32,int64,newaxis,ndarray,array,exp,sqrt,log,isfinite,isscalar,concatenate,hstack,vstack,array_repr,outer,copyto
import numpy.linalg as linalg
import scipy
import scipy.stats
import scipy.misc
import sys
import os
import textwrap
import sklearn
import sklearn.base

class Art(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
	"""The Art class represents a sets of Gaussian means and covariance matrices.
		
		it is derived from scikit, see
		http://scikit-learn.org/stable/developers/#rolling-your-own-estimator
		
		Keyword fields:
		
		mu		-- means, for each category
		sigma	-- covariance matrices, for each category

		normalized_classifier_categories
			when set to True, it assumes the inputs to be the integers 0,1,..,n-1
			
		
		debug 
			1 - print training step pattern
			2 - print ART after each training
			3 - print each new_category and learn()
			4 - print internals of logpdf and others
		"""
	def __init__(self, n, S_MAX=None, logS_MAX=None,plambda=0.9, dtype=float32, 
			diag=True,allowExtendedVigilance=True,
			N0=None,enable_growth=True,
			Knuth_variance=True,normalized_classifier_categories=False,b_min=0,
			name='art',usecl=False,debug=0):
		"""Constructor parameters
		n 		= data dimensionality (number of input features)
		S_MAX 	= maximum hypervolume of a cluster
		plambda = initial scaling factor
		dtype	= data type
		diag	= use diagonal covariance matrices?
		allowExtendedVigilance=1 1: behave like in published article
		                         0: do not extend a cluster if the result would exceed S_MAX; create a new one instead.
		                         
		"""
		self.usecl	= usecl
		self.name	= name
		self.dtype	= dtype
		self.dtype_size = numpy.dtype(dtype).itemsize
		if (S_MAX is not None) and (logS_MAX is not None):
			raise Exception(name+": Cannot specify both S_MAX and logS_MAX")
		elif logS_MAX is not None:
			self.logS_MAX = float(logS_MAX)
		elif S_MAX is not None:
			self.logS_MAX = math.log(S_MAX)
		else:
			raise Exception(name+": Neither S_MAX nor logS_MAX specified")
		
		self.plambda = float(plambda)
		if n==1: diag=True
		if N0 is None:
			N0=1
		self.n       = n
		self.diag    = diag			# 1: use diagonal matrix, 0 = use full cov. matrix		
		self.mu      = zeros((N0,n),dtype=self.dtype)
		self.N0		 = N0
		self.enable_growth=enable_growth
		if self.diag or n==1:
			sdim = (N0,n)
		else:
			sdim = (N0,n,n)
		self.sdim    = sdim
		self.sigma   = zeros(sdim, dtype=self.dtype) # cov matrix summed (must be divided by nj-1)
		self.sigma_j_init = None # initialized later
		self.nj      = zeros(N0,self.dtype)
		self.nsamples= 0
		self.N       = 0
		# fields computed from others:
		self.sigi    = zeros(sdim, dtype=self.dtype) # sqrt(1/sigma)
		self.logS    = zeros(N0,dtype=self.dtype) # log(S)
		# temp storage
		# allocate for a single category		
		self.mu_tmp		= empty(self.mu.shape, dtype=self.dtype)
		self.sigma_tmp  = empty(self.sigma.shape,dtype=self.dtype)
		self.logS_tmp   = empty(self.logS.shape,dtype=self.dtype)
		

		self.P_w     = zeros(N0,dtype=self.dtype)  	# based on eq. (4)
		self.logP_w  = empty(0,dtype=self.dtype)  	# based on eq. (4)
		self.P_w_a   = empty(0,dtype=self.dtype)    # log eq. (3)
		self.logp	 = empty(0,dtype=self.dtype)  	# based on eq. (4)
		self.log0    =  float.fromhex('-0x1.0p+127') # approximates log(0), but avoid -inf, still storable on float32
		self.FLT_MIN =  float.fromhex('0x1.0p-149')
		self.plus_inf = float.fromhex('0x1.0p+127') # approximates +inf (but avoid), for comparisons only, storable on float32
		# other flags
		self.allowExtendedVigilance = allowExtendedVigilance # 1: original behavior doc in article, 0 = never allow S_J>S_MAX
		self.Knuth_variance=Knuth_variance
		self.debug   = debug			# 1: print each training pattern 2: print mu and sigma 3: print probabs
		self.duration_cl_fit_bar=0

		self.normalized_classifier_categories = normalized_classifier_categories
		self.b_min=b_min
		if normalized_classifier_categories:
			assert n==1, "n must be 1 when using normalized_classifier_categories"
			assert N0>1, "N0 must be greater than 1 when using normalized_classifier_categories"
			copyto(self.mu, numpy.arange(N0,dtype=self.dtype).reshape((N0,1)) )
			# we know in advance the number of classes. 
			self.N = N0
			# also set unitary covariance
			copyto(self.sigma, numpy.ones(sdim,dtype=self.dtype))
			copyto(self.sigi,  numpy.ones(sdim,dtype=self.dtype))
			
				
		return

	def reset(self):	
		self.N=0
		self.duration_cl_fit_bar=0
		self.nsamples= 0
		return
	
	def __str__(self):
		wrapper = textwrap.TextWrapper(initial_indent="  ",subsequent_indent="  ",width=120)
		l=[]
		N=self.N
		l.append("%s: logS_MAX=%g lambda=%g N0=%d N=%d n=%d diag=%d allow=%d Knuth_variance=%d nsamples=%d b_min=%d" % (self.name, self.logS_MAX,self.plambda, self.N0, self.N, self.n,self.diag,self.allowExtendedVigilance,self.Knuth_variance,self.nsamples,self.b_min));
		pt=False # show array types? True/False
		if self.debug >= 2:
			l.extend(wrapper.wrap('mu    ' + (str(self.mu.dtype) if pt else '') + str(self.mu.shape) + "    =\n " + str(self.mu[:self.N,]))) 
			l.extend(wrapper.wrap('sigma=' + repr(self.get_SIGMA())))
			l.extend(wrapper.wrap('sigi  ' + repr(self.sigi[:self.N,])))
			l.extend(wrapper.wrap('log(S)' + (str(self.logS.dtype)if pt else '')+str(self.logS.shape)+"     = " + str(self.logS[:N])))			
			l.extend(wrapper.wrap('nj    ' + (str(self.nj.dtype)if pt else '')+str(self.nj.shape)+"     = " + str(self.nj[:N])))
			l.extend(wrapper.wrap('P_w   ' + (str(self.P_w.dtype)if pt else '')+str(self.P_w.shape)+" = "+ str(self.P_w[:N])))
			l.extend(wrapper.wrap('logP_w' + str(self.logP_w[:N])))
		return "\n".join(l);

	
	def set_row(self,X,x,j):
		"""Assuming X a matrix and x a vector, sets the j-th row of the matrix.
		If X is a vector and x is a scalar, then sets the j-th element
		In a special case it can append a last element to the array (extend the 0-th axis)"""
		if not hasattr(x,'shape'):
			x=array(x,dtype=self.dtype)
		
		#if self.debug>=3:print self.name,'set_row j=',j,'X=',repr(X),'x=',repr(x), 'exp=',array(expand_dims(x,axis=0),dtype=self.dtype)
		if j<X.shape[0]:
			X[j,]=x
		else:
			x0_shape = (1,) + X.shape[1:]
			x=x.reshape(x0_shape).astype(self.dtype)
			while j > (X.shape[0]-1): # extend one row
				if X.size == 0:
					X = x
				else:
					X = concatenate((X, x),axis=0) # avoid vstack in the X=1-dim vector case
		X=self.intern(X)
		#if self.debug>=3: print self.name,'set_row result shape=',X.shape,X
		return X

	def intern(self,x,newshape=None):
		"""Convert x to the internal type"""
#		print id(self),': intern x=',type(x)
		if not isinstance(x,ndarray):
			x=array(x,dtype=self.dtype)
			
		if newshape is not None and x.shape!=newshape and x.shape!=():
			x=x.reshape(newshape)
			
		if x.shape==():
			x=array(x,dtype=self.dtype)
		else:
			x=x.astype(self.dtype)
		return x
	
	def extern(self, x):
		"""convert x from internal representation to numpy array"""
		if isinstance(x, ndarray):
			return x
		return array(x)
	
	def new_category(self,a):
		"""The method new_category feed the first pattern a to an empty ART, which will create the first category
			
			Keyword arguments:
				a -- array
			Result:
				j -- index of new category
				S_j -- log volume of j category """
		if self.N >= self.N0 and not self.enable_growth:
			raise TrainingOverflowException(self.N0)

		assert not self.normalized_classifier_categories
				
		a = self.intern(a, newshape=(1, self.n)) # make sure a is row vector
		j = self.N;
		if self.debug >= 3: print '     %s: new category: %d with input:%s' % (self.name,j, str(a))		
		self.N = self.N + 1
		self.nsamples = self.nsamples + 1		

		self.mu = self.set_row(self.mu,   a, j)
		self.nj = self.set_row(self.nj, 1.0, j)
		self.P_w = self.set_row(self.P_w, 0.0, j) # make room
		self.logP_w = self.set_row(self.logP_w, 0.0, j) # make room  

		if self.sigma_j_init is None:
			f =  self.plambda * exp(self.logS_MAX/float(self.n))
			
			if self.diag:
				sigma_j_init = empty(self.n,dtype=self.dtype)
				sigma_j_init.fill(f)
			else:
				sigma_j_init = eye(self.n) * f
			self.sigma_j_init = self.intern(sigma_j_init)
		sig_j = self.sigma_j_init
		self.sigma = self.set_row(self.sigma, sig_j, j)
		if self.diag or self.n==1:				
			sigi_j   = sqrt(1.0 / sig_j)        # (N,n) matrix, inverse of each element
			logSig_j = log(sig_j)
			logS_j   = sum(logSig_j[isfinite(logSig_j)])
		else:
			chol_j = numpy.linalg.cholesky(sig_j)
			sigi_j = linalg.pinv(chol_j)        # equivalent of (1/sqrt(sig))
			sg,logS_j = linalg.slogdet(sig_j)	# log of determinant

		self.sigi  = self.set_row(self.sigi, sigi_j, j)
		self.logS  = self.set_row(self.logS,  logS_j, j)
		self.update_P_w()
		return j

	def update_P_w(self):
		"""Called internally when nj changes"""
		if self.nsamples>0:
			N=self.N
			self.P_w  = self.nj.copy()
			self.P_w /= float(self.nsamples) # always use inplace operations to avoid side effect in PyOpenCL

			P_w=self.extern(self.P_w[:N])
			eps=self.FLT_MIN
			#P_w[P_w < eps]=eps # make sure we don't compute log(0)
			P_w+=eps # make sure we don't compute log(0)
			self.logP_w = self.intern(numpy.log(P_w))
		return
	


	def logpdf(self,X,selection=None):
		""" compute multivariate normal distribution between an input vector x (row)
			and a set of parameters mu and sigma. Sigma is considered a diagonal matrix 
		when selection is specified, then logpdf is computed only on the selected categories
		selection must be an integer index array
		
			"""
		n   = self.n
		N   = self.N
	# L = number of inputs
#		print 'logpdf x=',X		
		assert not self.normalized_classifier_categories
		L = 1 if len(X.shape)<2 else X.shape[0]
		mu  = self.mu[:N,]
		if isscalar(X):
			X=array(X)
		if len(X.shape)<2: X=X.reshape((1,X.shape[0])).astype(self.dtype)
		assert L == X.shape[0]
		assert n == X.shape[1]
		sigi=self.sigi[:N,]				# 1/sigma for each element
		logS=self.logS[:N]
		result = empty((L,N), dtype=self.dtype)		
		if selection is None:
			selection = xrange(N)
		elif len(selection)<N:
			result.fill(self.log0)
			
		Nsel=len(selection)
		allselected = (Nsel == N)
		half_log_twopi = math.log(2.0*math.pi)/2.0
		if self.debug>=4:
			print self.name," logpdf allselected=",allselected,'Nsel=',Nsel,'N=',N,'n=',n,'diag=',self.diag
			print 'X=', repr(X)
			print 'mu=',repr(mu)
			print 'sigi=         ',repr(sigi)
			print '1/sqrt(sigma)=',repr(sqrt(1.0/self.get_SIGMA()))
			print 'sigma=',self.get_SIGMA()			
			print 'logS=',repr(logS)
			
		for i in xrange(L): # for each input vector
			x=X[i,:]
			assert x.size == n
			if self.diag or n==1:
				if allselected:
					# below the vectorized form is 5 times faster than the for loop for n=1024 N=1024
					result[i,:] = -n*half_log_twopi - 0.5 * logS - 0.5 * sum(((x-mu)*sigi)**2, axis=1)
				else:
					for j in selection:
						d = (x - self.mu[j])*sigi[j]
						result[i,j] = -n*half_log_twopi - 0.5 * logS[j] - 0.5 * sum(d**2)
	
			else: # generic non-diagonal covariance case. sigi = pinv(chol(sigma))
				for j in selection:
					d = dot(x - mu[j],sigi[j])
					result[i,j] = -n*half_log_twopi - 0.5 * logS[j] - 0.5 * dot(d,d.T)
			if self.debug>=4: print '    ',self.name,'logpdf returns: ',result
			# --- self verification ----------------------------------
			if self.debug>=4: self.verify_logpdf(x,result,allselected=allselected,selection=selection)		
		# --------------------------------------------------------
		assert result.shape[0]==L
		assert result.shape[1]==N
		if L==1: result=result[0]
		self.logp = result
		return result
	
	def verify_logpdf(self,x,result,allselected=True,selection=None):
		N=self.N
		n=self.n
		jr = xrange(N) if allselected else selection
		allowed_rerr = 0.5E-5 if self.dtype == float32 else 0.5E-14 
		for j in jr:
			r=result[j]
			if n==1:
				desired = scipy.stats.norm.logpdf(x, loc=self.mu[j], scale=sqrt(self.get_SIGMA()[j]))[0]
			else:
				import scipy.stats._multivariate as multivariate
				mng = multivariate.multivariate_normal_gen()
				desired = mng.logpdf(x, mean=self.get_MU()[j], cov=self.get_SIGMA()[j])
				
			rerr=abs(r - desired)
			if abs(desired)>1E-27: rerr = rerr / abs(desired)
			if  rerr > allowed_rerr:
				print 'LOGPDF MISMATCH got %e expected %e err=%e' % (r, desired, rerr)
		return
	
	def normalize_logsum(self, log_p1, log_p2):
		"""assuming log_p[i] contains log(p1) + log(p2), computes the quantiy		
			p[i]/sum(p)
			input:  log_p1, log_p2 vectors of length N
			result: normalized probability vector 
			"""
#		print 'normalize_logsum: log_p1',log_p1.shape, 'log_p2',log_p2.shape
		assert(len(log_p1.shape)==1)
		assert(len(log_p2.shape)==1)
		lp1 = self.extern(log_p1)
		lp2 = self.extern(log_p2)
		lp = lp1+lp2                 # p1 * p2
		lpfilt = lp[lp > self.log0]
		s = scipy.misc.logsumexp(lpfilt) if lpfilt.size > 0 else self.log0 # log sum e^lp = log prod(p)
		res = exp(lp - s)            # p / sum(p)
		if self.debug >= 4:
		  print self.name,' normalize_logsum'
		  print '		log(p)=',lp
		  print '		lp1   =',lp1
		  print '		lp2   =',lp2
		  print '		p=     ',exp(lp)
		  print '		ref=   ',exp(lp)/sum(exp(lp))
		  print '		res=   ',res
		return self.intern(res) 
	
	
	def compute_posterior_probab(self,a,selection=None):
		"""compute eq. (3)"""
		n = self.n
		N = self.N
		assert a.size == self.n
		if N<1: return
		logp = self.logpdf(a,selection=selection)
		self.P_w_a = self.normalize_logsum(self.logP_w[:N], logp[:N])
		return 
	
	
	def predict(self,A,logS_MAX=None,selection=None):	
		"""
		function j = predict(art,a)
		predict compute probabilities
		based on equations: (4), (5) and (3) based on an input pattern a
		a is assumed an 1 x n vector
		P_w_a eq. (3) 
		p_a_w eq. (5)
		P_w   eq. (4) assumed already computed in art
		
		param logS_MAX is the vigilance criteria.
		returns j = cluster indexes, ore -1 if no match in selection
		"""
		N=self.N

		if self.normalized_classifier_categories:
			return A

		if A.size==self.n:A=A.reshape(1,self.n)
		elif A.ndim==1: A=A[newaxis,:]


		if self.debug >= 3:
			print '    ',self.name, ' predict input: ',A
			if selection is not None:
				print '    ',self.name, ' selection: ',selection
		nsamples = A.shape[0]
		assert(A.shape[1] == self.n)
		
		jj=empty(nsamples,dtype=int32)

		for i in xrange (nsamples):
			a=A[i]
			if N < 1 or (selection is not None and len(selection)<1):
				j = -1
			else:
				self.compute_posterior_probab(a,selection=selection)
				j = self.get_winner()
				if self.debug >= 3:
					print '      P_w=',self.P_w[:N]
					print '      P_w_a=',self.P_w_a[:N]
					print '      winner j=',j
			jj[i]		=	j
		if nsamples==1:
			return jj[0]
		else:
			return jj
			
	def predict_mu(self,A):
		"""similar to predict, but returns mu, the center of the winner category, or None if no match"""
		j = self.predict(A)
		if j<0: return None
		return self.extern(self.mu[j,:])
	  

	def get_winner(self, logS=None, logS_MAX=None):
		"""
			Computes winner category j.
			If logS_MAX is specified then only categories satisfying S[j]<=S_MAX are considered.
			If none found, returns j=-1
		"""
		N=self.N
		if N<1: return -1
		assert(len(self.P_w_a.shape) == 1)
		P_w_a = self.extern(self.P_w_a[:N]).copy() # make a copy because some entries will be reset
		if (logS_MAX is not None) and (logS_MAX < self.plus_inf):
			logS_MAX = self.extern(logS_MAX)
			if logS is None: logS=self.logS
			logS  = self.extern(logS)[:N]
			P_w_a[logS > logS_MAX] = self.log0 # reset probabs where hypervolume > S_MAX
			
		j = argmax(P_w_a)
		
		# in degenerate cases when all p_a_w fields are zero then argmax returns 0
		# which would falsely yield to the conclusion that category j=0 is the winner
		# when in fact there is no winner, thus a new category needs to be created
#		print 'P_w_a=',P_w_a
		assert(j<N)
		if self.logp[j] <= self.log0 or P_w_a[j] <= self.log0:
			j = -1

		return j
	
	def update_mu_sigma_j(self,a,mu_old, sold, nj):
		a = self.extern(a)
		nj		= float(nj)
		d_old   = a-mu_old			# difference between current input and prev. mean
		mu_j   	= mu_old + d_old/(nj+1.0)
		d_new	= a-mu_j			# difference between current input and updated mean
		njdiv 	= nj if self.Knuth_variance else nj+1.0 # nj used in division of variance
		njdiv = float(njdiv)
		if self.diag:			
			if self.Knuth_variance:				
				snew_n = sold + (d_new*d_old)
			else:
				snew_n = sold + (d_new*d_new)			
			sig_j = snew_n / njdiv
			log_sig_j = log(sig_j)
			logS_j = sum(log_sig_j[isfinite(log_sig_j)])
		else:
			snew_n   = sold + outer(d_new,d_new)
			sig_j = snew_n / njdiv
			sg,logS_j = linalg.slogdet(sig_j)
			assert sg>=0, 'the determinant of the covariance matrix is negative!!! '+repr(sig_j)

		return mu_j, snew_n, sig_j, logS_j


	def update_mu_sigma_tmp(self,a,selection=None):
		"""Computes mu, sigma and logS (temporary values)"""
		N = self.N
		if selection is None:
			selection = xrange(N)		
		self.logS_tmp.fill(self.plus_inf)
		for j in selection:
			nj 		= self.extern(self.nj[j])
			mu_old 	= self.extern(self.mu[j])
			sold    = self.extern(self.sigma[j])
	# --------------------------------
			mu_j, snew_n, sig_j, logS_j = self.update_mu_sigma_j(a, mu_old, sold, nj)
			self.mu_tmp    = self.set_row(self.mu_tmp, mu_j, j)
			self.sigma_tmp = self.set_row(self.sigma_tmp, snew_n, j)
			self.logS_tmp  = self.set_row(self.logS_tmp,  logS_j, j)
			
		if self.debug>=4: 
				print '   ',self.name,'update_mu_sigma_tmp Nsel=',len(selection),'a=',a
				print '          nj           = ',self.nj[:N]
				print '          mu_tmp       = ',self.mu_tmp[:N,]				
				print '          sigma_tmp    = ',self.sigma_tmp[:N,]
				njdiv=self.extern(self.nj[:N]).reshape(self.get_SIGMA_shape()) 
				print '          s/n          = ',self.sigma_tmp[:N,]/njdiv
				print '          log(|S_tmp|) = ',self.logS_tmp[:N]
			
		
		return
		
	def choose_normalized_classifier_categories(self,a):
			assert a==int(a)
			j = int(a) - self.b_min
			assert j>=0
			assert j<self.N0, ('N0=%d a=%s j=%d b_min=%d' % (self.N0,repr(a),j,self.b_min))
			return j


	def choose(self,a,selection=None,logS_MAX=None):
		if self.normalized_classifier_categories:
			return self.choose_normalized_classifier_categories(a)

		if logS_MAX is None: logS_MAX=self.logS_MAX;
		if selection is not None and len(selection)<1: return -1
		self.compute_posterior_probab(a, selection=selection)
		self.update_mu_sigma_tmp(a, selection=selection)
		logS = self.logS if self.allowExtendedVigilance else self.logS_tmp
		return self.get_winner(logS, logS_MAX=logS_MAX)
	
	

	def commit(self,a,j):
		if j<0: return
		
		self.nsamples	=	self.nsamples + 1
		self.nj[j] = self.nj[j] + 1

		# when normalized classifier categories are used, the categories are determined in
		# init to 0,1,2,...
		# in this case we don't modify mu, sigma
		# but update nsamples, nj and P_w
		if not self.normalized_classifier_categories:
			self.mu			=	self.set_row(self.mu, self.mu_tmp[j], j)		
			njdiv = float(self.nj[j])
			if self.Knuth_variance: njdiv -= 1.0
			sig_j = self.sigma_tmp[j] / njdiv
			if self.diag or self.n==1:
				sigi_j   = sqrt(1.0 / sig_j)		# (N,n) matrix, inverse of each element
			else:
				chol_j = linalg.cholesky(sig_j)
				sigi_j = linalg.pinv(chol_j)		# equivalent of (1/sqrt(sig)) (?)
			self.sigma = self.set_row(self.sigma, self.sigma_tmp[j], j)
			self.sigi  = self.set_row(self.sigi,  sigi_j, j)
			self.logS  = self.set_row(self.logS,  self.logS_tmp[j], j)

		if self.debug >= 3: print '      ',self.name,'commited pattern j=',j,'nj=',self.nj[j], 'mu=',self.mu[j], 'sigma=',self.sigma
		self.update_P_w()
		return
		

	def fit1_normalized_classifier_categories(self,a):
		"""Special case of fit1 when using normalized category labels: a=0,1,..N-1"""
		assert a==int(a)
		j = int(a) - self.b_min
		assert j>=0
		assert j<self.N0

		self.nsamples	=	self.nsamples + 1
		self.nj[j] = self.nj[j] + 1
		if self.debug >= 3: print '      ',self.name,'commited pattern j=',j,'nj=',self.nj[j], 'mu=',self.mu[j], 'sigma=',sig_j
		self.update_P_w()

		return j

	def fit1(self,a):
		if self.normalized_classifier_categories:
			return self.fit1_normalized_classifier_categories(a)

		j = self.choose(a,logS_MAX=self.logS_MAX)
		if j<0:
			j = self.new_category(a)
		else:
			self.commit(a,j)
		return j
	

	def fit(self,A, B=None,i_first=0): 
		"""function [art,j]=fit(art,A)
		fits train ARTMAP over dataset taken from A
		 each row of A represent an input pattern"""
		if len(A.shape)==1 and A.shape[0]==self.n: # a single vector given as a single row
			A=A.reshape((1,self.n))
		elif len(A.shape)==1: A=A[newaxis,:] # n=1, convert to a column vector
		A=self.intern(A)
		ndata = A.shape[0]
		n = A.shape[1]
		assert(n == self.n)
		for i in xrange(i_first,ndata):
			a=A[i]
			if self.debug > 0: print '---- Feeding input pattern(',i,'): ',a
			j = self.fit1(a)
			if self.debug > 1:
				print '---- ART after pattern %d -----' % i
				print self
		return j

	def reorder(self, orderby='P_w', asc=True):
		"""Re-orders categories"""
		if self.N<2: return
		N=self.N
		if orderby == 'P_w':  ix = argsort(self.P_w[:N])
		elif orderby == 'mu': ix = argsort(self.mu[:N,0])
		elif orderby == 'S':  ix = argsort(self.logS[:N])
		
		if not asc: ix = ix[::-1]
#		print 'N=',self.N, 'P_w=',repr(self.P_w)
#		print 'argsort: ',type(ix),'len=',len(ix),ix
		self.P_w 	= self.P_w[ix]
		self.nj 	= self.nj[ix]
		self.logS   = self.logS[ix]
		self.mu 	= self.mu[ix,]
		self.sigma 	= self.sigma[ix,]
		self.sigi 	= self.sigi[ix,]
		
	def get_cov(self, j):
		"""Returns the covariance matrix of the j-th category.
		It always returns an n x n matrix, even if internally is stored in diagonal"""
		C = self.extern(self.sigma[j])
		if self.diag:
			C=diag(C)
		return C

	def get_MU(self):
		"""Returns externalized centroids"""
		return self.extern(self.mu[:self.N, ])
	
	def get_SIGMA_shape(self):
		N=self.N
		shp=(N,1) if (self.diag) else (N,1,1)
		return shp
	
	def get_SIGMA(self):
		"""Returns externalized covariance matrices"""
		N=self.N
		njdiv=self.extern(self.nj[:N]).reshape(self.get_SIGMA_shape()).copy()
		if self.Knuth_variance: njdiv[njdiv>1] -= 1.0
		return self.extern(self.sigma[:N, ])/njdiv
	
	def set_MU_SIGMA_P(self, mu=None,sigma=None,P=None):
		"""Sets the classifer state from external matrices.
		Recomputes sigi, logS"""
		N=mu.shape[0]
		assert mu.shape[0]==sigma.shape[0]
		assert mu.shape[0]==P.shape[0]
		assert self.n==mu.shape[1]
		assert P.ndim==1
		self.N=N
		self.mu 	= self.intern(mu)
		self.sigma	= self.intern(sigma)
		
		self.nj 	= self.intern(numpy.ones(N))
		self.P_w	= self.intern(P)
		self.logP_w = self.intern(numpy.log(P))

		self.nsamples	=	N
		for j in xrange(N):
			sig_j = sigma[j]
			if self.diag or self.n==1:				
				sigi_j   = sqrt(1.0 / sig_j)        # (N,n) matrix, inverse of each element
				logSig_j = log(sig_j)
				logS_j   = sum(logSig_j[isfinite(logSig_j)])
			else:
				chol_j = numpy.linalg.cholesky(sig_j)
				sigi_j = linalg.pinv(chol_j)        # equivalent of (1/sqrt(sig))
				sg,logS_j = linalg.slogdet(sig_j)	# log of determinant
			self.sigi  = self.set_row(self.sigi,  sigi_j, j)
			self.logS  = self.set_row(self.logS,  logS_j, j)

		
		
		return self

	def get_SIGMA_INV(self):
		"""Returns externalized inverse covariance matrices"""
		return self.extern(self.sigi[:self.N, ])

	def get_Prior(self):
		"""Returns prior probability of each category"""
		return self.extern(self.P_w[:self.N])
	
	def get_params(self, deep=True):
		return {"logS_MAX": self.logS_MAX, "plambda":self.plambda, 
			"Knuth_variance":self.Knuth_variance, "diag":self.diag,
			"allowExtendedVigilance":self.allowExtendedVigilance}

	def get_params_str(self):
		"""Returns a single string containing parameters suitable for file names"""
		return "diag_%d_allow_%d_Knuth_%d_lambda_%g_logSMAX_%g" % (self.diag,self.allowExtendedVigilance, self.Knuth_variance, self.plambda,self.logS_MAX)

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			self.setattr(parameter, value)
	"""
	def __del__(self):
		print 'Art del',self.name
	"""
	


class TrainingOverflowException(Exception):
	def __init__(self,N):
		self.N=N;
	def __str__(self):
		return repr(self.N)

