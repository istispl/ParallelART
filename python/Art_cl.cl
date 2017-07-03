
// host program should define the following macros:

// #define TYPE float
// number of threads in block
// #define REDUCE_blockSize 256

// constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
// #define USE_KNUTH_VARIANCE 1 // defined in app

#define TYPE __DATA_TYPE__

//		self.log0    =  float.fromhex('-0x1.0p+127') # approximates log(0), but avoid -inf, still storable on float32
//		self.plus_inf = float.fromhex('0x1.0p+127')  # approximates +inf (but avoid), for comparisons only, storable on float32
#define MINUS_INF -0x1.0p+127
#define PLUS_INF   0x1.0p+127

#ifdef _USE_BASE_TWO

	#define LOG(x) native_log2(x)
	#define EXP(x) native_exp2(x)
	#define HALF_LOG_TWOPI 1.325748064736159F
	#define LOG_E 1.44269504089F

#else
	#define LOG(x) log(x)
	#define EXP(x) exp(x)
	#define HALF_LOG_TWOPI 0.91893853320467267F
	#define LOG_E 1
#endif

__inline TYPE sqr(TYPE x){
	return x*x;
}


// defines a parallel for loop for i = 0 to n-1
// executed on global work items

#define PARALLEL_FOR_GLOBAL(i,i0,n) \
	for (int i = get_global_id(0)+(i0); i < (n); i += get_global_size(0))


#define PARALLEL_FOR_GLOBAL_Y(j,j0,N) \
	for (int j = get_global_id(1)+(j0); j < (N); j += get_global_size(1))

// for i=i0 .. n-1
// tid = local thread id = get_local_id(0) must be initialized by the caller
#define PARALLEL_FOR_LOCAL(tid,i,i0, n) \
	for (unsigned i = (tid)+(i0); i < (n); i += get_local_size(0))


// defines the start of a block which will be executed by a single local thread (thread 0)
#define SINGLE_SECTION_LOCAL if (get_local_id(0)==0)
#define SINGLE_SECTION_GLOBAL if (get_global_id(0)==0)



/* operation macros for the reducers */
#define op_max(x,y) x > y ? x : y
#define op_min(x,y) x < y ? x : y
#define op_add(x,y) ((x) + (y))


/* local reduction. compute 'sum' of elements of array x[0..n-1] in local memory
   The result is stored in x[0].
   the array is overwritten with temporary results
   
   op = reduction operator (op_add, op_min, op_max or other expression)
   x  = vector, in local memory to be reduced
   n  = number of elements of x
   
   A faster variant is generated, based on device SIMD size (aka nvidia warp size = 32 for example)
   using python code, this one is left for reference
   
   */
#define local_reduce_op_safe(op,x,n,tid) \
{ \
	barrier(CLK_LOCAL_MEM_FENCE); \
	for(unsigned int s=(n/2); s>0; s>>=1){ \
        if (tid < s && (tid+s)<n){\
        	x[tid] = op(x[tid], x[tid + s]);\
        }\
        barrier(CLK_LOCAL_MEM_FENCE);\
	}	\
}

//#define local_reduce_op(op,sdata,n,tid) local_reduce_op_safe(op,sdata,n,tid)
_LOCAL_REDUCE_OP_PLACEHOLDER_


/* 
  reduce an operation from global memory, using local memory as temp storage
  map = expression. map(i) should return the i-th element to be reduced  
   op = reduction operator (op_add, op_min, op_max)
    N = total (global) number of elements to reduce
   lx = vector in local memory for temp storage
lsize = size of local vector. Must be equal to workgroup size
  
  If invoked on multiple work-groups, then each group reduces only a portion of the data, 
  and a second stage is needed.
  The result is stored in lx[0].
 */
#define map_reduce_op(map, op, n, lx, lsize, neutral) \
{ \
	unsigned tid = get_local_id(0);\
	unsigned i   = tid+get_group_id(0)*(lsize*2); \
	unsigned gridSize = lsize*2*get_num_groups(0);\
	TYPE mySum=neutral;\
	if (i<n) mySum = map(i); \
	unsigned i2 = i+lsize; \
	if (i2<n) mySum = op(mySum, map(i2));\
	for (i+=gridSize; i<n; i+=gridSize){\
		mySum = op(mySum, map(i));\
		i2 = i+lsize;\
		if (i2<n) mySum = op(mySum, map(i2)); \
	}\
	/* step 2. operate with items n .. N to local if greater */ \
	lx[tid]=mySum; \
	local_reduce_op(op,lx,lsize,tid); \
}


/* parallel compute i = argmax(x) where x is a vector in local memory with n elements
   input ix must be initialized with indices 0...n by the caller
   returns max(x) in x[0] and argmax(x) in ix[0]
   arrays x[] and ix[] are mangled
*/
inline void local_reduce_argmax(__local TYPE * x, __local int * ix, unsigned n){		
	barrier(CLK_LOCAL_MEM_FENCE);
	unsigned i = get_local_id(0);
	for(unsigned int s=(REDUCE_blockSize/2); s>0; s>>=1){
        if (i < s && (i+s)<n){
        	if (x[i+s] > x[i]){
        		x[i] = x[i+s];
        		ix[i] = ix[i+s];
        	}
        }
        barrier(CLK_LOCAL_MEM_FENCE);
	}
}
// the macro allows gx(i) to be an expression instead of a simple array lookup, where i is the index
#define map_reduce_argmax(map, N, lx, lix, lsize) \
{ \
	/* step 1 copy first n items to local */ \
	unsigned tid=get_local_id(0);\
	unsigned lsize1 = min(lsize,N);\
	PARALLEL_FOR_LOCAL(tid,i,0,lsize1){ \
		lx[tid]  = map(i); \
		lix[tid] = i;\
	} \
	/* step 2. copy items n .. N to local if greater */ \
	PARALLEL_FOR_LOCAL(tid,i,lsize1,N){ \
		TYPE x = map(i); \
    	if (x > lx[tid]){ \
    		lx[tid] = x; \
    		lix[tid] = i; \
    	} \
	} \
	/* step 3. argmax in local memory */ \
	local_reduce_argmax(lx, lix, lsize1);\
}

/** matrix address
 * i = row
 * j = column
 * ld = leading dimension (stride)
 */

#define MATRIX(i,j,ld) ((i)*(ld)+(j))

/** 3d matrix, plane-row-column order
 * i = plane  (z)
 * j = row    (y)
 * k = column (x)
 */
#define MATRIX3(i,j,k,ldi,ldj) ((i)*(ldi) + (j)*(ldj) + (k))


// adds to sdata[tid] += gdata[k]
//#define REDUCE_KERNEL_GLOBAL_ADDER_DIAG(mySum,k) mySum += sqr(A[k]-MU[MATRIX(j,k,n)]) * ISIGMA[MATRIX(j,k,n)];
	
#define REDUCE_KERNEL_GLOBAL_ADDER_DIAG(mySum,k) mySum += A[k]

	
	
/** adds to sdata[tid] += 
 * define q_i = a_i - mu_i
 * 
 * 
 *  */
#define REDUCE_KERNEL_GLOBAL_ADDER_MATRIX(mySum,k) { \
	TYPE v_k = 0; \
	for (int _i = 0; _i<n; _i++)\
	  v_k += (A[_i]-MU[MATRIX(j,_i,n)]) * ISIGMA[MATRIX3(j,_i,k, n*n, n)];\
	mySum += v_k * (A[k]-MU[MATRIX(j,k,n)]);\
}



/**
 * multivariate normal PDF
 * @param A: input vector[n]
 * @param MU: mean values, matrix Mxn
 * @param SIGMA: 1/covariances (diagonals only), Mxn
 * @param LOGSF: logarithm to scale factor, added to exp
 * 
 * kernel ndrange dimensions:
 * 		x (0): along features (n)
 * 		y (1): along categories (N)
 * 		z (2): along different input data vectors (used in testing)
 * first stage's output matrix is
 *      P[z][j][group]
 *      	z = plane
 *      	j = row
 *      	group 
 */
__kernel void logpdf_stage_1(
	__global TYPE const * restrict  A,
	int A_offset,int A_stride,
	__global TYPE const * restrict MU,
	__global TYPE const * restrict ISIGMA,
	__global unsigned int  const * restrict j_indices,
	int use_j_indices,
	__global TYPE * P, unsigned N, unsigned ldp_z, unsigned ldp_y, 
	unsigned _n_)
{
	__local  TYPE sdata[REDUCE_blockSize];
	unsigned z = get_global_id(2);
	A += A_offset+z*A_stride;

	unsigned j = use_j_indices ? j_indices[get_global_id(1)] : get_global_id(1);
	
	#define map(i) sqr((A[i]-MU[MATRIX(j,i,_n_)]) * ISIGMA[MATRIX(j,i,_n_)])
	map_reduce_op(map, op_add, _n_, sdata, REDUCE_blockSize, 0);
	#undef map

	SINGLE_SECTION_LOCAL{
		P[MATRIX3(z, j,get_group_id(0),ldp_z, ldp_y)]    = sdata[0];		
	}
}	

/**
 * second stage launches a single horizontal block and N vertical blocks
 * @param P_in (N x np) matrix produced by stage 1
 * @param logsf (N) vector added to P_in, before exponentiation
 * @param P_out (N) vector destination 
 */
__kernel void logpdf_stage_2(
	__global const TYPE * logP_in, // [np]
	__global const TYPE * restrict logS,
	__global const unsigned int * restrict j_indices,
	int use_j_indices,
	__global TYPE * restrict logP_out,	
	int ldp_z, int ldp_y, // strides of 3d matrix logP_in
	int ldp_out,		// stride of output matrix logP_out
	unsigned _n_
)
{
	__local  TYPE sdata[REDUCE_blockSize]; 
	unsigned j 	= use_j_indices ? j_indices[get_global_id(1)] : get_global_id(1);
	unsigned tid = get_local_id(0);
	unsigned z = get_global_id(2);
	PARALLEL_FOR_LOCAL(tid,i,0, REDUCE_blockSize){
		sdata[i] = (i<ldp_y) ?  logP_in[MATRIX3(z,j, i, ldp_z, ldp_y)] : 0;
	}
	local_reduce_op(op_add, sdata, REDUCE_blockSize,tid)
	SINGLE_SECTION_LOCAL{
		logP_out[MATRIX(z,j,ldp_out)]    = -0.5F * LOG_E * sdata[0] -_n_ * HALF_LOG_TWOPI - 0.5F * logS[j];
	}
}	

/**
 * BLAS-1 function
 * This function scales the vector x by the scalar a and overwrites it with the result. 
 * Hence, the performed operation is x [ j ] = a * x [ j ]
 * f a is an device array
 * 
 * multiply vector with scalar(which also in dev memory)
 */
__kernel void scal_dev_kernel(
	__global const TYPE * X, 
	__global TYPE const * restrict A,
	__global TYPE * dst
)
{
	int i=get_global_id(0);
	dst[i] = X[i] * A[0];
}

/**
 * scale by 1/A
 */
__kernel void scal_inv_dev_kernel(
	__global const TYPE * X, 
	__global TYPE const * restrict A,
	__global TYPE * dst
)
{
	int i=get_global_id(0);
	dst[i] = X[i] / A[0];

}

#ifdef __DEBUG__


void printART(){
SINGLE_SECTION_LOCAL{
	printf("predict_kernel input logp=[");
	for (int k=0; k<N; k++) printf(" %g",logp[k]);
	printf("]\n");

	printf("logS=[");
	for (int k=0; k<N; k++) printf(" %g",logS[k]);
	printf("]\n");

	printf("nj=[");
	for (int k=0; k<N; k++) printf(" %g",nj[k]);
	printf("]\n");
}
}
#endif // __DEBUG__


/* to be invoked on a single workgroup
   computes the bayesian prob field using logp and logP_w as prior
   and also computes the arg max */ 
__kernel void predict_kernel(
		const __global TYPE * restrict logp,  // in: log(p(a|w_j)) computed before 
		const __global TYPE * restrict logS,  // in: log(|S[j]|) = logarithm of hypervolumes, used for vigilance test
		const __global TYPE * restrict logP_w,// in: log(P(w_j|a)) prior probs
		int N,
		float logS_MAX,
		__global int * restrict winner_j, // out: output category
		__global TYPE * restrict P_w_a    // out: normalized prob. field P(w_j|a) eq. (3)
){
	__local  TYPE locals[REDUCE_blockSize];
	__local  int lix[REDUCE_blockSize];
	#define map(i) (logS[i] <= logS_MAX ? (logp[i]+logP_w[i]) : MINUS_INF)
	map_reduce_argmax(map, N, locals, lix,REDUCE_blockSize);
	#undef map
	
	int j   = lix[0];    // winner
	TYPE mx = locals[0]; // max(log(P(a|w_j)))	
	barrier(CLK_LOCAL_MEM_FENCE);
	SINGLE_SECTION_LOCAL{
		winner_j[0] = mx > MINUS_INF ? j : -1; // -1 indicates no winner
	}
	// compute sum(exp(log...))
	#define map(i) EXP( (logp[i]>MINUS_INF) ? (logp[i] + logP_w[i] - mx) : MINUS_INF)
	map_reduce_op(map, op_add, N, locals, REDUCE_blockSize, 0);
	#undef map
	barrier(CLK_LOCAL_MEM_FENCE);
	TYPE logsumexp = LOG(locals[0])+mx; // sum(exp(log(p) + log(nj)))
	barrier(CLK_LOCAL_MEM_FENCE);
	unsigned tid=get_local_id(0);
	PARALLEL_FOR_GLOBAL(i,0,N){
		P_w_a[i] = logp[i] > MINUS_INF ? EXP(logp[i] + logP_w[i] - logsumexp) : 0.0F;		
	}
	/*
	 barrier(CLK_LOCAL_MEM_FENCE);
	SINGLE_SECTION_LOCAL{
		printf("predict_kernel result P_w_a=[");
		for (int k=0; k<N; k++) printf(" %g",P_w_a[k]);
		printf("]\n");	
	}
	*/
}


/** increments g_nj[j].
	computes P_w and logP_w from g_nj */

inline void art_update_nj_P_w(
		__global TYPE * restrict g_nj,		//  [N]
		__global TYPE * restrict P_w,		//  [N]
		__global TYPE * restrict logP_w,	//  [N]
		int j,								// winning category
		int N,
		int nsamples)
{
	// update P_w and logP_w
	float fsamples = 1.0F+nsamples;
	PARALLEL_FOR_GLOBAL(jj,0,N){
		if (jj == j){
			TYPE nj = g_nj[jj]+1;       // increment
			P_w[jj]  = nj / fsamples;
			g_nj[jj] = nj;
		} else 			
			P_w[jj]    = g_nj[jj] / fsamples;
		logP_w[jj] = LOG(P_w[jj]);
	}
}

/** increments g_nj[j].
	computes P_w and logP_w from g_nj */
__kernel void art_update_nj_P_w_kernel(
		const __global TYPE * restrict A,  	// 1. [n] in
		int A_offset,              			// 2.
		__global TYPE * restrict g_nj,		//  [N]
		__global TYPE * restrict P_w,		//  [N]
		__global TYPE * restrict logP_w,		//  [N]
		int N,
		int nsamples)
{
	int j = (int)A[A_offset];
	art_update_nj_P_w(g_nj,P_w,logP_w,j,N,nsamples);
}



/** copy the j-th category from mu_tmp,sigma_tmp,logS_tmp to mu, sigma and logS_tmp, also compute inverse sigma and  */ 
__kernel void copy_mu_sigma_kernel(		
		__global TYPE * g_nj,	   // 1. [N]
		__global TYPE * mu,        // 2. [N x n]
		__global TYPE * sigma,     // 3. [N x n]
		__global TYPE * sigi,      // 4. [N x n]
		__global TYPE * logS,      // 5. [N]
		__global TYPE * P_w,       // 6. [N]
		__global TYPE * logP_w,    // 7. [N]
		const __global TYPE * mu_tmp,    // 8. [Nxn] 
		const __global TYPE * sigma_tmp, // 9. [Nxn] 
		const __global TYPE * logS_tmp,  // 10. [N] 
		int j, // 11. winner category number
		int N, // 12. number of categories
		int nsamples, // 13. number of samples seen previously
		int _n_
){
	mu 	  += _n_ * j; // jump to j-th row in the matrices
	sigma += _n_ * j;
	sigi  += _n_ * j;
	mu_tmp+= _n_ * j;
	sigma_tmp += _n_ * j;
	
	// update mu, sigma, sigi vectors
	float nj = g_nj[j];
#ifndef USE_KNUTH_VARIANCE
		nj+=1;
#endif
	PARALLEL_FOR_GLOBAL(i,0,_n_){
		mu[i]    = mu_tmp[i];
		sigma[i] = sigma_tmp[i];
		sigi[i]  = rsqrt(sigma_tmp[i]/nj);
	}
	// update scalar entries
	SINGLE_SECTION_GLOBAL{		
		logS[j] = logS_tmp[j];
	}
	art_update_nj_P_w(g_nj,P_w,logP_w,j,N,nsamples);
}


/** computes temporary mu and sigma values
  to be invoked with a single work-group, with global work size = local work size = REDUCE_blocksize
  */

__attribute__((reqd_work_group_size(REDUCE_blockSize, 1, 1)))
__kernel void update_mu_sigma_kernel(
		const __global TYPE * restrict A,   // 1. [n]
		int A_offset,              // 2.
		const __global TYPE * restrict g_nj,	   // 3. [N]
		const __global TYPE * restrict g_mu,        // 4. [N x n]
		const __global TYPE * restrict g_sigma,     // 5. [N x n]
		const __global TYPE * restrict logS,      // 6. [N]
		__global TYPE * restrict g_mu_tmp,    // 7. [Nxn] out
		__global TYPE * restrict g_sigma_tmp, // 8. [Nxn] out
		__global TYPE * restrict logS_tmp,  // 9. [N] out
		int j0, // 10
		__global unsigned int  const * restrict j_indices, // 11
		int use_j_indices,  // 12
		int Nsel, 			// 13
		int _n_				// 14
)
{
	__local  TYPE localSum[REDUCE_blockSize];
	A += A_offset;
	PARALLEL_FOR_GLOBAL_Y(jj, 0,Nsel)
	{
		unsigned j = use_j_indices ? j_indices[jj+j0] : jj+j0;
		// jump to j-th row in the matrices
		const __global TYPE * mu		= g_mu + _n_ * j; 
		const __global TYPE * sigma		= g_sigma + _n_ * j;
		__global TYPE * mu_tmp			= g_mu_tmp+ _n_ * j;
		__global TYPE * sigma_tmp		= g_sigma_tmp + _n_ * j;		
		float nj = g_nj[j]; // not incremented yet
		PARALLEL_FOR_GLOBAL(i,0,_n_){
			TYPE  a  = A[i];
			TYPE mu_j = mu[i];
			TYPE d_old = a-mu_j;
			mu_j += d_old / (nj+1.0F);
			mu_tmp[i] = mu_j;		
			TYPE d_new = a-mu_j; 
	#ifdef USE_KNUTH_VARIANCE	
			TYPE s = sigma[i] + d_new*d_old;  // modified eq. sigma must be divided by (nj-1) later
	#else		
			TYPE s = sigma[i] + d_new*d_new;  // modified eq. sigma must be divided by (nj-1) later
	#endif		
			sigma_tmp[i] = s; // sigma is variance*nj
		} // for i
	
		barrier(CLK_LOCAL_MEM_FENCE);
	#ifdef USE_KNUTH_VARIANCE
		float njdiv = nj;
	#else 
		float njdiv = nj+1.0f;
	#endif
		// compute logS
		#define map(i) ( sigma_tmp[i] > 0 ? LOG(sigma_tmp[i]/njdiv) : 0)
		map_reduce_op(map, op_add, _n_, localSum, REDUCE_blockSize, 0);
		#undef map		
		SINGLE_SECTION_LOCAL{
			logS_tmp[j] = localSum[0];	
		}
	} // for jj
}


inline TYPE compute_log_sigma_i(TYPE a, TYPE mu, TYPE sigma, float nj){
	TYPE d_old = a-mu;
	mu += d_old / (nj+1.0F);
	TYPE d_new = a-mu; 
#ifdef USE_KNUTH_VARIANCE	
	TYPE s = sigma + d_new*d_old;  // modified eq. sigma must be divided by (nj-1) later
	s /= nj;
#else		
	TYPE s = sigma + d_new*d_new;  // modified eq. sigma must be divided by (nj-1) later
	s /= (nj+1.0F);
#endif	
	return s>0 ? LOG(s) : 0.0f;
}

/** compute the updated value of logS[j] for every j **/
__attribute__((reqd_work_group_size(REDUCE_blockSize, 1, 1)))
__kernel void compute_new_logS_kernel_stage_1(
		const __global TYPE * restrict A,   		// 1. [n]
		int A_offset,              					// 2.
		const __global TYPE * restrict g_nj,		// 3. [N]
		const __global TYPE * restrict g_mu,        // 4. [N x n]
		const __global TYPE * restrict g_sigma,     // 5. [N x n]
		__global TYPE * restrict logS_tmp,  		// 6. [N] out
		const __global unsigned   * restrict j_indices, // 7.
		int use_j_indices,  							// 8.
		int Nsel, 										// 9.
		int ld_dest,
		int _n_
)
{
	__local  TYPE localSum[REDUCE_blockSize];
	A += A_offset;
	PARALLEL_FOR_GLOBAL_Y(jj, 0,Nsel)
	{
		unsigned j = use_j_indices ? j_indices[jj] : jj;
		// jump to j-th row in the matrices
		const __global TYPE * mu		= g_mu + _n_ * j; 
		const __global TYPE * sigma		= g_sigma + _n_ * j;
		float nj = g_nj[j]; // not incremented yet
		
		#define map(i) ( compute_log_sigma_i(A[i], mu[i], sigma[i], nj) )
		map_reduce_op(map, op_add, _n_, localSum, REDUCE_blockSize, 0);
		#undef map		
		SINGLE_SECTION_LOCAL{
			logS_tmp[j*ld_dest + get_group_id(0)] = localSum[0];	
		}
	} // for jj
}

__kernel void compute_new_logS_kernel_stage_2(
	const __global TYPE * logS_tmp_1,				// 1. in
	__global TYPE * logS_tmp,						// 2. out
	const __global unsigned   * restrict j_indices, // 3.
	int use_j_indices,  							// 4.
	int Nsel, 										// 5.
	int ld											// 6.
)
{
	__local  TYPE sdata[REDUCE_blockSize];
	PARALLEL_FOR_GLOBAL_Y(jj, 0,Nsel)
	{
		unsigned j = use_j_indices ? j_indices[jj] : jj;
		int tid = get_local_id(0);
		PARALLEL_FOR_LOCAL(tid,i,0, REDUCE_blockSize){
			sdata[i] = (i<ld) ?  logS_tmp_1[MATRIX(j, i, ld)] : 0;
		}
		local_reduce_op(op_add, sdata, REDUCE_blockSize,tid)
		SINGLE_SECTION_LOCAL{
			logS_tmp[j]    = sdata[0];
		}
	}
}	



__kernel void create_new_category_kernel(
		__global const TYPE * A,   // [n] input vector
		int A_offset,
		__global TYPE * g_nj,	   // [N] category sizes previously
		__global TYPE * mu,        // [N x n]
		__global TYPE * sigma,     // [N x n]
		__global TYPE * sigi,      // [N x n]
		__global TYPE * logS,      // [N]
		__global TYPE * P_w,       // [N]
		__global TYPE * logP_w,    // [N]
		TYPE initSigmaValues,
		int j, // index of new category (row)
		int N, // number of categories (already incremented)
		int nsamples, // number of samples seen previously
		int _n_
)
{
// --------------------------
	mu 	  += _n_ * j; // jump to j-th row in the matrices
	sigma += _n_ * j;
	sigi  += _n_ * j;
	A += A_offset;
	
	PARALLEL_FOR_GLOBAL(i,0,_n_){
		mu[i] 	 = A[i];
		sigma[i] = initSigmaValues;
		sigi[i]  = rsqrt(initSigmaValues);
		TRACE(("%4d gws=%d: create_new_category_kernel i=%d n=%d  a_i=%f\n",
				(int)get_global_id(0), (int)get_global_size(0),i, _n_, mu[i] ));
		
	} // for i
	
	// update j-th category scalars
	SINGLE_SECTION_GLOBAL{
		g_nj[j] = 1.0F; 
		logS[j] = _n_ * LOG(initSigmaValues);
	}
	// update vertical sums
	float fsamples = 1.0F+nsamples;
	PARALLEL_FOR_GLOBAL(jj,0,N){
		TYPE nj = jj==j ? 1.0F : g_nj[jj]; // g_nj[j] was updated previously in global mem, avoid reading hazard
		P_w[jj]    = nj / fsamples;
		logP_w[jj] = LOG(P_w[jj]);
	}
}

/** pseudocode is
 	 compute_new_logS(a,selection=selection)
	 logpdf(a,selection=selection)
	 j=predict_kernel(logS_MAX=logS_MAX)
	 if j<0 then new category(j)
	 else commit(j)
 * 
 */
__attribute__((reqd_work_group_size(REDUCE_blockSize, 1, 1)))
__kernel void art_fit_kernel(
		const __global TYPE * restrict A,  	// 1. [n] in
		int A_offset,              			// 2.
		__global TYPE * restrict g_nj,		// 3. [N] in/out
		__global TYPE * restrict g_mu,      // 4. [N x n] in/out
		__global TYPE * restrict g_sigma,   // 5. [N x n] in/out
		__global TYPE * restrict g_sigi,    // 6. [N x n]
		__global TYPE * restrict logS,      // 7. [N]
		__global TYPE * restrict g_mu_tmp,  // 8. [Nxn] out
		__global TYPE * restrict g_sigma_tmp, // 9. [Nxn] out
		__global TYPE * restrict logS_tmp,    // 10. [N] out
		
		__global TYPE * restrict log_p_a_w, // 11 [N] tmp 
		__global TYPE * restrict logP_w,    // 12 [N] tmp
		__global TYPE * restrict P_w,		// 13 [N] tmp
		__global TYPE * restrict P_w_a,		// sum_w_j[w]/ 14 [N] out: normalized prob. field P(w_j|a) eq. (3)
		__global int * winner_j,			// 15 [1] out: [0] winner category number, [1]=1 if new category was added		
		int j0, 							// 16
		__global unsigned  const * restrict j_indices, // 17
		int use_j_indices,  				// 18
		int Nsel, 							// 19
		int N,					// 20 in/out
		float logS_MAX, 					// 21
		TYPE initSigmaValues,				// 22
		int nsamples,						// 23 number of samples seen previously
		int _n_
){
	winner_j[0] = -1;
	winner_j[1]=0;
	
	TRACE(("%4d: art_fit_kernel Nsel=%d N=%d logS_MAX=%f initSigmaValues=%e nsamples=%d\n",(int)get_global_id(0),Nsel,N,initSigmaValues,logS_MAX,nsamples));
	if (N>0)
	compute_new_logS_kernel_stage_1(
			A,   			// 1. [n]
			A_offset,		// 2.
			g_nj,			// 3. [N]
			g_mu,			// 4. [N x n]
			g_sigma,		// 5. [N x n]
			logS_tmp,		// 6. [N] out
			j_indices,		// 7.
			use_j_indices,  // 8.
			Nsel, 			// 9.
			1,
			_n_
	);


	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	if (N>0)
	logpdf_stage_1(
			A,
			A_offset, 0, 
			g_mu,
			g_sigi,
			j_indices,
			use_j_indices,
			log_p_a_w, Nsel, 1,1,
			_n_);
	
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	if (N>0)
	logpdf_stage_2(
			log_p_a_w,
			logS,
			j_indices,
			use_j_indices,
			log_p_a_w,
			1,1,1,
			_n_);


	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	if (N>0)
	predict_kernel(
			log_p_a_w,  // in: log(p(a|w_j)) computed before 
			allowExtendedVigilance ? logS : logS_tmp,  	// in: log(|S[j]|) = logarithm of hypervolumes, used for vigilance test
			logP_w,		// in: log(P(w_j|a)) prior probs
			Nsel,
			logS_MAX,
			winner_j, // out: output category
			P_w_a    // out: normalized prob. field P(w_j|a) eq. (3)
			);
	
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	int j = winner_j[0];
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	// without the barrier above, it crashes on Intel
	
	TRACE(("%4d: predict j=%d P_w_a[j]=%f\n",(int)get_global_id(0),j,P_w_a[j]));

	if (j < 0){
		j=N;			// index of new category
		
		create_new_category_kernel(
			A,   // [n] input vector
			A_offset,
			g_nj,		// [N] category sizes previously
			g_mu,		// [N x n]
			g_sigma,	// [N x n]
			g_sigi,		// [N x n]
			logS,		// [N]
			P_w,		// [N]
			logP_w,		// [N]
			initSigmaValues,
			j, 			// index of new category (row)
			N+1, 		// number of categories (already incremented)
			nsamples, 	// number of samples seen previously
			_n_
		);
		winner_j[0]=j;
		winner_j[1]=1; // signal new category
	} else {
		update_mu_sigma_kernel(
				A,			// 1. [n]
				A_offset,	// 2.
				g_nj,		// 3. [N]
				g_mu,		// 4. [N x n]
				g_sigma,	// 5. [N x n]
				logS,		// 6. [N]
				g_mu_tmp,	// 7. [Nxn] out
				g_sigma_tmp,// 8. [Nxn] out
				logS_tmp,	// 9. [N] out
				j0, 		// 10
				j_indices,	// 11
				use_j_indices,//12
				Nsel, 			// 13
				_n_
		);
		
		copy_mu_sigma_kernel(
				g_nj,		// 1. [N]
				g_mu,		// 2. [N x n]
				g_sigma,	// 3. [N x n]
				g_sigi,		// 4. [N x n]
				logS,		// 5. [N]
				P_w,		// 6. [N]
				logP_w,		// 7. [N]
				g_mu_tmp,	// 8. [Nxn] 
				g_sigma_tmp,// 9. [Nxn] 
				logS_tmp,	// 10. [N] 
				j,			// 11. winner category number
				N,			// 12. number of categories
				nsamples,	// 13. number of samples seen previously
				_n_
		);

	}

}

__kernel 
void copy_matrix_kernel(__global const TYPE * restrict X, int ldx, 
		__global TYPE * restrict Y,  int ldy, int x_rows, int x_cols){
	int i=get_global_id(0);
	int j=get_global_id(1);
	Y[MATRIX(i,j,ldy)]  = (i<x_rows && j<x_cols) ? X[MATRIX(i,j,ldx)] : 0.0F;
}


__attribute__((reqd_work_group_size(REDUCE_blockSize, 1, 1)))
__kernel void bar_fit_kernel(
		const __global TYPE * restrict A,  	// 1. [n] in
		int A_offset,              			// 2.
		__global TYPE * restrict g_nj,		// 3. [N] in/out
		__global TYPE * restrict g_mu,      // 4. [N x n] in/out
		__global TYPE * restrict g_sigma,   // 5. [N x n] in/out
		__global TYPE * restrict g_sigi,    // 6. [N x n]
		__global TYPE * restrict logS,      // 7. [N]
		__global TYPE * restrict g_mu_tmp,  // 8. [Nxn] out
		__global TYPE * restrict g_sigma_tmp, // 9. [Nxn] out
		__global TYPE * restrict logS_tmp,    // 10. [N] out
		
		__global TYPE * restrict log_p_a_w, // 11 [N] tmp 
		__global TYPE * restrict logP_w,    // 12 [N] tmp
		__global TYPE * restrict P_w,		// 13 [N] tmp
		__global TYPE * restrict P_w_a,		// 14 [N] out: normalized prob. field P(w_j|a) eq. (3)
		__global int * winner_j,			// 15 [1] out: [0] winner j category number, [1]=1 if new category was added
											//             [2] winner k, 
		int j0, 							// 16
		__global unsigned  const * restrict j_indices, // 17
		int use_j_indices,  				// 18
		int Nsel, 							// 19
		int Na,								// 20 in/out
		float logS_MAX, 					// 21
		TYPE initSigmaValues,				// 22
		int nsamples,						// 23 number of samples seen previously
		
		int Na_max,							// 24. max Na allowed
		
		// here comes ART-B much simplified
		const __global TYPE * restrict B,  	// 25. [n] in
		int B_offset,              			// 26
		__global TYPE * restrict g_nj_b,	// 27
		__global TYPE * restrict g_mu_b,	// 28
		int Nb,								// 29
		int Nb_max,							// 30
		// and now, the map field
		int ldw,							// 31 stride of w, P_b_a matrices
		__global TYPE * w,					// matrix of [Na_max x Nb_max]
		__global TYPE * P_b_a,			// matrix of [Na_max x Nb_max]
		__global TYPE * sum_w_j,		// vector of [Na_max]
		int n_a, 
		int n_b,
		TYPE b_min 					// min value of b (category label)
)
{
	SINGLE_SECTION_GLOBAL{
		winner_j[0]=-1; // result j
		winner_j[1]=0;  // signal if new category added in ART_a
		winner_j[2]=-1; // result k -1 means error
		winner_j[3]=Nb;  // out: Nb categories
	}
	// predict k
	TYPE b = B[B_offset];
	
	int k  = round(b-b_min); // classify, b must be an integer category
	if (k<0 || k>=Nb_max) return; // ignore out of bounds
	SINGLE_SECTION_GLOBAL{
		winner_j[2]=k;
	}
	if (k>=Nb){
		Nb=k+1;
		SINGLE_SECTION_GLOBAL{
			winner_j[3]=Nb;
		}
	}
	// predict j
	art_fit_kernel(
			A,A_offset,
			g_nj,		// 3. [N] in/out
			g_mu,		// 4. [N x n] in/out
			g_sigma,	// 5. [N x n] in/out
			g_sigi,		// 6. [N x n]
			logS,		// 7. [N]
			g_mu_tmp,	// 8. [Nxn] out
			g_sigma_tmp,	// 9. [Nxn] out
			logS_tmp,		// 10. [N] out
			
			log_p_a_w, // 11 [N] tmp 
			logP_w,    // 12 [N] tmp
			P_w,		// 13 [N] tmp
			P_w_a,		//sum_w_j[w]/ 14 [N] out: normalized prob. field P(w_j|a) eq. (3)
			winner_j,			// 15 [1] out: [0] winner category number, [1]=1 if new category was added		
			j0, 							// 16
			j_indices, // 17
			use_j_indices,  				// 18
			Nsel, 							// 19
			Na,					// 20 in/out
			logS_MAX, 					// 21
			initSigmaValues,				// 22
			nsamples,						// 23 number of samples seen previously
			n_a
	);
	
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	int j = winner_j[0];
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	
	
	// update mapfield
	SINGLE_SECTION_GLOBAL{
		g_mu_b[k] = b;
		g_nj_b[k] += 1.0f;
		w[MATRIX(j,k, ldw)] += 1.0F;
		sum_w_j[j]		    += 1.0F;
		
	}
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	// normalize row j of the mapfield
	PARALLEL_FOR_GLOBAL(i,0,Nb){
		P_b_a[MATRIX(j,i,ldw)] = w[MATRIX(j,i,ldw)] / sum_w_j[j]; 
	}
	
}

// regression - use full ART-a and ART-b
__attribute__((reqd_work_group_size(REDUCE_blockSize, 1, 1)))
__kernel void bar_fit_regression_kernel(
		const __global TYPE * restrict A,  	// 1. [n] in
		int A_offset,              			// 2.
		__global TYPE * restrict g_nj,		// 3. [N] in/out
		__global TYPE * restrict g_mu,      // 4. [N x n] in/out
		__global TYPE * restrict g_sigma,   // 5. [N x n] in/out
		__global TYPE * restrict g_sigi,    // 6. [N x n]
		__global TYPE * restrict logS,      // 7. [N]
		__global TYPE * restrict g_mu_tmp,  // 8. [Nxn] out
		__global TYPE * restrict g_sigma_tmp, // 9. [Nxn] out
		__global TYPE * restrict logS_tmp,    // 10. [N] out
		
		__global TYPE * restrict log_p_a_w, // 11 [N] tmp 
		__global TYPE * restrict logP_w,    // 12 [N] tmp
		__global TYPE * restrict P_w,		// 13 [N] tmp
		__global TYPE * restrict P_w_a,		// 14 [N] out: normalized prob. field P(w_j|a) eq. (3)
		__global int * winner_j,			// 15 [1] out: [0] winner j category number, [1]=1 if new category was added
											//             [2] winner k, 
		int j0, 							// 16
		__global unsigned  const * restrict j_indices, // 17
		int use_j_indices,  				// 18
		int Nsel, 							// 19
		int Na,								// 20 in/out
		float logS_MAX_a, 					// 21
		TYPE initSigmaValues_a,				// 22
		int nsamples,						// 23 number of samples seen previously
		
		int Na_max,							// 24. max Na allowed
		
		// here comes ART-B 
		const __global TYPE * restrict B,  	// 25. in
		int B_offset,              			// 26
		__global TYPE * restrict g_nj_b,	// 27
		__global TYPE * restrict g_mu_b,	// 28
		__global TYPE * restrict g_sigma_b, // 29
		__global TYPE * restrict g_sigi_b,  // 30
		__global TYPE * restrict logS_b,    // 31
		__global TYPE * restrict g_mu_b_tmp,    // 32. [Nxn] out
		__global TYPE * restrict g_sigma_b_tmp, // 33. [Nxn] out
		__global TYPE * restrict logS_b_tmp,    // 34. [N] out
		
		__global TYPE * restrict log_p_b_w,		// 35 [N] tmp 
		__global TYPE * restrict logP_b_w,		// 36 [N] tmp
		__global TYPE * restrict P_prior_b,		// 37 [N] tmp
		__global TYPE * restrict P_w_b,			// 38 [N] out: normalized prob. field P(w_j|a) eq. (3)
		int Nb,								// 39
		int Nb_max,							// 40
		float logS_MAX_b,					// 41
		TYPE initSigmaValues_b,				// 42
		// and now, the map field
		int ldw,							// 43 stride of w, P_b_a matrices
		__global TYPE * w,			// 44 matrix of [Na_max x Nb_max]
		__global TYPE * P_b_a,		// 45 matrix of [Na_max x Nb_max]
		__global TYPE * sum_w_j,		// 46 vector of [Na_max]
		int n_a,
		int n_b
)
{
	SINGLE_SECTION_GLOBAL{
		winner_j[0]=-1; // result j
		winner_j[1]=0;  // signal if new category added in ART_a
		winner_j[2]=-1; // result k -1 means error
		winner_j[3]=0;  // signal if new category added in ART_b
	}
	// predict j
	art_fit_kernel(
			A,A_offset,
			g_nj,		// 3. [N] in/out
			g_mu,		// 4. [N x n] in/out
			g_sigma,	// 5. [N x n] in/out
			g_sigi,		// 6. [N x n]
			logS,		// 7. [N]
			g_mu_tmp,	// 8. [Nxn] out
			g_sigma_tmp,	// 9. [Nxn] out
			logS_tmp,		// 10. [N] out
			
			log_p_a_w, // 11 [N] tmp 
			logP_w,    // 12 [N] tmp
			P_w,		// 13 [N] tmp
			P_w_a,		//sum_w_j[w]/ 14 [N] out: normalized prob. field P(w_j|a) eq. (3)
			winner_j,			// 15 [1] out: [0] winner category number, [1]=1 if new category was added		
			j0, 							// 16
			j_indices, // 17
			use_j_indices,  				// 18
			Nsel, 							// 19
			Na,					// 20 in/out
			logS_MAX_a, 					// 21
			initSigmaValues_a,				// 22
			nsamples,						// 23 number of samples seen previously
			n_a
	);
	// predict k
	art_fit_kernel(
			B,B_offset,
			g_nj_b,		// 3. [N] in/out
			g_mu_b,		// 4. [N x n] in/out
			g_sigma_b,	// 5. [N x n] in/out
			g_sigi_b,	// 6. [N x n]
			logS_b,		// 7. [N]
			g_mu_b_tmp,	// 8. [Nxn] out
			g_sigma_b_tmp,	// 9. [Nxn] out
			logS_b_tmp,		// 10. [N] out
			log_p_b_w,			// 11 [N] tmp 
			logP_b_w,			// 12 [N] tmp
			P_prior_b,			// 13 [N] tmp
			P_w_b,				// sum_w_j[w]/ 14 [N] out: normalized prob. field P(w_j|a) eq. (3)
			winner_j+2,			// 15 [1] out: [0] winner category number, [1]=1 if new category was added		
			0, 								// j0=0 // 16
			j_indices, 						// 17
			0, 								//use_j_indices,  // 18
			Nb, 							// Nsel=Nb // 19
			Nb,								// 20 in/out
			logS_MAX_b, 					// 21
			initSigmaValues_b,				// 22
			nsamples,						// 23 number of samples seen previously
			n_b
	);
	
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	int j = winner_j[0];
	int k = winner_j[2];
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	
	
	// update mapfield
	SINGLE_SECTION_GLOBAL{
		w[MATRIX(j,k, ldw)] += 1.0F;
		sum_w_j[j]		    += 1.0F;
		
	}
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	// normalize row j of the mapfield
	PARALLEL_FOR_GLOBAL(i,0,Nb){
		P_b_a[MATRIX(j,i,ldw)] = w[MATRIX(j,i,ldw)] / sum_w_j[j]; 
	}
	
}



__kernel void bar_update_mapfield_kernel(
	int j,int k,int Nb,
	int ldw,					// stride of w, P_b_a matrices	
	__global TYPE * w,			// matrix of [Na_max x Nb_max]
	__global TYPE * P_b_a,		// matrix of [Na_max x Nb_max]
	__global TYPE * sum_w_j		// vector of [Na_max]
	)
{
	// update mapfield
	SINGLE_SECTION_GLOBAL{
		w[MATRIX(j,k, ldw)] += 1.0F;
		sum_w_j[j]		    += 1.0F;
		
	}

	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
	// normalize row j of the mapfield
	PARALLEL_FOR_GLOBAL(i,0,Nb){
		P_b_a[MATRIX(j,i,ldw)] = w[MATRIX(j,i,ldw)] / sum_w_j[j]; 
	}

/*	PARALLEL_FOR_GLOBAL(i,0,Na){
		P_b_a[MATRIX(i,k,ldw)] = w[MATRIX(i,k,ldw)] / sum_w_j[j]; 
	}
*/
}
