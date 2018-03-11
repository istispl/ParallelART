"""a profiler for OpenCL, ported from Java"""
import platform
import sys
import numpy
import time
import math
import operator
import minitable

from ordereddict import OrderedDict
class ProfilerData(object):
	def __init__(self, name, WARMUP=0):
		"""
		_n = number of measurements
		d = time, in seconds"""
		self._enabled = True
		self._WARMUP = WARMUP	# set to a negative value to skip some initial samples 
		self._n = self._WARMUP # number of samples accumulated
		self._min = 1E+38
		self._max = 1E-38
		self._timeSum = 0		# cumulative sum of durations
		self._timeAvg = 0		# running average 
		self._timeSum2 = 0		# used for computing online Variance (Knuth's alg)
		self._opsSum  = 0
		self._bytesTransferredSum = 0
		self._gopsMax = 0
		self._gbpsMax = 0
		self._name = name
		self._last = 0
		self._lws = 0
		return

	def add(self, d, ops=0, bytesTransferred=0, lws=None):
		if d <= 0 or ops < 0:
			return  # ignore negative time.
		self._n += 1.0
		self._last = d 
		if self._n > 0:
			if d < self._min:
				self._min = d
			if d > self._max:
				self._max = d
			self._timeSum += d
			
			delta = d - self._timeAvg 
			self._timeAvg += delta/self._n 
			self._timeSum2 += delta * (d - self._timeAvg)
			gops = ops / d
			gbps = bytesTransferred / d
			
			self._opsSum += ops
			self._bytesTransferredSum += bytesTransferred
			if gops > self._gopsMax:
				self._gopsMax = gops
			if gbps > self._gbpsMax:
				self._gbpsMax = gbps
			if lws is not None:
				if isinstance(lws, tuple) or isinstance(lws, list):
					self._lws = reduce(operator.mul, lws)
				else:
					self._lws = lws
		return self

	@property
	def name(self):
		return self._name
	@property
	def count(self):
		return self._n
	
	@property
	def totalTime(self):
		"""total execution time (sum of all samples)"""
		return self._timeSum
	
	@property
	def avg(self):
		"""average execution time, in seconds"""
		return self._timeAvg
	
	@property
	def var(self):
		"""Variance of execution time variance"""
		if self._n > 1:
			return self._timeSum2/(self._n-1.0)
		else:
			return 0
		
		
	@property
	def stddev(self):
		return math.sqrt(self.var)
	
	@property
	def last(self):
		"""Returns the last recorded duration"""
		return self._last
	
	
	
	@property
	def GOPSAvg(self):
		"""returns GOPS/sec, average"""
		if self._timeSum > 0:
			return self._opsSum / self._timeSum / 1E9
		else:
			return 0.0


	
	@property
	def GBPSAvg(self):
		"""returns bandwidth, in GB/sec"""
		if self._timeSum > 0:
			return self._bytesTransferredSum / self._timeSum / 1E9
		else:
			return 0.0


	
	@property
	def GBTotal(self):
		"""returns total number of Gigabytes transferred"""
		return self._bytesTransferredSum * 2**(-30)
	@property
	def MBperCall(self):
		if self._n > 0:
			return self._bytesTransferredSum * 2**(-20) / self._n
		else:
			return 0

	def __str__(self):
		tc=1E3	# time conversion to ms
		return "%20s ncalls=%4d total=%8.3f min=%8.3f max=%8.3f avg=%8.3f std=%8.3f" % (
         self.name, self.count,self._timeSum*tc, self._min*tc, self._max*tc, self.avg*tc, self.stddev*tc)
	
	def to_dict(self, map=None):
		"""Returns a dict with the measured properties."""
		tuconv = 1E3 # time unit converter: sec -> ms
		if map is None: map=OrderedDict()
		map['funcname'] = self._name
		map['exectime_avg'] = self.avg * tuconv
		map['exectime_min'] = self._min * tuconv
		map['exectime_max'] = self._max * tuconv
		map['exectime_dev'] = self.stddev * tuconv
		map['exectime_total'] = self.totalTime * tuconv
		map['GOPS_avg'] = self.GOPSAvg
		map['GOPS_max'] = self._gopsMax
		map['GBPS_avg'] = self.GBPSAvg
		map['GBPS_max'] = self._gbpsMax
		map['GB_total'] = self.GBTotal
		map['MB_per_call'] = self.MBperCall
		map['ncalls'] = self._n
		map['localWorkSize'] = self._lws
		return map
	
	@staticmethod
	def get_ddl():
		ddl=OrderedDict((
		('funcname', 		'varchar(255)'),
		('exectime_avg',	'double'),
		('exectime_min',	'double'), 
		('exectime_max',	'double'),
		('exectime_dev',	'double'),
		('exectime_total',	'double'),
		('GOPS_avg',		'double'),
		('GOPS_max',		'double'),
		('GBPS_avg',		'double'),
		('GBPS_max',		'double'), 
		('GB_total',		'double'),
		('MB_per_call',		'double'),
		('ncalls',			'bigint'),
		('localWorkSize',	'bigint'),
		))	
		return ddl

# ---------- static to access list of profile table methods ----------------------------------------------
# class variable (static variable)
profiler_table = dict()
profiler_enabled=True
def get_profiler_entry(name): 
	global profiler_table,profiler_enabled
	if name in profiler_table:
		return profiler_table.get(name)
	else:
		d = ProfilerData(name)
		profiler_table[name]=d
		return d

def profile(name, d, ops=0, bytesTransferred=0, lws=0):
	""" <summary>
		 main entry point.
		 </summary>
		 <param name="name"> = function (or code section) name being profiled </param>
		 <param name="d"> = execution time, in seconds </param>
		 <param name="problemSize"> = estimative problem size. Computational complexity order.
		   For example, for linear search inside a list of N elements, problemSize=N,
		   for binary search problemSize=log2(N) and so one.
		   The ratio problemSize/d_ns gives an estimate of Giga-Operations/Second </param>
		 <param name="bytesTransferred"> = number of bytes transferred to/from global memory.
		    Used to measure bandwidth. </param>
	"""
	global profiler_table,profiler_enabled
	if not profiler_enabled or name is None:
		return 
	p = get_profiler_entry(name)
	p.add(d, ops=ops, bytesTransferred=bytesTransferred, lws=lws)
	return p

def profile_cl(name, events=None, ops=0, bytesTransferred=0, lws=0):
	global profiler_table,profiler_enabled
	if events is None: return
		
	if not ((type(events) is list) or (type(events) is tuple)):
		raise Exception("Events must be a list")
	else:
		if len(events)<1: return -1
		events[-1].wait()
		
	dur = sum(evt.profile.end - evt.profile.start for evt in events) / 1E9
	profile(name, dur, ops=ops, bytesTransferred=bytesTransferred,lws=lws);
	return dur

def get_sorted_profiler_table():
	s = sorted(profiler_table.values(), reverse=True, key=lambda p:p._timeSum)
	return s


def dump_simple():
	global profiler_table,profiler_enabled
	if not profiler_enabled: return
	s = get_sorted_profiler_table()
	for p in s:
		print p



def dump(fn=None,fmt=None, order=None, desc=False, title=None):
	"""
	 more formatted dump </summary>
	 <param name="order"> = column name to order by, currently timeStum </param>
	 <param name="dir"> = direction 1= ascending -1=descending 
	"""
	global profiler_table,profiler_enabled
	tuconv = 1E3 # time unit converter: sec -> ms
	
	if fmt is None:
		fmt=minitable.new_Formatter(fn,title=title,writeProlog=True)
	
	if title == None:
		title = "Profiler Statistics"
	fmt.table(title)
	fmt.trh()
	fmt.td("%-30s", "Name")
	fmt.td("%7s", "calls")
	fmt.td("%6s", "[%]")
	fmt.td("%10s", "Total[ms]")
	fmt.td("%7s", "[%]")	
	fmt.td("%8s", "Avg[ms]")
	fmt.td("%8s", "Min[ms]")
	fmt.td("%8s", "Max[ms]")
	fmt.td("%8s", "Std[ms]")
	fmt.td("%8s", "avGOP/s")
	fmt.td("%8s", "mxGOP/s")
	fmt.td("%8s", "avgGB/s")
	fmt.td("%8s", "maxGB/s")
	
	fmt.td("%8s", "MB/call")
	fmt.td("%4s", "lws")
	fmt.tre()
	
	sum_n = reduce(lambda x, y: x+y, [p._n for p in profiler_table.values() ])
	sum_t = reduce(lambda x, y: x+y, [p._timeSum for p in profiler_table.values() ])
	
	for p in get_sorted_profiler_table():
			if p._n > 0:
				fmt.tr()
				fmt.td("%-30s", p._name)
				fmt.td("%6d", p._n)				
				fmt.td("%7.2f", p._n*100.0/sum_n)
				fmt.td("%10.3f", p._timeSum * tuconv)
				fmt.td("%7.2f", p._timeSum*100.0/sum_t)
				fmt.td("%8.3f", p.avg * tuconv)
				fmt.td("%8.3f", p._min * tuconv)
				fmt.td("%8.3f", p._max * tuconv)
				fmt.td("%8.3f", p.stddev * tuconv)
				fmt.td("%8.3f", p.GOPSAvg)
				fmt.td("%8.3f", p._gopsMax)
				fmt.td("%8.3f", p.GBPSAvg)
				fmt.td("%8.3f", p._gbpsMax)
				fmt.td("%8.3f", p.MBperCall)
				fmt.td("%4d", p._lws)
				fmt.tre()
	fmt.tablee()


def dump_to_db(session):
	global profiler_table,profiler_enabled
	if session is None or session.id<1: return 
	if not profiler_table: return
	
	tbl = MeasurementTable(session,"ProfilerData", ProfilerData.get_ddl())
	try:
		for name in  profiler_table:
			p = profiler_table.get(name)
			tbl.write(**(p.to_dict()))
	finally:
		tbl.close()
	return
	

def get_db_connection():
	import mysql.connector

# TODO read config dict from file 
	config={
		"host": "main.splash.ro",
		"port": 2202,
		"user": "statistics",
		"password": "stats",
		"database": "statistics",
		"get_warnings": True,
		"autocommit": True,
		"pool_name": "benchmark.pool",
		"pool_size": 3
	}


	db = mysql.connector.connect(**config)

	return db


class MeasurementSession: # 32 or 64 bit # holds the PDB ID # max launched OpenCL work-items (0=unlimited)
	def __init__(self, id=None, host=None, clPlatform=None, clDevice=None, clQueueFlags=None, 
				floatPrecision=None, 
				commandLine=None, programName=None, scriptName=None, problemId=None, clDeviceType=None, clMaxThreads=0,
				remarks=None,params=None,
				durationSec=0.0):
		if host is None:
			host = platform.node()
		if scriptName is None:
			scriptName = sys.argv[0]
		if commandLine is None:
			commandLine = ' '.join(sys.argv)
		self.id = id
		self.host = host
		self.clPlatform = clPlatform
		self.clDevice = clDevice
		self.clQueueFlags = clQueueFlags
		self.floatPrecision = floatPrecision
		self.commandLine = commandLine
		self.programName = programName
		self.scriptName = scriptName
		self.problemId = problemId
		self.clDeviceType = clDeviceType
		self.clMaxThreads = clMaxThreads
		self.remarks=remarks
		self.params = params
		self.durationSec = durationSec
		
	
	def update_duration(self, durationSec):
		self.durationSec = durationSec
		con = get_db_connection()
		try:
			cr = con.cursor()
			# 16 fields
			cr.execute("UPDATE Sessions SET durationSec=%s, problemId=%s, remarks=%s, params=%s WHERE id=%s",
					(durationSec,self.problemId,self.remarks,self.params,self.id))
			con.commit()
		finally:
			con.close()
		return
	
	def writeToDB(self):
		con = get_db_connection()
		print 'Sending session information to mysql at '+con.server_host
		try:
			cr = con.cursor()
			# 16 fields
			cr.execute("""INSERT INTO Sessions \
			(host,clPlatform,clDevice,clQueueFlags,clMaxThreads,osArch,osName,osVersion,javaVersion,\
			numCPUs,maxJavaMemory,\
			programName,problemId,commandLine,scriptName,floatPrecision,remarks,params,durationSec) \
			VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
				(
				self.host,
				self.clPlatform,
				self.clDevice, 
				self.clQueueFlags,
				self.clMaxThreads,
				platform.processor(), # arch
				'numpy',
				numpy.__version__,
				sys.version.split()[0], # python version
				1, #availableProcessors());
				1, #st.setLong(i++, Runtime.getRuntime().maxMemory());
				self.programName,
				self.problemId,
				self.commandLine,
				self.scriptName,
				self.floatPrecision,
				self.remarks,
				self.params,
				self.durationSec))			
			self.id  = cr.lastrowid
			con.commit()
		finally:
			if cr is not None: cr.close()
			con.close()
		return
	
	def writeIdToFile(self, fn):
		with open(fn,'w') as f:
			f.write(str(self.id))
		return
	
	def setSessionNumericAttribute(self, name, value):
		con = get_db_connection()
		try:
			cr = con.cursor()
			# 16 fields
			cr.execute("""INSERT INTO SessionAttrsNumber \
			(sessionid,name,attrValue) \
			VALUES(%s,%s,%s) \
			ON DUPLICATE KEY UPDATE attrValue=%s""",
				(
				self.id,
				name,value,value))			
			con.commit()
		finally:
			if cr is not None: cr.close()
			con.close()
		return
		
		
	
class MeasurementTable:
	def __init__(self,
				session,
				name, ddef):
		
		self.name = name
		self.session = session
		self.cnx = None
		self.cursor = None
		
		# build DDL statement CREATE TABLE...
		ddl="CREATE TABLE IF NOT EXISTS "+name+" (sessionId BIGINT NOT NULL "
		for key, value in ddef.iteritems():
			ddl += ",\n%s %s" % (key,value)
		ddl += ",\n CONSTRAINT %s_fk_1 FOREIGN KEY (sessionId) REFERENCES Sessions(id) ON DELETE CASCADE" % (name)
		ddl += "\n);"
		
		# build SQL INSERT INTO... statement
		sessionId = 0 if self.session is None else self.session.id
		sql = "INSERT INTO "+name+" (sessionId"
		for key, value in ddef.iteritems():
			sql += ",%s" % (key)
		sql+=")\n"
		sql+="VALUES("
		sql+=str(sessionId)
		for key, value in ddef.iteritems():
			sql+=", %("+key+")s"
		sql+=")"
		self.insert_sql=sql
		
		self.cnx = get_db_connection() if session is not None else None
		if self.cnx is not None:
			self.cursor = self.cnx.cursor()
			print "MeasurementTable %s init" % (self.name)
			self.cursor.execute(ddl)
		return 
	
	def write(self, **kwargs):	
		if self.cursor is not None:
			if 0:
				print self.insert_sql,kwargs
			self.cursor.execute(self.insert_sql, kwargs)
		return
	
	def close(self):
		print "MeasurementTable %s close" % (self.name)
		if self.cursor is not None:
			self.cnx.commit()
			self.cursor.close()
			self.cursor=None
		if self.cnx is not None:
			self.cnx.close()
			self.cnx=None
		return

		
