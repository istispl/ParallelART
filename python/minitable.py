# -*- coding: utf-8 -*-
import os.path
import string
import sys
class Formatter:
	"""A simple HTML formatter to write tables.
Note, to render formulas, it uses MathJax, http://docs.mathjax.org/en/latest/tex.html	
so simply use latex equations"""
	def __init__(self,fn,title=None,writeProlog=True, append=False):
		self.fig_width = 8
		self.fig_ctr   = 0
		self.row       = 0
		self.col       = 0
		self.hcol      = 0
		self.fileName  = fn
		# check previous size (if exists), for append
		# always use binary mode to exactly match bytes to characters
		prevSize=0
		if (fn is not None) and os.path.isfile(fn):
			prevSize = os.path.getsize(fn)
		if fn and fn != '-':
			self.writer = open(fn,'r+b' if (append and prevSize>0) else 'wb')
		else:
			self.writer = sys.stdout
		
		if writeProlog and (prevSize==0 or not append):
			self.prolog(title)
	
		# seek back to overwrite previous epilog
		# TODO verify if the portion being overwritten is really the epilog
		if append and not self.writer.isatty():
			if prevSize>len(self.getEpilogStr()) :
				self.writer.seek(-len(self.getEpilogStr()), 2)
			else:
				self.writer.seek(0, 2)
		return
			

	def write(self,x):
		self.writer.write(x)
		return self
	
	def writeln(self,x):
		self.writer.write(x+"\n")
		self.writer.flush()
		return self
	
	def prolog(self,title):
		self.write("""<!DOCTYPE html>
<html>
<head>""")
		if title is not None:
			self.write("<title>"+title+"</title>")
		self.write("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jquery.tablesorter/2.20.1/css/theme.default.css"/>
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery.tablesorter/2.20.1/js/jquery.tablesorter.min.js"></script>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<style type="text/css">
td {
	
}
</style>
</head>
<body>
<script>
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ],
    processEscapes: true,
	processRefs: false
  }
});
</script>
<article>
<header>
""")


		if title is not None:
			self.write("<h1>"+title+"</h1>")
			self.write("<address class='author'><a rel='author' href='#'>Art Vandelay</a></address>")
		self.writeln("</header>")
		return self

	def abstract(self,txt):
		self.writeln(r"<summary>")
		self.writenln(txt)
		self.writeln(r"</summary>")
		return self

# returns the epilog string. this is called also when open-for-append to remove the previous epilog
	def getEpilogStr(self):
		return """
</article>
</body>
</html>"""
		
	def epilog(self):
		self.write(self.getEpilogStr())
		return self
	 
	def section(self,title):
		"""Writes a section header and begins a new section"""
		self.writeln("<h1>%s</h1>"%(self.escape_html(title)))
		return self

	def subsection(self,title):
		"""Writes a subsection header and begins a new section"""
		self.writeln("<h2>%s</h2>"%(self.escape_html(title)))
		return self
	
	def sectione(self):
		self.writer.flush()
		return self

	
	def par(self,txt):
		"""Write a paragraph of text"""
		self.writeln("<p>"+txt+"</p>")
		return self


	def pre(self,txt):
		"""Write a preformatted text (aka verbatim)"""
		self.writeln("<pre>"+txt+"</pre>")
		return self
	
	
	
	def table(self,caption=None):
		self.row=0
		self.col=0
		self.hcol=0			
		self.writeln("<table class='tablesorter' id='myTable'>")
		if caption is not None:
			self.writeln("<caption>"+caption+"</caption>")
		return self
		
	def getTableEpilog(self):
		return """
</tbody>
</table>
<script>
$(document).ready(function() 
    { 
        $("#myTable").tablesorter(); 
    } 
); 
</script>
"""
		
	def tablee(self):
		self.write(self.getTableEpilog())
		return self

# call this function after opening the formatter.
# it tells the formatter to continue an existing table
	def continueTable(self):
		self.writer.flush()
		s=self.getTableEpilog()
		if self.writer.tell() > len(s):
			self.writer.seek(-len(s),1) # seek back from current pos		
			if self.writer.read(len(s)) == s: # ends with </table> ?
				self.writer.seek(-len(s),1) # if yes, seek back
				
		return self
		
	def trh(self):
		self.hcol=0	
		self.writeln("<thead>")
		self.write("<tr>")
		return self
	
	def th(self,x, args=() ):
		self.write("<th>").write(x % args).write("</th>")
		return self
	
	def trhe(self):
		self.writeln("</tr>")	
		self.writeln("</thead>")
		return self

	def tr(self):
		if self.row==0:
			self.writeln("<tbody>")
		self.writeln("<tr>")
		return self 
		
	def tre(self):
		self.row+=1
		self.writeln("\n</tr>")
		return self
		
	# table cell begin
	def tdb(self):
		self.write("<td>")
		return self	
	# table cell end
	def tde(self):
		self.write("</td>")
		return self
	# write value in a table cell (automatically calls tdb(), tde())
	def td(self,x,args=() ):
		self.tdb()
		self.write(x % args)
		self.tde()
		return self
		
	# writes in math mode
	def math(self,x):
		return self.write("$"+x+"$")
		
	def tdm(self,x,args=() ):
		"""Write a table cell in math mode"""
		self.tdb().math(x % args).tde()
		return self
		
	def thm(self,x,args=() ):
		"""Write a table head cell in math mode"""
		self.th("$"+x+"$",args=args)
		return self
	
	def tdn(self,x,args=()):
		"""Write a table cell containing a number"""
		self.td(x,args=args)
		return self
	
	
	def write_imgref(self,imgname,caption=None,label=None):
		"""
		Writes a reference to an image file name 
		"""
		if caption is not None:
			self.writeln("<figure>")
		self.writeln("<img src='%s'/>" % (imgname))
		if caption is not None:
			self.writeln("<figcaption>"+caption+"</figcaption>")
			self.writeln("</figure>")
		return self

	def tee(self,s):
		self.write(s+'\n')
		print(s)
		return self

	def close(self):
		if self.writer is not None:
			self.epilog()
			if self.writer!=sys.stdout:
				self.writer.close()
			self.writer=None
		return

	def escape_html(self,s):
		return s
		
	def math(self,s):
		"""Write a short math expression"""
		self.write('$'+s+'$')
		return self

	def hyperlink(self, url, text):
		return self.write("<a href='%s'>%s</a>" % (url,text))

# definition list
	def dlb(self):
		return self.write("<dl>")
	def dle(self):
		return self.writeln("</dl>")
	def dtb(self):
		return self.write("<dt>")
	def dte(self):
		return self.writeln("</dt>")
	def ddb(self):
		return self.write("<dd>")
	def dde(self):
		return self.writeln("</dd>")
# a row in a def. list. a shortcut 		
	def dlr(self, label, descr):
		return self.dtb().write(label).dte().ddb().write(descr).dde()

# this is a helper method that invokes other primitives
# prints a single line vector, as a single line table
# name is rendered in math mode, values in number mode
	def printVector(self, x, name=None, maxValues=80):
		self.table().tr()
		if name is not None:
			self.tdm(name)
		for v in x:
			self.tdn("%g", (v,))
		self.tre().tablee()
		return self
	
	def __enter__(self):
		return self




# -----------------------------------------------------------------------

class FormatterGnuPlot(Formatter):
	"""A simple text formatter to write tables in gnuplot format"""
	def __init__(self,fn,title=None,writeProlog=True,append=False):
		Formatter.__init__(self,fn,title=title,writeProlog=False,append=append)
	
	
	def prolog(self,title):
		return self

	def abstract(self,txt):
		self.writenln('# ' + txt)
		return self

		
	def getEpilogStr(self):
		return ""
	 
	def section(self,title):
		self.writenln('# ' + title)
		return self

	def subsection(self,title):
		return self
	
	def sectione(self):		
		return self
	def par(self,txt):
		self.writeln("# "+txt)
		return self


	def pre(self,txt):
		self.writeln("# "+txt)
		return self
	
	
	
	def table(self,caption=None):
		self.row=0
		self.col=0
		self.hcol=0		
		if caption is not None:	
			self.writeln("# "+caption)
		return self
		
	def getTableEpilog(self):
		return ""
	
	def trh(self):
		self.hcol=0	
		return self
	
	def th(self,x, args=()):
		self.write((x % args)+"\t")
		return self
	
	def trhe(self):
		self.writeln("")
		return self

	def tr(self):
		return self 
		
	def tre(self):
		self.writeln("")
		return self
	def tdb(self):
		return self
		
	def tde(self):
		self.write("\t")
		return self

	def math(self,x):
		return self.write(x)
		
	def tdn(self,x,args=()):
		"""Write a table cell containing a number"""
		self.td(x,args=args)
		return self
	
	def write_imgref(self,imgname,caption=None):
		"""
		Writes a reference to an image file name.
		imgname should be a relative path to the current output dir 
		caption is an optional caption to be placed under the figure.
		"""
		self.writeln("# image '%s'" % (imgname))
		return self

# -----------------------------------------------------------------------

class FormatterLatex(Formatter):
	def __init__(self,fn,title=None,writeProlog=True,append=False):
		Formatter.__init__(self,fn,title=title,writeProlog=writeProlog,append=append)
		self.tabular=""
		self.header_line=""
		return

	def prolog(self,title):
		self.write(r"""
\documentclass[a4paper]{article}
\usepackage[cm]{fullpage}
%\documentclass[conference,a4paper,twocolumn]{IEEEtran}
\usepackage{graphicx}
\usepackage[cmex10]{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage{placeins}
\begin{document}
""")
		if title is not None:
			self.write(
r"""					
\title{%s}
""" % (self.escape_nonmath(title)))
			
		self.writeln(r"\author{Art Vandelay}")
		self.writeln(r"\maketitle")
		
		return self

	def abstract(self,txt):
		self.writeln(r"\begin{abstract}").writenln(txt).writeln(r"\end{abstract}")
		return self

	def getEpilogStr(self):
		return r"""
\end{document}
"""

	def pre(self,txt):
		"""Write a preformatted text (aka verbatim)"""
		self.writeln("\\begin{verbatim}\n"+txt+"\\end{verbatim}")
		return self

	def table(self,caption=None):
		self.row=0
		self.col=0
		self.hcol=0
		self.writeln(r"\begin{table}[H]")
		if caption is not None:
			self.writeln(r"\centering\caption{" + caption + "}")
		return self
	
	def getTableEpilog(self):
		return r"""\bottomrule
\end{tabular}
\end{table}"""
		
	def tr(self):
		self.col=0 
		return self
	
	def tre(self):
		self.write("\\\\\n")
		return self
		
	def tdb(self):
		if self.col>0: self.write(" & ")
		return self
		
	def tde(self):
		self.col+=1
		return self

	def trh(self):
		self.tabular=""
		self.header_line=""
		self.hcol=0		
		return self
	
	def th(self,x,args=()):
		self.tabular += "l "
		if self.hcol > 0: self.header_line += " & "		
		self.header_line += str(x % args);
		self.hcol += 1
		return self
		
	def thm(self,x,args=() ):
		"""Write a table head cell in math mode"""
		self.th("$"+x+"$",args=args)
		return self
		
	
	def trhe(self):
		self.writeln(r"\begin{tabular}{"+self.tabular+"}")
		self.writeln(r"\toprule")
		self.writeln(self.header_line+r" \\")
		self.writeln(r"\midrule")
		return self
	
	def write_imgref(self,imgname,caption=None,label=None):
		"""
		Writes a reference to an image file name 
		"""
		self.fig_ctr += 1
		if label is None: label=str(self.fig_ctr)
		self.write(r"""
\begin{figure}[h]
\includegraphics[width=%gcm]{%s}
"""% (self.fig_width, imgname))
		if caption is not None: self.writeln(r"\caption{%s}" % (self.escape_nonmath(caption)))
		self.writeln(r"\label{fig:%s}" % label)
		self.writeln(r"\end{figure}")
		return self

	def section(self,title):
		self.write("""
\section{%s}
"""%(self.escape_nonmath(title)))
		return self
		
	def subsection(self,title):
		self.write("""
\subsection{%s}
"""%(self.escape_nonmath(title)))
		return self

		
	def sectione(self):
		self.writeln(r"\FloatBarrier")
		return self
	
	def par(self,txt):
		"""Write a paragraph of text"""
		self.writeln(r"")
		self.writeln(txt)
		return self


	def escape_nonmath(self,s):
		"""Escape characters that are only allowed in math-mode, thus allow s to be displayed in non-math mode"""
		s=string.replace(s,'_','\\_')
		return s
	
	def math(self,s):
		"""Write a short math expression"""
		self.write(r"$" + s + "$")
		return self
		
	def hyperlink(self,  url, text):
		return self.write(r"\href{%s}{%s}" % (url,text))
		
	def printVector(self, x, name=None, maxValues=80):
		tabular=""
		if name is not None: tabular+="l "
		tabular+=" ".join( ("r " for v in x) )
		self.write(r"\begin{tabular}{" + tabular + "}\n")
		if name is not None: self.math(name)
		for v in x:
			self.write("& %.3f"% (v,))
		self.write("\n" + r"\end{tabular}" + "\n")
		return self

		
def new_Formatter(fn,title=None,writeProlog=True,append=False):
	"""A factory method, returns a Formatter instance based on fn extension.
	Recognizes html and tex"""
	if fn is None or fn=='-':
		return FormatterGnuPlot(fn,title=title,writeProlog=writeProlog,append=append)
	
	path,ext = os.path.splitext(fn)
	if    (ext == '.tex') or (ext == '.latex'): return FormatterLatex(fn,title=title,writeProlog=writeProlog,append=append)
	elif  ext == '.txt': return FormatterGnuPlot(fn,title=title,writeProlog=writeProlog,append=append)
	else: return Formatter(fn,title=title,writeProlog=writeProlog,append=append)
	return None
