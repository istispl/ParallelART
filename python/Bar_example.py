#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys
import Bar
import config
import sklearn.datasets
import sklearn.metrics


# example usage of Bar classifier
# based on http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
	

if __name__ == '__main__':
	global g_options
	g_options, args,g_ctx,g_queue = config.main_init(module=sys.modules[__name__])

	# load the 'digits' dataset from sklearn
	digits = sklearn.datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	N0 = len(digits.target_names)  # number of categories

	# create a BAR Classifier
	# Note, for optimal results, need to tune the logS_MAX_A, logS_MAX_B, plambda_a, plambda_b hyperparameters
	classifier = Bar.Bar(N0_b=N0,diag=True, regression=False, logS_MAX_A=-2, logS_MAX_B=-2, plambda_a=0.1)
	# We learn the digits on the first half of the digits
	classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

	# Now predict the value of the digit on the second half:
	expected = digits.target[n_samples // 2:]
	predicted = classifier.predict(data[n_samples // 2:])

	print("Classification report for classifier %s:\n%s\n"
      % (classifier, sklearn.metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % sklearn.metrics.confusion_matrix(expected, predicted))


