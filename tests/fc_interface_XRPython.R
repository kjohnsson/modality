Sys.setenv(PYTHONPATH="/usr/local/lib/python2.7/site-packages")
library(XRPython)

data <- randn(1000)
data_trunc <- 0.01*round(100*data)

hartigan_diptest_fc <- PythonFunction("hartigan_diptest_fc", "modality")
hartigan_diptest_fc(data, blurring='none', .get=TRUE)
hartigan_diptest_fc(data_trunc, .get=TRUE)

calibrated_diptest_fc <- PythonFunction("calibrated_diptest_fc", "modality")
calibrated_diptest_fc(data_trunc, 0.05, 'shoulder', .get=TRUE)
